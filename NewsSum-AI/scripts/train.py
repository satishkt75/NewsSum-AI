
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cnn_dailymail','xsum','multi_news'], default='cnn_dailymail')
    parser.add_argument('--variant', choices=['baseline','salience','faith','salience_faith'], default='salience_faith')
    parser.add_argument('--backbone', default='facebook/bart-large')
    parser.add_argument('--output', default='outputs/training')
    parser.add_argument('--limit-train', type=int)
    parser.add_argument('--limit-val', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--entity-vocab', default=None)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from newsum.data.pipeline import (load_news_split, encode_record, make_spacy_processors,
                                      DATASETS)
    from newsum.models.salience_faith_net import SalienceFaithNet
    from newsum.training.engine import train
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[DOC_SEP]']})
    segmenter, ner = make_spacy_processors(ner=args.variant in ('faith','salience_faith'))
    if args.resume:
        restored = Path(args.resume).parent / 'last'
        if not restored.exists():
            raise SystemExit(f'Cannot resume: missing model checkpoint {restored}')
        model = SalienceFaithNet.load_research_checkpoint(restored)
    else:
        model = SalienceFaithNet.from_pretrained(args.backbone)
        model.resize_token_embeddings(len(tokenizer))
    model.configure_variant(args.variant)
    if args.variant in ('faith','salience_faith'):
        if args.entity_vocab is None:
            raise SystemExit('Entity-aware training requires --entity-vocab (create using scripts/build_entity_vocab.py).')
        model.entity_vocab_ids = json.loads(Path(args.entity_vocab).read_text())
        if not model.entity_vocab_ids:
            raise SystemExit('Entity vocabulary is empty; cannot train entity-aware model.')
    print(f'Preparing {args.dataset} training data', flush=True)
    train_ds = load_news_split(args.dataset, 'train', args.limit_train, args.seed)
    val_ds = load_news_split(args.dataset, 'validation', args.limit_val, args.seed)
    def encode(ds, training):
        return [encode_record(r, args.dataset, tokenizer, segmenter, ner, training=training)
                for r in ds]
    training_examples = encode(train_ds, True)
    validation_examples = encode(val_ds, False)
    path = Path(args.output) / args.dataset / args.variant / f'seed_{args.seed}'
    path.mkdir(parents=True, exist_ok=True)
    (path / 'run_args.json').write_text(json.dumps(vars(args), indent=2))
    saved = train(model, tokenizer, training_examples, validation_examples, path,
                  epochs=args.epochs, batch_size=args.batch_size,
                  accumulation=args.accumulation, seed=args.seed,
                  max_train_steps=args.max_steps, resume=args.resume)
    print(f'Best checkpoint: {saved}')

if __name__ == '__main__':
    main()
