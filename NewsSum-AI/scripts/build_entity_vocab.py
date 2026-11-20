#!/usr/bin/env python3
"""Build the corpus-level summary-entity subword vocabulary from training split only."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['cnn_dailymail','xsum','multi_news'])
    p.add_argument('--backbone', default='facebook/bart-large')
    p.add_argument('--max-records', type=int, default=None)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    from newsum.data.pipeline import load_news_split, DATASETS, extract_entities, make_spacy_processors
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    _, ner = make_spacy_processors(ner=True)
    ds = load_news_split(args.dataset, 'train', args.max_records)
    ids = set()
    for i, row in enumerate(ds):
        for entity, _ in extract_entities(row[DATASETS[args.dataset][3]], ner):
            ids.update(tokenizer(entity, add_special_tokens=False)['input_ids'])
            ids.update(tokenizer(' '+entity, add_special_tokens=False)['input_ids'])
        if (i + 1) % 1000 == 0:
            print(f'{i+1}/{len(ds)} records scanned', flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(sorted(ids)))
    print(f'Saved {len(ids)} entity token IDs to {args.output}')

if __name__ == '__main__':
    main()
