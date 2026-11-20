
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', required=True, choices=['cnn_dailymail','xsum','multi_news'])
    p.add_argument('--split', default='test')
    p.add_argument('--limit', type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--no-rerank', action='store_true')
    args = p.parse_args()
    from newsum.data.pipeline import load_news_split, make_document
    from newsum.inference.generator import NewsSummarizer
    ds = load_news_split(args.dataset, args.split, args.limit)
    service = NewsSummarizer(args.checkpoint, args.dataset, reranking=not args.no_rerank)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as f:
        for i, row in enumerate(ds):
            source, ref = make_document(row, args.dataset, training=False)
            output = service.summarize(source)
            f.write(json.dumps({'id':i,'source':source,'reference':ref,
                                'prediction':output['summary']}, ensure_ascii=False)+'\n')
            if (i+1) % 20 == 0:
                print(i+1, flush=True)

if __name__ == '__main__':
    main()
