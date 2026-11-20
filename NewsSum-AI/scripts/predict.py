
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', default='cnn_dailymail', choices=['cnn_dailymail','xsum','multi_news'])
    p.add_argument('--text')
    p.add_argument('--file')
    p.add_argument('--output')
    p.add_argument('--no-rerank', action='store_true')
    args = p.parse_args()
    if bool(args.text) == bool(args.file):
        p.error('Provide exactly one of --text or --file')
    source = Path(args.file).read_text(encoding='utf8') if args.file else args.text
    from newsum.inference.generator import NewsSummarizer
    summarizer = NewsSummarizer(args.checkpoint, args.dataset, reranking=not args.no_rerank)
    result = summarizer.summarize(source, include_candidates=True)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload, encoding='utf8')

if __name__ == '__main__':
    main()
