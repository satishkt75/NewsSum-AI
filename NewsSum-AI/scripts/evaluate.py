#!/usr/bin/env python3
"""Compute reference-based metrics from aligned prediction records (JSONL)."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='JSONL with source, reference, prediction fields')
    parser.add_argument('--output', default='outputs/evaluation.json')
    parser.add_argument('--bert-score', action='store_true')
    parser.add_argument('--entity-metrics', action='store_true')
    args = parser.parse_args()
    from newsum.evaluation.metrics import rouge_scores, bert_score, entity_scores
    rows = [json.loads(s) for s in Path(args.predictions).read_text().splitlines() if s.strip()]
    sources = [r['source'] for r in rows]
    preds = [r['prediction'] for r in rows]
    refs = [r['reference'] for r in rows]
    metrics = {'count': len(rows), **rouge_scores(preds, refs)}
    if args.bert_score:
        metrics['BERTScore_F1'] = bert_score(preds, refs)
    if args.entity_metrics:
        from newsum.data.pipeline import make_spacy_processors
        _, nlp = make_spacy_processors(ner=True)
        metrics.update(entity_scores(sources, preds, refs, nlp))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
