
from collections import Counter


def rouge_scores(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    sums = {k: 0.0 for k in ('rouge1', 'rouge2', 'rougeLsum')}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref.replace('. ', '.\n'), pred.replace('. ', '.\n'))
        for key in sums:
            sums[key] += scores[key].fmeasure * 100
    return {key: val/max(1, min(len(predictions), len(references))) for key, val in sums.items()}


def bert_score(predictions, references, device=None):
    from bert_score import score
    _, _, f1 = score(predictions, references, lang='en', model_type='roberta-large',
                      num_layers=17, rescale_with_baseline=False, device=device,
                      verbose=False)
    return float(f1.mean())


def entity_scores(sources, predictions, references, nlp):
    from newsum.inference.reranker import entity_precision, entity_recall
    precision = [entity_precision(x, y, nlp) for x, y in zip(sources, predictions)]
    recall = [entity_recall(z, y, nlp) for z, y in zip(references, predictions)]
    return {'entity_precision': sum(precision)/max(1,len(precision)),
            'entity_recall': sum(recall)/max(1,len(recall))}


def evaluate_optional_factuality(sources, predictions, scorers):
    """Pass preinstalled evaluator objects with a `.score(source, summary)` method.

    Supported adapters are explicit: keys 'AlignScore', 'QAFactEval', 'SummaC',
    'FactCC'. A missing scorer is reported as unavailable, never fabricated.
    """
    result = {}
    for name in ('AlignScore', 'QAFactEval', 'SummaC', 'FactCC'):
        scorer = scorers.get(name)
        if scorer is None:
            result[name] = None
            continue
        vals = [float(scorer.score(x, y)) for x, y in zip(sources, predictions)]
        result[name] = sum(vals)/len(vals) if vals else None
    return result
