"""Source grounding, NLI over source chunks, and faithfulness-aware selection."""
import re
from dataclasses import dataclass, asdict


def canonicalize_entity(s):
    s = s.lower().strip()
    s = re.sub(r'^\b(?:a|an|the)\s+', '', s)
    s = re.sub(r'\bper\s*cent\b', '%', s)
    s = re.sub(r'[^\w%]+', ' ', s)
    return ' '.join(s.split())


def entity_precision(source, summary, nlp):
    from newsum.data.pipeline import extract_entities
    found = {canonicalize_entity(ent) for ent, _ in extract_entities(summary, nlp)}
    source_clean = canonicalize_entity(source)
    if not found:
        return 1.0
    supported = sum(bool(re.search(r'(?<!\w)' + re.escape(ent) + r'(?!\w)', source_clean)) for ent in found)
    return supported / len(found)


def entity_recall(reference, summary, nlp):
    from newsum.data.pipeline import extract_entities
    true = {canonicalize_entity(x) for x, _ in extract_entities(reference, nlp)}
    pred = {canonicalize_entity(x) for x, _ in extract_entities(summary, nlp)}
    return len(true & pred) / len(true) if true else 1.0


class NLIScorer:
    def __init__(self, checkpoint='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
                 device=None, max_tokens=400):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()
        self.max_tokens = max_tokens
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entailment = next((v for k, v in labels.items() if 'entail' in k), None)
        if self.entailment is None:
            raise ValueError('Checkpoint does not identify an entailment logit')

    def score(self, source, candidate):
        import torch
        from newsum.data.pipeline import segment_spans
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        chunks = [self.tokenizer.decode(ids[i:i+self.max_tokens], skip_special_tokens=True)
                  for i in range(0, len(ids), self.max_tokens)] or ['']
        sentences = [candidate[a:b].strip() for a, b in segment_spans(candidate)]
        sentences = [x for x in sentences if x]
        if not sentences:
            return 0.0
        scores = []
        for sentence in sentences:
            pairs = self.tokenizer(chunks, [sentence]*len(chunks),
                                   truncation=True, max_length=512, padding=True,
                                   return_tensors='pt').to(self.device)
            with torch.inference_mode():
                probabilities = self.model(**pairs).logits.float().softmax(-1)
            scores.append(float(probabilities[:, self.entailment].max().item()))
        return sum(scores) / len(scores)


@dataclass
class Candidate:
    summary: str
    logprob: float
    faithfulness: float
    entity_precision: float
    final_score: float = 0.0
    eligible: bool = True


def rerank(source, candidates, nli, nlp, beta=0.6, threshold=0.5,
           entity_tolerance=0.2):
    if not candidates:
        raise ValueError('Beam generator did not return any candidate')
    evaluated = [Candidate(text, float(score), nli.score(source, text),
                           entity_precision(source, text, nlp))
                 for text, score in candidates]
    highest_entity = max(x.entity_precision for x in evaluated)
    entity_ok = [x for x in evaluated if x.entity_precision >= highest_entity-entity_tolerance]
    eligible = [x for x in entity_ok if x.faithfulness >= threshold]
    if not eligible:
        eligible = entity_ok or evaluated
    for x in evaluated:
        x.final_score = x.logprob + beta*x.faithfulness
        x.eligible = x in eligible
    chosen = max(eligible, key=lambda x: x.final_score)
    return chosen, [asdict(x) for x in evaluated]
