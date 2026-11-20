"""Dataset harmonisation, canonical segmentation, and collator."""
import bisect
import hashlib
import re
import unicodedata

DATASETS = {
    'cnn_dailymail': ('ccdv/cnn_dailymail', '3.0.0', 'article', 'highlights', 512, 128),
    'xsum': ('EdinburghNLP/xsum', None, 'document', 'summary', 512, 64),
    'multi_news': ('alexfabbri/multi_news', None, 'document', 'summary', 1024, 256),
}
ENTITY_TYPES = {'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
                'WORK_OF_ART', 'LAW', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL'}


def normalize(text):
    text = unicodedata.normalize('NFKC', str(text))
    return re.sub(r'\s+', ' ', text).strip()


def clean_training_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(?i)\bby\s+(?:daily mail reporter|associated press)\b\s*[-:,.]*', ' ', text)
    return normalize(text)


def make_document(record, dataset_name, training=False):
    _, _, source_key, target_key, _, _ = DATASETS[dataset_name]
    source = str(record[source_key])
    if dataset_name == 'multi_news':
        source = re.sub(r'\s*\|{5}\s*', ' [DOC_SEP] ', source)
    source = clean_training_text(source) if training else normalize(source)
    summary = normalize(record[target_key])
    return source, summary


def segment_spans(text, nlp=None):
    if nlp is not None:
        doc = nlp.make_doc(text)
        if nlp.has_pipe('sentencizer') or nlp.has_pipe('parser') or nlp.has_pipe('senter'):
            # Full processing is needed to set sentence boundaries.
            doc = nlp(text)
            return [(s.start_char, s.end_char) for s in doc.sents]
    # Deterministic fallback, for environments without spaCy sentence segmentation.
    spans = []
    for m in re.finditer(r'[^.!?]+(?:[.!?]+|$)', text):
        if m.group().strip():
            spans.append((m.start(), m.end()))
    return spans or [(0, len(text))]


def token_segment_ids(offsets, spans, max_segment_tokens=64):
    """Align token offset pairs with sentence spans; split spans every 64 tokens."""
    ends = [end for _, end in spans]
    last_sentence = None
    within = 0
    current_segment = -1
    out = []
    for start, end in offsets:
        if end <= start:
            out.append(-1)
            continue
        sent = min(bisect.bisect_right(ends, start), len(spans) - 1)
        if sent != last_sentence or within >= max_segment_tokens:
            current_segment += 1
            within = 0
        out.append(current_segment)
        within += 1
        last_sentence = sent
    return out


def extract_entities(text, nlp):
    if nlp is None:
        return []
    return [(ent.text, ent.label_) for ent in nlp(text).ents if ent.label_ in ENTITY_TYPES]


def source_entity_token_ids(text, nlp, tokenizer):
    ids = set()
    for entity, _ in extract_entities(text, nlp):
        ids.update(tokenizer(entity, add_special_tokens=False)['input_ids'])
        ids.update(tokenizer(' ' + entity, add_special_tokens=False)['input_ids'])
    return sorted(ids)


def encode_record(record, dataset_name, tokenizer, sentencizer=None, ner=None, training=False):
    source, summary = make_document(record, dataset_name, training)
    max_input, max_target = DATASETS[dataset_name][4:]
    enc = tokenizer(source, max_length=max_input, truncation=True,
                    return_offsets_mapping=True)
    offsets = enc.pop('offset_mapping')
    spans = segment_spans(source, sentencizer)
    enc['segment_ids'] = token_segment_ids(offsets, spans)
    enc['labels'] = tokenizer(text_target=summary, max_length=max_target,
                              truncation=True)['input_ids']
    enc['supported_entity_ids'] = source_entity_token_ids(source, ner, tokenizer)
    enc['record_id'] = str(record.get('id', record.get('document_id',
                          hashlib.md5(source.encode()).hexdigest())))
    return enc


class NewsCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        import torch
        max_src = max(len(x['input_ids']) for x in examples)
        max_dst = max(len(x['labels']) for x in examples)
        padded = {'input_ids': [], 'attention_mask': [], 'segment_ids': [],
                  'labels': [], 'supported_entity_ids': [], 'record_id': []}
        for sample in examples:
            n = max_src - len(sample['input_ids'])
            t = max_dst - len(sample['labels'])
            padded['input_ids'].append(sample['input_ids'] + [self.tokenizer.pad_token_id] * n)
            padded['attention_mask'].append(sample['attention_mask'] + [0] * n)
            padded['segment_ids'].append(sample['segment_ids'] + [-1] * n)
            padded['labels'].append(sample['labels'] + [-100] * t)
            padded['supported_entity_ids'].append(sample.get('supported_entity_ids', []))
            padded['record_id'].append(sample.get('record_id', ''))
        for key in ('input_ids', 'attention_mask', 'segment_ids', 'labels'):
            padded[key] = torch.tensor(padded[key], dtype=torch.long)
        return padded


def load_news_split(name, split, limit=None, seed=42):
    from datasets import load_dataset
    hub, config, _, _, _, _ = DATASETS[name]
    dataset = load_dataset(hub, name=config, split=split, trust_remote_code=False)
    if split == 'train':
        # Skip duplicates only on training data. Validation/test are left untouched.
        seen = set()
        unique_indices = []
        for index, record in enumerate(dataset):
            document, summary = make_document(record, name, training=True)
            if not document or not summary:
                continue
            digest = hashlib.md5(document.encode('utf8')).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            unique_indices.append(index)
        dataset = dataset.select(unique_indices)
    if limit is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(int(limit), len(dataset))))
    return dataset


def make_spacy_processors(ner=True):
    import spacy
    segmenter = spacy.blank('en')
    segmenter.add_pipe('sentencizer')
    entity_model = None
    if ner:
        try:
            entity_model = spacy.load('en_core_web_trf')
        except OSError as exc:
            raise RuntimeError('Install spaCy NER: python -m spacy download en_core_web_trf') from exc
    return segmenter, entity_model
