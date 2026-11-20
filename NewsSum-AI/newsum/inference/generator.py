
import torch
from newsum.data.pipeline import DATASETS, segment_spans, token_segment_ids
from newsum.inference.reranker import rerank

DECODING = {'cnn_dailymail': (128, 2.0, 3), 'xsum': (64, 1.0, 0),
            'multi_news': (256, 1.5, 3)}


class NewsSummarizer:
    def __init__(self, checkpoint, dataset='cnn_dailymail', device=None, reranking=True):
        from transformers import AutoTokenizer
        from newsum.models.salience_faith_net import SalienceFaithNet
        from newsum.data.pipeline import make_spacy_processors
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = SalienceFaithNet.load_research_checkpoint(checkpoint).to(self.device).eval()
        self.dataset = dataset
        self.segmenter, self.ner = make_spacy_processors(ner=reranking)
        self.nli = None
        if reranking:
            from newsum.inference.reranker import NLIScorer
            self.nli = NLIScorer(device=self.device)

    def summarize(self, source, include_candidates=False):
        from newsum.data.pipeline import normalize
        source = normalize(source)
        max_len = DATASETS[self.dataset][4]
        enc = self.tokenizer(source, truncation=True, max_length=max_len,
                             return_offsets_mapping=True, return_tensors='pt')
        offsets = enc.pop('offset_mapping')[0].tolist()
        segments = token_segment_ids(offsets, segment_spans(source, self.segmenter))
        seg = torch.tensor([segments], device=self.device)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        max_new, penalty, trigram = DECODING[self.dataset]
        generation = dict(num_beams=5, num_return_sequences=5,
                          max_new_tokens=max_new, length_penalty=penalty,
                          no_repeat_ngram_size=trigram,
                          return_dict_in_generate=True, output_scores=True,
                          early_stopping=True)
        with torch.inference_mode():
            output = self.model.generate(inputs=enc['input_ids'],
                                         attention_mask=enc['attention_mask'],
                                         segment_ids=seg, **generation)
        strings = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # HF sequences_scores are length-penalised sequence log likelihoods.
        values = (output.sequences_scores.tolist() if output.sequences_scores is not None
                  else [0.0]*len(strings))
        candidates = list(zip(strings, values))
        use_rerank = self.nli is not None and self.model.use_entity
        if use_rerank:
            selected, diagnostics = rerank(source, candidates, self.nli, self.ner)
            result = selected.summary
        else:
            result = strings[0]
            diagnostics = [{'summary': s, 'logprob': p} for s, p in candidates]
        return {'summary': result,
                **({'candidates': diagnostics} if include_candidates else {})}
