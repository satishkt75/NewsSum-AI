[![DOI](https://zenodo.org/badge/1465004259.svg)](https://doi.org/10.5281/zenodo.24387757)

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/satishkt75/NewsSum-AI)

# NewsSum-AI: SalienceFaithNet

**NewsSum-AI** is a faithfulness-aware artificial intelligence framework for abstractive summarisation of English news using salience planning and entity consistency.

The framework integrates a BART-large backbone with a learned sentence/segment salience planner, decoder-layer gated conditioning, entity-consistency modelling, cross-attention coverage regularisation, and DeBERTa-v3-based natural language inference (NLI) reranking.

It supports three benchmark datasets: **CNN/DailyMail, XSum, and Multi-News**, with four experimental variants and three random seeds (42, 1234, and 2025).

## 1. Key Features

- **BART-large Backbone:** Transformer-based abstractive news summarisation.
- **Salience Planning:** Identifies and prioritises important source sentences and segments.
- **Gated Decoder Conditioning:** Integrates salience representations into decoder layers.
- **Entity Consistency:** Encourages summaries to retain source-supported entities.
- **Coverage Regularisation:** Reduces redundant information through decoder cross-attention.
- **NLI-Based Reranking:** Uses DeBERTa-v3 to assess candidate summary entailment.
- **Multi-Dataset Support:** Supports CNN/DailyMail, XSum, and Multi-News.
- **Ablation Studies:** Provides four experimental configurations.
- **Statistical Evaluation:** Supports statistical significance analysis.

## 2. System Requirements

- Python 3.10 or 3.11
- Linux or Windows operating system
- CUDA-compatible NVIDIA GPU recommended
- Minimum 16 GB GPU memory for reduced batch configurations
- Internet connection for downloading datasets and pretrained models

For large-scale experiments, multiple high-memory GPUs are recommended.

## 3. Installation

Clone the repository:

```bash
git clone https://github.com/satishkt75/NewsSum-AI
cd NewsSum-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Install the spaCy language model:

```bash
python -m spacy download en_core_web_trf
```

Install a compatible PyTorch version with CUDA support by following the official instructions:

https://pytorch.org/get-started/locally/

## 4. Supported Datasets

| Dataset | Description |
|---|---|
| CNN/DailyMail | News articles with reference summaries |
| XSum | Extreme summarisation dataset |
| Multi-News | Multi-document news summarisation dataset |

The framework supports dataset-specific preprocessing, tokenisation, sentence segmentation, and summarisation configurations.

## 5. Model Variants

The framework provides four experimental configurations:

| Variant | Description |
|---|---|
| Baseline | BART-large summarisation |
| Salience | Salience-guided summarisation |
| Faith | Entity-consistency and coverage objectives |
| SalienceFaith | Combined salience planning and faithfulness-aware learning |

Three random seeds are used for experimental consistency:

`42, 1234, 2025`

## 6. Training

### Baseline Training

Run a baseline training experiment using XSum:

```bash
python scripts/train.py --dataset xsum --variant baseline \
  --limit-train 32 --limit-val 8 \
  --batch-size 1 --accumulation 2 \
  --max-steps 2 --output outputs/smoke
```

### Entity Vocabulary Construction

Build the entity-token vocabulary using training summaries:

```bash
python scripts/build_entity_vocab.py --dataset xsum \
  --max-records 500 \
  --output outputs/xsum_entity_vocab.json
```

### SalienceFaith Training

```bash
python scripts/train.py --dataset xsum --variant salience_faith \
  --entity-vocab outputs/xsum_entity_vocab.json \
  --limit-train 32 --limit-val 8 \
  --batch-size 1 --accumulation 2 \
  --max-steps 2 --output outputs/smoke
```

The above commands use reduced datasets and training steps for initial testing. For complete experiments, use the full training datasets and the configurations defined in `configs/datasets.yaml`.

## 7. News Summarisation and Inference

Generate a summary using a trained checkpoint:

```bash
python scripts/predict.py \
  --checkpoint outputs/smoke/xsum/salience_faith/seed_42/best \
  --dataset xsum \
  --file examples/sample_news.txt \
  --output outputs/sample.json
```

For faster inference without NLI-based reranking, use:

```bash
--no-rerank
```

The inference pipeline supports candidate generation, entity-based analysis, and NLI-based reranking.

## 8. Model Evaluation

Generate predictions on the test dataset:

```bash
python scripts/batch_predict.py \
  --checkpoint /path/to/best \
  --dataset xsum \
  --split test \
  --limit 100 \
  --output outputs/xsum_test.jsonl
```

Evaluate the generated predictions:

```bash
python scripts/evaluate.py \
  --predictions outputs/xsum_test.jsonl \
  --output outputs/xsum_metrics.json \
  --bert-score \
  --entity-metrics
```

The evaluation framework supports:

- ROUGE scores
- BERTScore
- Entity precision and consistency metrics
- Statistical significance analysis

## 9. Ablation Studies and Cross-Dataset Evaluation

Run ablation experiment commands:

```bash
python scripts/run_ablations.py \
  --dataset xsum \
  --entity-vocab outputs/xsum_entity_vocab.json
```

Run cross-dataset evaluation:

```bash
python scripts/run_cross_dataset.py \
  --cnn-checkpoint /path/to/cnn/best \
  --xsum-checkpoint /path/to/xsum/best
```

These scripts provide the experiment commands for the respective configurations.

## 10. Model Architecture

The NewsSum-AI framework consists of the following major components:

1. **Input Processing:** Processes news articles using tokenisation and sentence segmentation.
2. **BART Encoder:** Generates contextual representations of the input news articles.
3. **Salience Planner:** Computes sentence-level and segment-level importance scores.
4. **Gated Decoder Conditioning:** Integrates the salience planning representation into decoder layers.
5. **Entity Consistency Module:** Models source-supported entity information during training.
6. **Coverage Regularisation:** Uses decoder cross-attention to encourage comprehensive and less redundant summaries.
7. **Summary Generation:** Generates candidate summaries through beam search.
8. **NLI-Based Reranking:** Uses DeBERTa-v3 to evaluate and rank candidate summaries.

## 11. Project Structure

```text
NewsSum-AI/
│
├── configs/
│   └── datasets.yaml
│
├── examples/
│   └── sample_news.txt
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── batch_predict.py
│   ├── evaluate.py
│   ├── build_entity_vocab.py
│   ├── run_ablations.py
│   └── run_cross_dataset.py
│
├── src/
│   └── newsum/
│       ├── models/
│       ├── data/
│       ├── training/
│       ├── inference/
│       └── evaluation/
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 12. Output Structure

Training checkpoints and experiment outputs are organised by dataset, model variant, and random seed.

```text
outputs/
│
├── training/
│   └── xsum/
│       └── salience_faith/
│           └── seed_42/
│               └── best/
│
├── smoke/
│
├── xsum_entity_vocab.json
│
├── sample.json
│
├── xsum_test.jsonl
│
└── xsum_metrics.json
```

## 13. Testing

Run the project test suite:

```bash
pytest -q
```

The tests cover:

- Salience planner functionality and gradient flow
- Tensor shapes and normalisation
- Research loss calculations
- Tokenizer-offset sentence segmentation
- Entity-based reranking
- Statistical evaluation routines

## 14. Acknowledgements

This project builds upon the following pretrained models and benchmark resources:

- BART-large
- DeBERTa-v3
- CNN/DailyMail
- XSum
- Multi-News

## 15. Citation

If you use NewsSum-AI in your research, please cite the project and acknowledge the pretrained models and datasets used in your work.
