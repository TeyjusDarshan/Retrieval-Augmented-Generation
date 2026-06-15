# Retrieval-Augmented Generation

A from-scratch implementation of dense, sparse, and hybrid retrieval systems, built and evaluated on the SQuAD dataset. This project covers the full pipeline: training a bi-encoder retrieval model, building search indices, and evaluating retrieval quality across three retriever architectures.

---

## Overview

This repo implements and compares three retrieval strategies:

- **BM25** — sparse retrieval using the Lucene-style BM25 algorithm
- **Dense Passage Retrieval (DPR)** — dense retrieval using a fine-tuned bi-encoder built on ModernBERT
- **Hybrid** — combines BM25 and DPR scores using Reciprocal Rank Fusion (RRF)

The fine-tuned model is hosted on Hugging Face at [`Teyjus/modernbert-squad-finetuned`](https://huggingface.co/Teyjus/modernbert-squad-finetuned).

---

## Project Structure

```
.
├── training_scripts/
│   └── train.py                  # Bi-encoder training with in-batch negatives
├── Models/
│   └── retreiver_model.py        # DocumentEncoder (mean pooling + L2 norm)
├── Indexer/
│   ├── bm25_indexer.py           # Builds and saves the BM25 sparse index
│   └── dense_indexer.py          # Generates embeddings and stores them in ChromaDB
├── Retriever/
│   ├── retriever.py              # Abstract base class
│   ├── bm25_retriever.py         # BM25 retrieval
│   ├── dense_passage_retriever.py# Dense retrieval via ChromaDB
│   └── hybrid_retreiver.py       # Hybrid retrieval with RRF fusion
├── evaluator/
│   ├── BM25_evaluator.py         # Evaluate BM25 Recall@K
│   ├── DPR_evaluator.py          # Evaluate DPR Recall@K
│   └── hybrid_evauator.py        # Evaluate Hybrid Recall@K
├── Metrics/
│   └── metrics_calculator.py     # Recall@K implementation
├── indices/
│   ├── bm25_wikipedia_index/     # Saved BM25 index files
│   ├── RAG_db_1/                 # ChromaDB store (baseline embeddings)
│   └── RAG_db_2/                 # ChromaDB store (fine-tuned embeddings)
├── upload_to_hub.py              # Upload model checkpoint to Hugging Face Hub
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

The dataset (`rajpurkar/squad`) is loaded automatically from Hugging Face Datasets when you run any script.

---

## Training

The bi-encoder is trained with **in-batch negatives** and **contrastive loss** on the SQuAD training set.

### How it works

- **Base model**: `answerdotai/ModernBERT-base`
- **Architecture**: A single `DocumentEncoder` encodes both questions and passages using mean pooling over the last hidden state, followed by L2 normalization.
- **Loss**: Cross-entropy over a similarity matrix computed as `Q @ C^T / temperature`. For a batch of size B, each question's positive passage is on the diagonal. All other passages in the batch serve as implicit negatives.
- **Temperature**: 0.05 (sharpens the similarity distribution)
- **Optimizer**: AdamW with weight decay of 0.01
- **Scheduler**: Linear warmup for 10% of total steps, then linear decay
- **Early stopping**: Monitors validation loss every 500 steps with patience of 5 evaluations

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `answerdotai/ModernBERT-base` |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Epochs | 3 (with early stopping) |
| Warmup | 10% of total steps |
| Temperature | 0.05 |
| Train/val split | 80/20 from SQuAD train set |
| Eval frequency | Every 500 steps |

### Run training

```bash
python -m training_scripts.train
```

Checkpoints are saved to `./model_weights/best_model_checkpoint/` whenever validation loss improves. The saved checkpoint includes the model weights, tokenizer, optimizer state, and scheduler state.

---

## Building the Indices

Before running retrieval or evaluation, you need to build the search indices.

### BM25 Index

Indexes all unique passages from both the SQuAD train and validation splits using the Lucene BM25 variant with English stemming.

```bash
python -m Indexer.bm25_indexer
```

Output is saved to `./bm25_wikipedia_index/`.

### Dense Index (ChromaDB)

Encodes all unique passages from the SQuAD validation split using a model checkpoint and stores the embeddings in ChromaDB.

```bash
python -m Indexer.dense_indexer \
    --model_name Teyjus/modernbert-squad-finetuned \
    --db_path ./indices/RAG_db_2 \
    --collection_name finetuned_wiki_embeddings \
    --batch_size 4
```

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `./model_weights/best_model_checkpoint` | Local path or Hugging Face model ID |
| `--db_path` | `./indices/RAG_db_2` | Directory for ChromaDB persistent storage |
| `--collection_name` | `finetuned_wiki_embeddings` | Collection name inside ChromaDB |
| `--batch_size` | `4` | Inference batch size |

The maximum context token length is 892, which is the longest passage in the SQuAD dataset.

---

## Evaluation

All evaluators use **Recall@K** as the metric. A query gets a score of 1 if the ground-truth passage appears in the top-K retrieved results, and 0 otherwise. The final score is averaged over the entire SQuAD validation set (~10,570 examples).

### Evaluate BM25

```bash
python -m evaluator.BM25_evaluator --recall_at 10
```

### Evaluate DPR

```bash
python -m evaluator.DPR_evaluator --recall_at 5
```

### Evaluate Hybrid

```bash
python -m evaluator.hybrid_evauator --recall_at 10
```

The `--recall_at` argument accepts any positive integer (common values: 1, 2, 3, 5, 10, 15).

---

## Results

All results are on the SQuAD validation set.

### BM25

| Metric | Score |
|---|---|
| Recall@1 | 0.702 |
| Recall@3 | 0.872 |
| Recall@10 | 0.911 |
| Recall@15 | 0.929 |

### DPR (fine-tuned)

The impact of fine-tuning is significant. Without fine-tuning, the base ModernBERT model achieves Recall@15 of **0.283**. After fine-tuning on SQuAD:

| Metric | Score |
|---|---|
| Recall@3 | 0.763 |
| Recall@10 | 0.901 |
| Recall@15 | 0.930 |

### Hybrid (BM25 + DPR with RRF)

The hybrid retriever consistently outperforms both individual systems by fusing BM25 and DPR rankings using Reciprocal Rank Fusion with k=60.

| Metric | Score |
|---|---|
| Recall@1 | 0.808 |
| Recall@3 | 0.899 |
| Recall@5 | 0.929 |
| Recall@10 | 0.960 |
| Recall@15 | 0.972 |

---

## Uploading to Hugging Face Hub

After training, push the best checkpoint to the Hub:

```bash
python upload_to_hub.py
```

This uploads everything in `./model_weights/best_model_checkpoint/` to `Teyjus/modernbert-squad-finetuned`. You need to be logged in first:

```bash
huggingface-cli login
```

---

## How the Hybrid Retriever Works

The `HybridRetreiver` fetches the top-K results from both BM25 and DPR independently, then merges them using **Reciprocal Rank Fusion (RRF)**:

```
RRF score(doc, rank) = 1 / (k + rank)
```

where `k=60` is a smoothing constant. Documents appearing in both result sets have their RRF scores summed. The final top-K documents are selected from the merged pool by score.

---

## Device Support

All scripts automatically detect and use the best available hardware: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU.
