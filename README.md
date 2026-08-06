# Smart-MCQ Solver Challenge

## Overview

This project explores different models and techniques for **Multiple Choice Question Solving** task. The objective is to train using the `data/train.csv` and predict the **top-3 most likely answers (A–E)** for each question in `data/test.csv`. The models are evaluated using the **MAP@3 (Mean Average Precision @ 3)** metric.

The repository implements and compares the following models:

* TF-IDF + Logistic Regression
* BiLSTM with Attention
* Retrieval-Augmented Generation (RAG)
* Zero-shot Large Language Models (LLMs from HuggingFace)

---

## Models Implemented

### 1. TF-IDF + Logistic Regression

Compares similarity of the question with each answer

**Pipeline**

* Text preprocessing
* TF-IDF feature extraction
* Pairwise dataset construction
* Logistic Regression classifier
* Top-3 prediction ranking

---

### 2. BiLSTM with Attention

A neural sequence model that scores each answer option independently.

**Architecture**

* Hugging Face tokenizer
* Embedding layer
* Bidirectional LSTM
* Attention mechanism
* Layer Normalization
* Fully Connected classifier

Training features include:

* Cross Entropy Loss
* AdamW optimizer
* ReduceLROnPlateau scheduler
* Gradient clipping
* Early stopping

---

### 3. Retrieval-Augmented Generation (RAG)

Knowledge-based retrieval system for MCQ answering.

Pipeline:

1. Knowledge Base construction (`data/train.csv`)
2. SentenceTransformer embeddings (BAAI/bge-base-en-v1.5)
3. FAISS retrieval (FlatIndexIP)
4. Cross-Encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
5. Top-3 answer prediction

---

### 4. Zero-Shot Classification

LLM-based approach without training.

Uses:

* Qwen 2.5
* Prompt-based reasoning (Zero-shot and Few-Shot Prompting)
* Top-3 answer extraction

---

## Evaluation Metrics

The following metrics are implemented:

* MAP@3
* Top-1 Accuracy
* Top-3 Accuracy
* Macro F1 Score

---

## Installation

After cloning the repository, create a virtual environment

```bash
python -m venv env
```

Activate it

Windows

```bash
env\Scripts\activate
```

Linux / macOS

```bash
source env/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Run any desired model out of the four models from the training script and comment out the other three models.

Example:

```bash
python src/train.py
```

---

## Inference

Generate predictions using the trained model.

```bash
python src/inference.py
```

Predictions are exported as:

```text
data/submission.csv
```