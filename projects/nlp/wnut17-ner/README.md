# WNUT17 Named Entity Recognition

## Overview

An end-to-end Named Entity Recognition (NER) project using **DistilBERT**, PyTorch, and Hugging Face Transformers.

The objective is to fine-tune a pretrained Transformer model to identify named entities in informal and noisy text, focusing on emerging and uncommon entities.

## Dataset

**WNUT 2017 Emerging and Rare Entities**

The dataset contains six entity categories:

- Corporation
- Creative Work
- Group
- Location
- Person
- Product

Entity annotations follow the BIO (Beginning, Inside, Outside) tagging scheme.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- Google Colab
- DistilBERT

## Methodology

1. Dataset exploration and preprocessing.
2. Subword tokenization and label alignment.
3. DistilBERT fine-tuning for token classification.
4. Model evaluation and error analysis.
5. Inference on unseen text.
6. Model saving and optional Hugging Face Hub publication.

## Evaluation

Model performance will be assessed using:

- Entity-level Precision
- Entity-level Recall
- Entity-level F1-score
- Per-entity F1-score

Results will be compared against published WNUT17 baselines.

## Project Structure

```text
wnut17-ner/
├── notebooks/
├── reports/
│   ├── evaluation/
│   └── error_analysis/
├── artifacts/
├── README.md
└── .gitignore
```

## Development Environment

Google Colab with GPU acceleration.

## Project Status

In Progress
