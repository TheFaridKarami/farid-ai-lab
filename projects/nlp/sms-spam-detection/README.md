# SMS Spam Detection

A deep learning project that classifies SMS messages as spam or legitimate (ham) using Natural Language Processing (NLP).

## Overview

The goal of this project is to build and evaluate a text classification system that identifies spam messages while minimizing false positives.

The project compares classical machine learning and deep learning approaches and provides a reusable inference pipeline for classifying new messages.

## Results

Models were evaluated on a held-out test set of 776 SMS messages.

### Classical Machine Learning — LinearSVC

| Metric | Score |
| --- | ---: |
| Accuracy | 97.81% |
| Precision | 95.51% |
| Recall | 86.73% |
| F1-score | 90.91% |

### Deep Learning — Transformer

| Metric | Score |
| --- | ---: |
| Accuracy | 98.84% |
| Precision | 97.85% |
| Recall | 92.86% |
| F1-score | 95.29% |

The Transformer achieved a higher F1-score than the classical baseline, improving the balance between precision and recall on the test set.

## Models and Approach

### Classical Machine Learning

- TF-IDF for text feature extraction.
- Logistic Regression, Multinomial Naive Bayes, and LinearSVC as baseline classifiers.
- LinearSVC selected as the classical baseline.

### Deep Learning

- Transformer classifier used as the final deep learning model.
- TextVectorization converts messages into numerical sequences for the Transformer.
- RNN, LSTM, and GRU model implementations are included in the project.

### Evaluation

Models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

Particular attention is given to spam precision and recall because false positives and missed spam messages have different consequences.

## Dataset

This project uses the SMS Spam Collection dataset.

After removing duplicate messages, the dataset contains 5,169 messages:

- Ham (legitimate): 4,516
- Spam: 653

The data is divided into training, validation, and test sets using stratified splitting to preserve class proportions. The test set is reserved for final evaluation.

## Project Structure

```text
sms-spam-detection/
├── artifacts/
├── data/
│   ├── raw/
│   └── README.md
├── docs/
│   └── methodology.md
├── models/
│   ├── text_vectorizer.keras
│   └── transformer.keras
├── notebooks/
│   └── sms_spam_detection.ipynb
├── reports/
│   └── figures/
├── scripts/
│   ├── predict.py
│   └── train.py
├── src/
│   ├── models/
│   │   ├── classical.py
│   │   ├── rnn.py
│   │   └── transformer.py
│   ├── config.py
│   ├── data.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── features.py
│   ├── inference.py
│   └── preprocessing.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Usage

### Predict on New Messages

From the project root, run:

```bash
python -m scripts.predict "Congratulations! You have won a free prize."
```

Replace the example message with the SMS you want to classify.

The prediction pipeline returns a predicted class (`spam` or `ham`) and a spam probability.

### Train Models

Run the classical machine learning training script:

```bash
python -m scripts.train
```

The Transformer training workflow is available in the Jupyter notebook.

## Testing

Run the test suite from the project root:

```bash
python -m pytest -v
```

The test suite covers data handling, preprocessing, embeddings, model behavior, evaluation, and inference.

## Limitations

- The dataset is relatively small and primarily contains English SMS messages.
- Real-world spam patterns may differ from those represented in the dataset.
- Model performance may change when messages differ from the training data.
- The default classification threshold may not be optimal for every application.
- High test performance does not guarantee equivalent performance on unseen real-world messages.

## Future Improvements

- Evaluate the model on additional, more recent SMS datasets.
- Investigate classification thresholds based on application requirements.
- Perform further error analysis on false positives and false negatives.
- Develop an API or user interface for practical use.
- Add automated testing and continuous integration.

## Technologies

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Pytest
- Jupyter Notebook

## Author

## Farid Karami

- [GitHub](https://github.com/Farid-Data)
- [LinkedIn](https://www.linkedin.com/in/thefaridkarami/)
