# SMS Spam Detection: Methodology

## 1. Objective

The objective of this project is to classify SMS messages into two categories:

- **Ham:** Legitimate messages.
- **Spam:** Unwanted or fraudulent messages.

The project compares classical machine learning with deep learning to evaluate their performance on SMS spam classification.

## 2. Dataset

The project uses the SMS Spam Collection dataset.

After removing duplicate messages, the dataset contains 5,169 messages:

| Class | Number of messages |
| --- | ---: |
| Ham | 4,516 |
| Spam | 653 |
| **Total** | **5,169** |

Spam represents approximately 12.6% of the dataset, making this an imbalanced binary classification problem.

## 3. Data Preparation

### Duplicate Removal

Duplicate messages were removed to reduce redundancy in the dataset.

The resulting dataset was used for the subsequent splitting and evaluation process.

### Train, Validation, and Test Splits

The dataset was divided into three subsets using stratified splitting with random seed 42.

| Split | Number of messages |
| --- | ---: |
| Training | 3,617 |
| Validation | 776 |
| Test | 776 |
| **Total** | **5,169** |

Stratification preserves approximately the same class proportions across the subsets.

The training set is used to fit models, the validation set supports model selection, and the test set is reserved for final evaluation.

### Text Processing

Text processing depends on the model.

For classical machine learning, messages are converted into numerical features using TF-IDF.

For the Transformer, Keras TextVectorization converts messages into integer sequences that can be processed by the neural network.

Text preprocessing and vocabulary adaptation are performed without using the test set.

## 4. Classical Machine Learning

Classical machine learning models provide a baseline for evaluating the deep learning approach.

The following classifiers were considered:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Classification (LinearSVC)

TF-IDF represents messages using weighted word and token features.

LinearSVC was selected as the classical baseline based on validation performance.

## 5. Deep Learning

A Transformer-based text classifier was used as the final deep learning model.

The pipeline consists of:

1. TextVectorization converts messages into integer sequences.
2. The Transformer processes the sequences to learn contextual relationships between tokens.
3. The classification layer produces a score used to estimate the probability of spam.
4. A classification threshold converts the output into a spam or ham prediction.

The saved model and text vectorizer are used together during inference.

## 6. Evaluation Metrics

The models are evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

### Accuracy

Measures the proportion of all predictions that are correct.

### Precision

Measures how many messages predicted as spam are actually spam.

High spam precision helps reduce legitimate messages incorrectly classified as spam.

### Recall

Measures how many actual spam messages are correctly identified.

High spam recall helps reduce the number of spam messages that pass undetected.

### F1-score

The harmonic mean of precision and recall. It provides a single measure that balances both metrics.

### Confusion Matrix

The confusion matrix shows the counts of correct and incorrect predictions for each class.

Because the dataset is imbalanced, accuracy alone is insufficient for assessing model performance.

## 7. Results

The final evaluation was performed on the held-out test set of 776 messages.

### LinearSVC

| Metric | Score |
| --- | ---: |
| Accuracy | 97.81% |
| Precision | 95.51% |
| Recall | 86.73% |
| F1-score | 90.91% |

Confusion matrix:

| Actual / Predicted | Ham | Spam |
| --- | ---: | ---: |
| Ham | 674 | 4 |
| Spam | 13 | 85 |

### Transformer

| Metric | Score |
| --- | ---: |
| Accuracy | 98.84% |
| Precision | 97.85% |
| Recall | 92.86% |
| F1-score | 95.29% |

Confusion matrix:

| Actual / Predicted | Ham | Spam |
| --- | ---: | ---: |
| Ham | 676 | 2 |
| Spam | 7 | 91 |

### Comparison

The Transformer achieved a higher F1-score than LinearSVC on the test set.

It also produced fewer false positives and false negatives in this evaluation.

These results suggest that the Transformer performed better on this dataset. They do not guarantee the same performance on new data from different sources.

## 8. Inference

The inference pipeline loads the saved Transformer and TextVectorization models.

For a new message, the pipeline:

1. Converts the input text into a numerical sequence.
2. Passes the sequence to the trained Transformer.
3. Obtains a spam probability.
4. Applies the classification threshold to produce a spam or ham label.

This separates prediction from model training and allows the trained model to be reused.

## 9. Limitations

- The dataset is relatively small and primarily contains English SMS messages.
- The class distribution is imbalanced.
- Real-world spam patterns may differ from those in the dataset.
- Performance on the test set may not generalize to every deployment environment.
- The default classification threshold may not be optimal for all applications.

## 10. Future Work

Potential improvements include:

- Evaluating the model on newer SMS datasets.
- Investigating classification thresholds based on application requirements.
- Performing more detailed error analysis.
- Comparing additional model architectures under consistent experimental conditions.
- Packaging the inference pipeline into an API or user interface.

## 11. Conclusion

This project demonstrates an end-to-end SMS spam classification workflow, from data preparation and classical baselines to Transformer-based classification and reusable inference.

On the held-out test set, the Transformer outperformed the LinearSVC baseline across the reported metrics.

The project provides a foundation for further experimentation with NLP models and practical text classification applications.
