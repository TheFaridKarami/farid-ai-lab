import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


class ClassificationEvaluator:
    def __init__(self, class_ids=None, class_names=None):
        self.class_ids = class_ids
        self.class_names = class_names

    def compute_metrics(self, y_true, y_pred) -> dict:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.class_ids, average="macro", zero_division=0
        )
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.class_ids, average="weighted", zero_division=0
        )
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    def build_report(self, y_true, y_pred, as_dict=False):
        return classification_report(
            y_true,
            y_pred,
            labels=self.class_ids,
            target_names=self.class_names,
            zero_division=0,
            output_dict=as_dict,
        )

    def compute_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=self.class_ids)

    def most_confused_pairs(self, y_true, y_pred, top_n=10):
        confusion_counts = self.compute_confusion_matrix(y_true, y_pred).copy()
        num_classes = len(confusion_counts)
        display_names = self._resolve_display_names(num_classes)

        np.fill_diagonal(confusion_counts, 0)  # ignore correct predictions
        flat_counts = confusion_counts.ravel()
        sorted_indices = np.argsort(-flat_counts, kind="stable")[:top_n]

        pairs = []
        for flat_index in sorted_indices:
            count = int(flat_counts[flat_index])
            if count == 0:
                break
            true_row, predicted_col = divmod(flat_index, num_classes)
            pairs.append((display_names[true_row], display_names[predicted_col], count))
        return pairs

    def _resolve_display_names(self, num_classes):
        if self.class_names is not None:
            return self.class_names
        if self.class_ids is not None:
            return self.class_ids
        return list(range(num_classes))
    
