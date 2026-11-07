import numpy as np
import pytest
from evaluation.evaluator import ClassificationEvaluator


@pytest.fixture
def evaluator():
    return ClassificationEvaluator(class_ids=[0, 1, 2], class_names=["a", "b", "c"])


# ---------- compute_metrics ----------

def test_perfect_predictions_score_one(evaluator):
    metrics = evaluator.compute_metrics([0, 1, 2], [0, 1, 2])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_length_mismatch_raises(evaluator):
    with pytest.raises(ValueError, match="same length"):
        evaluator.compute_metrics([0, 1], [0])


def test_metrics_match_hand_computed_values():
    evaluator = ClassificationEvaluator(class_ids=[0, 1])
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    # class 1: precision 3/4, recall 3/5 | class 0: precision 4/6, recall 4/5
    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.7)
    assert metrics["macro_precision"] == pytest.approx((0.75 + 4 / 6) / 2)
    assert metrics["macro_recall"] == pytest.approx(0.7)
    assert metrics["macro_f1"] == pytest.approx(0.69697, abs=1e-4)


def test_weighted_f1_exceeds_macro_f1_when_rare_class_fails():
    evaluator = ClassificationEvaluator(class_ids=[0, 1])
    y_true = [0] * 8 + [1] * 2
    y_pred = [0] * 10  # rare class 1 is never predicted
    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert metrics["macro_f1"] == pytest.approx(0.4444, abs=1e-4)
    assert metrics["weighted_f1"] == pytest.approx(0.7111, abs=1e-4)
    assert metrics["weighted_f1"] > metrics["macro_f1"]


# ---------- compute_confusion_matrix ----------

def test_confusion_matrix_keeps_shape_when_class_missing(evaluator):
    matrix = evaluator.compute_confusion_matrix([0, 1], [0, 1])  # class 2 absent
    assert matrix.shape == (3, 3)
    assert matrix[2].sum() == 0


def test_confusion_matrix_counts_are_correct(evaluator):
    matrix = evaluator.compute_confusion_matrix([0, 0, 1], [0, 1, 1])
    assert matrix[0, 0] == 1  # true 0 predicted 0
    assert matrix[0, 1] == 1  # true 0 predicted 1
    assert matrix[1, 1] == 1  # true 1 predicted 1


# ---------- build_report ----------

def test_report_dict_uses_class_names(evaluator):
    report = evaluator.build_report([0, 1], [0, 1], as_dict=True)
    assert {"a", "b", "c", "macro avg"} <= report.keys()
    assert report["a"]["precision"] == 1.0
    assert report["c"]["support"] == 0


def test_report_default_is_string(evaluator):
    assert isinstance(evaluator.build_report([0, 1], [0, 1]), str)


# ---------- most_confused_pairs ----------

Y_TRUE = [0, 0, 0, 1, 1, 2]
Y_PRED = [1, 1, 0, 2, 1, 2]


def test_most_confused_pairs_sorted_by_count(evaluator):
    assert evaluator.most_confused_pairs(Y_TRUE, Y_PRED, top_n=2) == [
        ("a", "b", 2),
        ("b", "c", 1),
    ]


def test_most_confused_pairs_respects_top_n(evaluator):
    assert evaluator.most_confused_pairs(Y_TRUE, Y_PRED, top_n=1) == [("a", "b", 2)]


def test_most_confused_pairs_empty_when_no_mistakes(evaluator):
    assert evaluator.most_confused_pairs([0, 1, 2], [0, 1, 2]) == []


def test_most_confused_pairs_ties_are_deterministic(evaluator):
    assert evaluator.most_confused_pairs([0, 1], [1, 0]) == [("a", "b", 1), ("b", "a", 1)]


def test_most_confused_pairs_falls_back_to_class_ids():
    evaluator = ClassificationEvaluator(class_ids=[10, 20])
    assert evaluator.most_confused_pairs([10], [20]) == [(10, 20, 1)]


def test_most_confused_pairs_accepts_numpy_class_names():
    evaluator = ClassificationEvaluator(
        class_ids=[0, 1, 2], class_names=np.array(["a", "b", "c"])
    )
    assert evaluator.most_confused_pairs(Y_TRUE, Y_PRED, top_n=1) == [("a", "b", 2)]