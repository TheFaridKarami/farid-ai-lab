import numpy as np
from src.evaluation import (
    evaluate_classical_model,
    evaluate_nlp_model,
)


class MockClassicalModel:
    def predict(self, X):
        return np.array([0, 1, 1, 0])


class MockNLPModel:
    def predict(self, X):
        return np.array([
            [0.1],
            [0.8],
            [0.9],
            [0.2],
        ])



def test_evaluate_classical_model(capsys):
    model = MockClassicalModel()

    X_test = np.array([
        ["a"],
        ["b"],
        ["c"],
        ["d"],
    ])

    y_test = np.array([0, 1, 1, 0])

    evaluate_classical_model(
        model,
        X_test,
        y_test,
        "Test Classical Model",
    )

    captured = capsys.readouterr()

    assert "Test Classical Model" in captured.out
    assert "Classification report" in captured.out
    assert "Confusion Matrix" in captured.out
    assert "Accuracy" in captured.out
    assert "Precision" in captured.out
    assert "Recall" in captured.out
    assert "F1 score" in captured.out


def test_evaluate_nlp_model(capsys):
    model = MockNLPModel()

    X_test = np.array([
        [1, 2, 0],
        [3, 4, 0],
        [5, 6, 0],
        [7, 8, 0],
    ])

    y_test = np.array([0, 1, 1, 0])

    evaluate_nlp_model(
        model,
        X_test,
        y_test,
        "Test NLP Model",
    )

    captured = capsys.readouterr()

    assert "Test NLP Model" in captured.out
    assert "Classification report" in captured.out
    assert "Confusion Matrix" in captured.out
    assert "Accuracy" in captured.out
    assert "Precision" in captured.out
    assert "Recall" in captured.out
    assert "F1 score" in captured.out
    
    
