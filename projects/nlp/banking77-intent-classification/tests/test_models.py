import pytest
from configs import load_config
from models.classical import ClassicalModel
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC


@pytest.fixture

def model_config():
    config = load_config("configs/model.yaml")
    return config["classical"]["models"]


@pytest.mark.parametrize(
    
    ("method", "expected_type"),
    
    [
        ("logistic_regression", LogisticRegression),
        ("linear_svc", LinearSVC),
        ("multinomial_nb", MultinomialNB),
        ("complement_nb", ComplementNB),
        ("sgd_classifier", SGDClassifier),
    ],
)
def test_classical_model_creation(model_config, method, expected_type):
    model = ClassicalModel(config=model_config, method=method)

    assert isinstance(model.model, expected_type)
    
    
     
     
     





