import pytest
from configs import load_config
from data.data import prepare_data, split_data
from features.text import TextVectorizer
from models.classical import ClassicalModel
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from training.trainer import Trainer


@pytest.fixture
def config():
    return load_config("configs/model.yaml")


@pytest.fixture
def trainer(config):
    return Trainer(config)


@pytest.fixture
def sample_texts():
    X_train = [
        "check account balance",
        "check account balance please",
        "transfer money now",
        "transfer money today",
    ]

    X_val = [
        "check account balance",
        "transfer money",
    ]

    return X_train, X_val


def test_trainer_initialization(trainer, config):
    assert isinstance(trainer, Trainer)
    assert trainer.config == config["classical"]


def test_extract_features(trainer, sample_texts):
    X_train, X_val = sample_texts

    vectorizer, X_train_features, X_val_features = (
        trainer.extract_features(
            X_train,
            X_val,
            method="bow",
        )
    )

    assert isinstance(vectorizer, TextVectorizer)
    assert X_train_features.shape[0] == len(X_train)
    assert X_val_features.shape[0] == len(X_val)
    assert X_train_features.shape[1] > 0
    assert X_train_features.shape[1] == X_val_features.shape[1]


def test_extract_features_rejects_invalid_method(
    trainer,
    sample_texts,
):
    X_train, X_val = sample_texts

    with pytest.raises(ValueError):
        trainer.extract_features(
            X_train,
            X_val,
            method="invalid_method",
        )


def test_train_model(trainer):
    X_train_features = csr_matrix([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])

    y_train = [
        "balance",
        "balance",
        "transfer",
        "transfer",
    ]

    model = trainer.train_model(
        X_train_features,
        y_train,
        method="logistic_regression",
    )

    assert isinstance(model, ClassicalModel)
    assert isinstance(model.model, LogisticRegression)

    predictions = model.model.predict(X_train_features)

    assert len(predictions) == len(y_train)
    assert set(predictions).issubset(set(y_train))


def test_training_workflow_with_real_data(trainer):
    train_df, _ = prepare_data()

    X_train, X_val, y_train, y_val = split_data(train_df)

    _, X_train_features, X_val_features = trainer.extract_features(
        X_train,
        X_val,
        method="bow",
    )

    model = trainer.train_model(
        X_train_features,
        y_train,
        method="logistic_regression",
    )

    predictions = model.model.predict(X_val_features)

    assert len(predictions) == len(y_val)
    assert set(predictions).issubset(set(y_train))