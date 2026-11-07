"""Tests for training/neural_trainer.py.

Run from the repository root so "configs/model.yaml" resolves:
    pytest tests/test_neural_trainer.py -v
"""

import copy

import numpy as np
import pytest
import torch
import training.neural_trainer as trainer_module
from models.neural import (
    EmbeddingBagMLP,
    GRUClassifier,
    LSTMClassifier,
    RNNClassifier,
)
from torch.utils.data import RandomSampler, SequentialSampler
from training.neural_trainer import NeuralTrainer

MODEL_CLASSES = {
    "embedding_bag_mlp": EmbeddingBagMLP,
    "rnn": RNNClassifier,
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
}
ALL_MODELS = list(MODEL_CLASSES)
RECURRENT_MODELS = ["rnn", "lstm", "gru"]


# --------------------------------------------------------------- helpers


def snapshot(model):
    """Copy of all weights, detached from the live model."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------- fixtures


@pytest.fixture
def sample_data():
    X_train = [
        "check account balance",
        "check account balance please",
        "transfer money now",
        "transfer money today",
        "freeze my card",
        "block my card please",
    ]
    X_val = ["check account balance", "transfer money", "freeze my card"]
    y_train = ["balance", "balance", "transfer", "transfer", "card", "card"]
    y_val = ["balance", "transfer", "card"]
    return X_train, X_val, y_train, y_val


@pytest.fixture
def trainer(monkeypatch, sample_data):
    """A trainer wired to the tiny dataset: no disk or network access."""
    monkeypatch.setattr(trainer_module, "prepare_data", lambda: (None, None))
    monkeypatch.setattr(trainer_module, "split_data", lambda _df: sample_data)

    t = NeuralTrainer("configs/model.yaml")
    t.config = copy.deepcopy(t.config)  # tests may edit it freely
    t.config["batch_size"] = 2
    t.config["epochs"] = 2
    t.config["patience"] = None
    return t


@pytest.fixture
def prepared(trainer):
    trainer.prepare_datasets()
    return trainer


# --------------------------------------------------------- 1. initialization


def test_trainer_initialization():
    t = NeuralTrainer("configs/model.yaml")

    assert t.model is None
    assert t.optimizer is None
    assert t.tokenizer is None
    assert t.train_loader is None
    assert t.val_loader is None


def test_fit_requires_prepare_datasets(trainer):
    with pytest.raises(RuntimeError, match="prepare_datasets"):
        trainer.fit("gru")


# ------------------------------------------------------ 2. dataset preparation


def test_prepare_datasets_shapes_and_dtypes(prepared, sample_data):
    X_train, X_val, _, _ = sample_data
    seq_len = prepared.config["sequence_length"]

    train_x, train_y = next(iter(prepared.train_loader))
    val_x, val_y = next(iter(prepared.val_loader))

    for features in (train_x, val_x):
        assert features.ndim == 2
        assert features.shape[1] == seq_len
        assert features.dtype == torch.long
    for labels in (train_y, val_y):
        assert labels.ndim == 1
        assert labels.dtype == torch.long

    assert len(prepared.train_loader.dataset) == len(X_train)
    assert len(prepared.val_loader.dataset) == len(X_val)


def test_prepare_datasets_encodes_labels(prepared):
    assert list(prepared.label_encoder.classes_) == ["balance", "card", "transfer"]


def test_train_loader_shuffles_and_val_loader_does_not(prepared):
    assert isinstance(prepared.train_loader.sampler, RandomSampler)
    assert isinstance(prepared.val_loader.sampler, SequentialSampler)


# ------------------------------------------------------ 3. model construction


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_build_model_returns_expected_class(prepared, model_name):
    model = prepared.build_model(model_name)

    assert isinstance(model, MODEL_CLASSES[model_name])
    assert prepared.model is model
    assert isinstance(prepared.optimizer, torch.optim.Adam)
    assert next(model.parameters()).device.type == prepared.device.type


def test_build_model_rejects_invalid_name(prepared):
    with pytest.raises(ValueError, match="Unsupported model"):
        prepared.build_model("invalid_model")


@pytest.mark.parametrize("model_name", RECURRENT_MODELS)
def test_recurrent_options_come_from_config(prepared, model_name):
    prepared.config["num_layers"] = 1
    prepared.config["bidirectional"] = True

    model = prepared.build_model(model_name)

    assert model.num_layers == 1
    assert model.bidirectional is True


def test_mlp_builds_when_recurrent_options_are_configured(prepared):
    # EmbeddingBagMLP does not accept these arguments; build_model must not pass them.
    prepared.config["num_layers"] = 2
    prepared.config["bidirectional"] = True

    assert isinstance(prepared.build_model("embedding_bag_mlp"), EmbeddingBagMLP)


# ------------------------------------------------------------ 4. reproducibility


def test_set_seed_makes_random_numbers_repeatable(trainer):
    trainer.config["seed"] = 123

    trainer._set_seed()
    first = torch.rand(5)
    trainer._set_seed()
    second = torch.rand(5)

    assert torch.equal(first, second)


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_build_model_is_reproducible(prepared, model_name):
    first = snapshot(prepared.build_model(model_name))
    second = snapshot(prepared.build_model(model_name))

    assert same_weights(first, second)


# ------------------------------------------------------------ 5. training epoch


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_train_epoch_returns_metrics_and_updates_weights(prepared, model_name):
    prepared.build_model(model_name)
    before = snapshot(prepared.model)

    metrics = prepared.train_epoch()

    assert set(metrics) == {"loss", "accuracy"}
    assert np.isfinite(metrics["loss"])
    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1
    assert prepared.model.training is True
    assert not same_weights(before, snapshot(prepared.model))


# ---------------------------------------------------------------- 6. evaluation


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_evaluate_returns_metrics_without_changing_weights(prepared, model_name):
    prepared.build_model(model_name)
    before = snapshot(prepared.model)

    metrics = prepared.evaluate()

    assert np.isfinite(metrics["loss"])
    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1
    # These two keys depend on what ClassificationEvaluator returns.
    assert {"macro_f1", "weighted_f1"} <= metrics.keys()
    assert prepared.model.training is False
    assert same_weights(before, snapshot(prepared.model))


def test_evaluate_is_deterministic(prepared):
    prepared.build_model("gru")

    first = prepared.evaluate()
    second = prepared.evaluate()

    assert first["loss"] == pytest.approx(second["loss"])


# ----------------------------------------------------------- 7. forward output


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_forward_output_shape(prepared, model_name):
    prepared.build_model(model_name)
    batch_features, _ = next(iter(prepared.val_loader))
    batch_features = batch_features.to(prepared.device)  # model lives on this device

    prepared.model.eval()
    with torch.inference_mode():
        logits = prepared.model(batch_features)

    num_classes = len(prepared.label_encoder.classes_)
    assert logits.shape == (len(batch_features), num_classes)
    assert torch.isfinite(logits).all()


# ------------------------------------------------------------- 8. training loop


def test_fit_returns_nested_history(prepared):
    history = prepared.fit("gru")

    assert len(history) == prepared.config["epochs"]
    for expected_epoch, entry in enumerate(history, start=1):
        assert entry["epoch"] == expected_epoch
        assert {"loss", "accuracy"} <= entry["train"].keys()
        assert {"loss", "accuracy"} <= entry["validation"].keys()
        assert np.isfinite(entry["train"]["loss"])
        assert np.isfinite(entry["validation"]["loss"])


def test_fit_stops_early_when_validation_loss_stalls(prepared):
    # With a learning rate of 0 the weights never change, so validation loss
    # never improves after epoch 1. patience=1 must stop training at epoch 2.
    prepared.config.update(learning_rate=0.0, epochs=5, patience=1)

    history = prepared.fit("embedding_bag_mlp")

    assert len(history) == 2


def test_fit_restores_weights_of_best_epoch(prepared, monkeypatch):
    prepared.config.update(epochs=3, patience=None)

    epoch_weights = []
    val_losses = iter([1.0, 0.5, 2.0])  # epoch 2 is the best

    def fake_train_epoch():
        with torch.no_grad():
            for param in prepared.model.parameters():
                param.add_(1.0)  # make every epoch's weights distinct
        epoch_weights.append(snapshot(prepared.model))
        return {"loss": 0.0, "accuracy": 0.0}

    def fake_evaluate():
        return {"loss": next(val_losses), "accuracy": 0.0}

    monkeypatch.setattr(prepared, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(prepared, "evaluate", fake_evaluate)

    prepared.fit("embedding_bag_mlp")

    assert same_weights(snapshot(prepared.model), epoch_weights[1])
    assert not same_weights(snapshot(prepared.model), epoch_weights[2])


def test_model_can_overfit_tiny_dataset(prepared):
    prepared.config.update(learning_rate=0.01, epochs=40, patience=None)

    history = prepared.fit("embedding_bag_mlp")

    assert history[-1]["train"]["loss"] < history[0]["train"]["loss"]