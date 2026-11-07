# tests/test_trainer_transformer.py
import numpy as np
import pandas as pd
import pytest
import torch
from models.transformer import TransformerClassifier
from training.trainer_transformer import PADDING_ID, TransformerTrainer, set_seed


def make_config(**overrides) -> dict:
    """Tiny config so every test runs in well under a second."""
    transformer = {
        "vocab_size": 50,
        "sequence_length": 6,
        "max_length": 8,
        "embedding_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "feedforward_dim": 32,
        "dropout": 0.0,  # dropout off: train and eval behave the same, tests stay deterministic
        "batch_size": 4,
        "learning_rate": 0.01,
        "epochs": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "grad_clip": 1.0,
        "patience": 2,
        "seed": 42,
    }
    transformer.update(overrides)
    return {"transformer": transformer}


def make_data() -> tuple[pd.Series, pd.Series]:
    """24 texts, 3 string-labelled classes, each class has its own keywords."""
    keywords = {
        "card_lost": "card lost stolen",
        "transfer": "transfer money send",
        "balance": "balance account check",
    }
    texts, labels = [], []
    for label, words in keywords.items():
        for i in range(8):
            texts.append(f"{words} number {i}")
            labels.append(label)
    return pd.Series(texts), pd.Series(labels)


@pytest.fixture
def trainer() -> TransformerTrainer:
    return TransformerTrainer(make_config())


@pytest.fixture
def prepared(trainer) -> dict:
    X, y = make_data()
    tokenizer, label_encoder, train_loader, val_loader = trainer.prepare_data(X, X, y, y)
    return {
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vocab_size": len(tokenizer.get_vocabulary()),
        "num_classes": len(label_encoder.classes_),
    }


# ---------- init and seeding ----------

def test_init_rejects_sequence_length_longer_than_max_length():
    with pytest.raises(ValueError, match="sequence_length"):
        TransformerTrainer(make_config(sequence_length=9, max_length=8))


def test_set_seed_makes_torch_random_numbers_repeatable():
    set_seed(7)
    first = torch.rand(3)
    set_seed(7)
    assert torch.equal(first, torch.rand(3))


# ---------- prepare_data / build_loader ----------

def test_prepare_data_batches_have_expected_shapes_and_dtypes(prepared):
    token_ids, labels = next(iter(prepared["train_loader"]))
    assert token_ids.shape == (4, 6)  # (batch_size, sequence_length)
    assert token_ids.dtype == torch.int64
    assert labels.dtype == torch.int64
    assert labels.min() >= 0 and labels.max() < prepared["num_classes"]


def test_label_encoder_maps_string_labels_to_sorted_ids(prepared):
    assert list(prepared["label_encoder"].classes_) == ["balance", "card_lost", "transfer"]


def test_tokenizer_is_fitted_on_train_text_only(trainer):
    X, y = make_data()
    X_val = pd.Series(["card lost zzzunique"])
    y_val = pd.Series(["card_lost"])
    tokenizer, _, _, _ = trainer.prepare_data(X, X_val, y, y_val)
    assert "zzzunique" not in tokenizer.get_vocabulary()


def test_validation_loader_keeps_original_order(trainer, prepared):
    _, y = make_data()
    labels = torch.cat([batch[1] for batch in prepared["val_loader"]]).numpy()
    assert np.array_equal(labels, prepared["label_encoder"].transform(y))


def test_train_loader_shuffle_is_reproducible_with_same_seed():
    X, y = make_data()
    first_batches = []
    for _ in range(2):
        trainer = TransformerTrainer(make_config())
        _, _, train_loader, _ = trainer.prepare_data(X, X, y, y)
        first_batches.append(next(iter(train_loader))[1])
    assert torch.equal(first_batches[0], first_batches[1])


def test_build_loader_rejects_length_mismatch(trainer, prepared):
    with pytest.raises(ValueError, match="same length"):
        trainer.build_loader(
            prepared["tokenizer"], prepared["label_encoder"],
            ["card lost", "transfer money"], ["card_lost"],
        )


def test_build_loader_rejects_text_with_no_tokens(trainer, prepared):
    # punctuation-only text is stripped by Keras and becomes all padding
    with pytest.raises(ValueError, match="no tokens"):
        trainer.build_loader(
            prepared["tokenizer"], prepared["label_encoder"], ["!!!"], ["card_lost"]
        )


# ---------- model, optimizer, scheduler ----------

def test_build_model_uses_config_and_padding_id(trainer):
    model = trainer.build_model(vocab_size=20, num_classes=3)
    assert isinstance(model, TransformerClassifier)
    assert model.padding_idx == PADDING_ID
    assert model.max_length == trainer.config["max_length"]
    assert next(model.parameters()).device.type == trainer.device.type


def test_optimizer_applies_weight_decay_to_matrices_only(trainer):
    model = trainer.build_model(vocab_size=20, num_classes=3)
    decay_group, no_decay_group = trainer._build_optimizer(model).param_groups

    assert decay_group["weight_decay"] == trainer.config["weight_decay"]
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert all(p.ndim < 2 for p in no_decay_group["params"])
    assert len(decay_group["params"]) + len(no_decay_group["params"]) == len(
        list(model.parameters())
    )


def test_scheduler_warms_up_then_decays_to_zero(trainer):
    model = trainer.build_model(vocab_size=20, num_classes=3)
    optimizer = trainer._build_optimizer(model)
    scheduler = trainer._build_scheduler(optimizer, total_steps=100)  # warmup = 10 steps

    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates[0] < rates[9]                              # warmup ramps up
    assert rates[9] == pytest.approx(trainer.config["learning_rate"])  # peak = configured lr
    assert rates[50] < rates[10]                            # decay after the peak
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)       # ends at zero


# ---------- epochs and training ----------

def test_train_epoch_updates_weights_and_eval_epoch_does_not(trainer, prepared):
    model = trainer.build_model(prepared["vocab_size"], prepared["num_classes"])
    snapshot = lambda: [p.detach().cpu().clone() for p in model.parameters()]

    before_eval = snapshot()
    trainer._run_epoch(model, prepared["val_loader"])
    assert all(torch.equal(a, b) for a, b in zip(before_eval, snapshot()))

    optimizer = trainer._build_optimizer(model)
    scheduler = trainer._build_scheduler(optimizer, total_steps=len(prepared["train_loader"]) * 5)
    trainer._run_epoch(model, prepared["train_loader"], optimizer, scheduler)
    assert not all(torch.equal(a, b) for a, b in zip(before_eval, snapshot()))


def test_train_model_can_overfit_a_tiny_dataset():
    trainer = TransformerTrainer(make_config(epochs=30, patience=30))
    X, y = make_data()
    tokenizer, encoder, train_loader, val_loader = trainer.prepare_data(X, X, y, y)

    _, history = trainer.train_model(
        train_loader, val_loader,
        vocab_size=len(tokenizer.get_vocabulary()),
        num_classes=len(encoder.classes_),
    )

    assert history["train_loss"][-1] < history["train_loss"][0]
    assert max(history["train_accuracy"]) >= 0.95


def test_early_stopping_halts_when_validation_stops_improving():
    # lr = 0 -> weights never change -> val accuracy is identical every epoch
    trainer = TransformerTrainer(make_config(learning_rate=0.0, epochs=20, patience=1))
    X, y = make_data()
    tokenizer, encoder, train_loader, val_loader = trainer.prepare_data(X, X, y, y)

    _, history = trainer.train_model(
        train_loader, val_loader,
        vocab_size=len(tokenizer.get_vocabulary()),
        num_classes=len(encoder.classes_),
    )

    assert len(history["val_accuracy"]) == 2  # epoch 1 sets the best, epoch 2 is stale


def test_train_model_restores_weights_from_best_epoch(trainer, prepared, monkeypatch):
    trainer.config["epochs"] = 10
    trainer.config["patience"] = 2

    val_scores = iter([0.5, 0.9, 0.3, 0.2, 0.1])  # best is epoch 2
    snapshots = []

    def scripted_run_epoch(model, loader, optimizer=None, scheduler=None):
        if optimizer is not None:  # a "training" epoch: change the weights visibly
            with torch.no_grad():
                model.classifier[1].weight.add_(1.0)
            snapshots.append(model.classifier[1].weight.detach().cpu().clone())
            return 0.0, 0.0
        return 0.0, next(val_scores)  # an "evaluation" epoch: scripted accuracy

    monkeypatch.setattr(trainer, "_run_epoch", scripted_run_epoch)

    model, history = trainer.train_model(
        prepared["train_loader"], prepared["val_loader"],
        prepared["vocab_size"], prepared["num_classes"],
    )

    assert history["val_accuracy"] == [0.5, 0.9, 0.3, 0.2]  # stopped after 2 stale epochs
    assert len(snapshots) == 4
    final_weight = model.classifier[1].weight.detach().cpu()
    assert torch.equal(final_weight, snapshots[1])          # epoch 2 weights, not epoch 4


# ---------- predict ----------

def test_predict_returns_one_valid_class_id_per_sample(trainer, prepared):
    model = trainer.build_model(prepared["vocab_size"], prepared["num_classes"])
    predictions = trainer.predict(model, prepared["val_loader"])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (24,)
    assert predictions.min() >= 0 and predictions.max() < prepared["num_classes"]