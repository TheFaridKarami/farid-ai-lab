from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from configs import load_config
from data.data import prepare_data, split_data
from evaluation.evaluator import ClassificationEvaluator
from features.tokenizer import SequenceTokenizer
from models.neural import (
    EmbeddingBagMLP,
    GRUClassifier,
    LSTMClassifier,
    RNNClassifier,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Must match the tokenizer's padding ID. The models infer sentence lengths
# from it, so a mismatch silently hurts accuracy instead of raising an error.
PADDING_IDX = 0

RECURRENT_MODELS = {"rnn", "lstm", "gru"}


class NeuralTrainer:
    def __init__(self, config_path: str | Path = "configs/model.yaml"):
        self.config = load_config(config_path)["neural"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.label_encoder = LabelEncoder()

        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.evaluator = ClassificationEvaluator()

        self.train_loader = None
        self.val_loader = None

    # ------------------------------------------------------------------ data

    def prepare_datasets(self) -> None:
        train_df, _ = prepare_data()
        X_train, X_val, y_train, y_val = split_data(train_df)

        self.tokenizer = SequenceTokenizer(
            vocab_size=self.config["vocab_size"],
            sequence_length=self.config["sequence_length"],
        )
        self.tokenizer.fit(X_train)

        y_train_ids = self.label_encoder.fit_transform(y_train)
        y_val_ids = self.label_encoder.transform(y_val)

        train_dataset = TensorDataset(
            self._to_tensor(self.tokenizer.transform(X_train)),
            self._to_tensor(y_train_ids),
        )
        val_dataset = TensorDataset( 
            self._to_tensor(self.tokenizer.transform(X_val)),
            self._to_tensor(y_val_ids),
        )

        loader_kwargs = {
            "batch_size": self.config["batch_size"],
            "pin_memory": self.device.type == "cuda",
        }
        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    @staticmethod
    def _to_tensor(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long)

    # ----------------------------------------------------------------- model

    def _set_seed(self) -> None:
        seed = self.config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def build_model(self, model_name: str) -> nn.Module:
        model_classes = {
            "embedding_bag_mlp": EmbeddingBagMLP,
            "rnn": RNNClassifier,
            "lstm": LSTMClassifier,
            "gru": GRUClassifier,
        }
        if model_name not in model_classes:
            raise ValueError(
                f"Unsupported model: {model_name}. Choose from {sorted(model_classes)}."
            )

        # Seed before creating weights so every model starts reproducibly.
        self._set_seed()

        model_kwargs = {
            "vocab_size": len(self.tokenizer.get_vocabulary()),
            "embedding_dim": self.config["embedding_dim"],
            "hidden_dim": self.config["hidden_dim"],
            "num_classes": len(self.label_encoder.classes_),
            "dropout": self.config["dropout"],
            "padding_idx": PADDING_IDX,
        }
        # EmbeddingBagMLP does not accept these, so only recurrent models get them.
        if model_name in RECURRENT_MODELS:
            model_kwargs["num_layers"] = self.config.get("num_layers", 1)
            model_kwargs["bidirectional"] = self.config.get("bidirectional", False)

        self.model = model_classes[model_name](**model_kwargs).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
        )
        return self.model

    # -------------------------------------------------------------- training

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        """One pass over `loader`. Updates weights only when train=True."""
        self.model.train(train)

        total_loss = 0.0
        total_samples = 0
        all_labels, all_predictions = [], []

        # inference_mode is a faster, stricter version of no_grad for evaluation.
        grad_context = torch.enable_grad() if train else torch.inference_mode()
        

        with grad_context:
            for token_ids, labels in loader:
                token_ids = token_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(token_ids)
                loss = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    # Keeps RNN gradients from exploding on long sentences.
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.get("grad_clip", 1.0),
                    )
                    self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_labels.append(labels.cpu())
                all_predictions.append(logits.argmax(dim=1).cpu())

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_predictions).numpy()

        return {
            "loss": total_loss / total_samples,
            "accuracy": float((y_true == y_pred).mean()),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def train_epoch(self) -> dict:
        result = self._run_epoch(self.train_loader, train=True)
        return {"loss": result["loss"], "accuracy": result["accuracy"]}

    def evaluate(self) -> dict:
        result = self._run_epoch(self.val_loader, train=False)
        metrics = self.evaluator.compute_metrics(result["y_true"], result["y_pred"])
        return {
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            **metrics,
        }

    def fit(self, model_name: str) -> list[dict]:
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("Call prepare_datasets() before fit().")

        self.build_model(model_name)

        epochs = self.config["epochs"]
        patience = self.config.get("patience")  # None disables early stopping

        history = []
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()

            history.append(
                {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                # Copy to CPU so later training steps cannot modify the snapshot.
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    print(f"Early stopping: no improvement for {patience} epochs.")
                    break

        # Return the best model, not whichever one the last epoch produced.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history