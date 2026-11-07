# src/training/trainer_transformer.py
from __future__ import annotations

import logging
import random

import numpy as np
import torch
from features.tokenizer import SequenceTokenizer
from models.transformer import TransformerClassifier
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

PADDING_ID = 0  # Keras TextVectorization reserves id 0 for padding (id 1 is [UNK])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformerTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config["transformer"]
        if self.config["sequence_length"] > self.config["max_length"]:
            raise ValueError("sequence_length must not exceed max_length.")
        set_seed(self.config["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- data ----------

    def prepare_data(self, X_train, X_val, y_train, y_val):
        tokenizer = SequenceTokenizer(
            vocab_size=self.config["vocab_size"],
            sequence_length=self.config["sequence_length"],
        )
        tokenizer.fit(np.asarray(X_train, dtype=str))  # train only: no leakage

        label_encoder = LabelEncoder().fit(y_train)  # string labels -> 0..76

        train_loader = self.build_loader(tokenizer, label_encoder, X_train, y_train, shuffle=True)
        val_loader = self.build_loader(tokenizer, label_encoder, X_val, y_val)
        return tokenizer, label_encoder, train_loader, val_loader

    def build_loader(
        self, tokenizer, label_encoder, texts, labels, shuffle: bool = False
    ) -> DataLoader:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")

        ids = np.asarray(tokenizer.transform(np.asarray(texts, dtype=str))).astype(np.int64)
        token_ids = torch.from_numpy(ids)

        empty_rows = int((token_ids == PADDING_ID).all(dim=1).sum())
        if empty_rows:
            raise ValueError(f"{empty_rows} text(s) have no tokens after vectorization.")

        label_ids = torch.as_tensor(label_encoder.transform(labels), dtype=torch.long)
        generator = torch.Generator().manual_seed(self.config["seed"]) if shuffle else None
        return DataLoader(
            TensorDataset(token_ids, label_ids),
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            generator=generator,
        )

    # ---------- model ----------

    def build_model(self, vocab_size: int, num_classes: int) -> TransformerClassifier:
        model = TransformerClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.config["embedding_dim"],
            num_classes=num_classes,
            max_length=self.config["max_length"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            feedforward_dim=self.config["feedforward_dim"],
            dropout=self.config["dropout"],
            padding_idx=PADDING_ID,
        )
        return model.to(self.device)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        # Weight decay on matrices only; biases and LayerNorm weights are 1-D.
        decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config["learning_rate"],
        )

    def _build_scheduler(self, optimizer, total_steps: int) -> LambdaLR:
        warmup = int(total_steps * self.config["warmup_ratio"])
        decay = max(1, total_steps - warmup)

        def multiplier(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup  # linear warmup
            return max(0.0, (total_steps - step) / decay)  # linear decay

        return LambdaLR(optimizer, multiplier)

    # ---------- training ----------

    def _run_epoch(self, model, loader, optimizer=None, scheduler=None) -> tuple[float, float]:
        training = optimizer is not None
        model.train(training)
        total_loss, correct, seen = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for token_ids, labels in loader:
                token_ids, labels = token_ids.to(self.device), labels.to(self.device)
                logits = model(token_ids)
                loss = nn.functional.cross_entropy(logits, labels)

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.config["grad_clip"])
                    optimizer.step()
                    scheduler.step()

                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                seen += labels.size(0)

        return total_loss / seen, correct / seen

    def train_model(self, train_loader, val_loader, vocab_size: int, num_classes: int):
        model = self.build_model(vocab_size, num_classes)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(
            optimizer, self.config["epochs"] * len(train_loader)
        )

        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        best_accuracy, best_state, stale_epochs = -1.0, None, 0

        for epoch in range(1, self.config["epochs"] + 1):
            train_loss, train_acc = self._run_epoch(model, train_loader, optimizer, scheduler)
            val_loss, val_acc = self._run_epoch(model, val_loader)

            for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
                history[key].append(value)
            logger.info(
                "epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
                epoch, self.config["epochs"], train_loss, train_acc, val_loss, val_acc,
            )

            if val_acc > best_accuracy:
                best_accuracy, stale_epochs = val_acc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
                if stale_epochs >= self.config["patience"]:
                    logger.info("early stopping at epoch %d", epoch)
                    break

        model.load_state_dict(best_state)  # keep the best epoch, not the last
        return model, history

    @torch.no_grad()
    def predict(self, model, loader) -> np.ndarray:
        """Returns encoded class ids (0..76). Decode with label_encoder.inverse_transform."""
        model.eval()
        outputs = [
            model(token_ids.to(self.device)).argmax(dim=1).cpu()
            for token_ids, _ in loader
        ]
        return torch.cat(outputs).numpy()