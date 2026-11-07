from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["PositionalEncoding", "TransformerClassifier"]


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding ("Attention Is All You Need")."""

    def __init__(self, embedding_dim: int, max_length: int) -> None:
        super().__init__()

        positions = torch.arange(max_length).unsqueeze(1)  # (max_length, 1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )  # (ceil(embedding_dim / 2),)

        encoding = torch.zeros(max_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: embedding_dim // 2])

        # Buffer: moves with .to(device) but is not a trainable parameter.
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (batch_size, sequence_length, embedding_dim)
        return embeddings + self.encoding[:, : embeddings.size(1)]


class TransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        max_length: int,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if num_heads < 1 or embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        if feedforward_dim is None:
            feedforward_dim = 4 * embedding_dim

        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.positional_encoding = PositionalEncoding(embedding_dim, max_length)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # our tensors are (B, L, D), not (L, B, D)
            norm_first=True,  # pre-norm: LayerNorm before each sub-layer
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),  # final norm for pre-norm stacks
            enable_nested_tensor=False,  # not supported with norm_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

        self._init_embedding()

    def _init_embedding(self) -> None:
        # std = D ** -0.5 together with the sqrt(D) scaling in forward gives
        # unit-variance embeddings, comparable to the positional signal.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.embedding_dim**-0.5)
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, sequence_length), right-padded with padding_idx
        sequence_length = token_ids.size(1)
        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max_length "
                f"{self.max_length}."
            )

        real_token_mask = token_ids != self.padding_idx  # True = real token

        if not real_token_mask.any(dim=1).all():
            raise ValueError("Input contains a sequence with no real tokens.")

        x = self.embedding(token_ids) * math.sqrt(self.embedding_dim)
        x = self.embedding_dropout(self.positional_encoding(x))
        # x: (batch_size, sequence_length, embedding_dim)

        # PyTorch's mask has the OPPOSITE meaning: True = ignore this position.
        x = self.encoder(x, src_key_padding_mask=~real_token_mask)

        # Masked mean pooling: padding must not change the sentence vector.
        summed = x.masked_fill(~real_token_mask.unsqueeze(-1), 0.0).sum(dim=1)
        counts = real_token_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
        pooled = summed / counts  # (batch_size, embedding_dim)

        return self.classifier(pooled)  # (batch_size, num_classes)


