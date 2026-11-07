import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class EmbeddingBagMLP(nn.Module):
    """A bag-of-embeddings classifier for text intent classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)

        mask = (token_ids != self.padding_idx).unsqueeze(-1)
        embeddings = embeddings * mask

        summed_embeddings = embeddings.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)

        mean_embeddings = summed_embeddings / lengths

        return self.classifier(mean_embeddings)


class _RecurrentClassifier(nn.Module):
    """Shared implementation for RNN, LSTM, and GRU classifiers."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        rnn_type: str,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        
        if rnn_type not in {"RNN", "LSTM", "GRU"}:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'.")

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "RNN":
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=recurrent_dropout,
                nonlinearity="tanh",
            )

        elif rnn_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=recurrent_dropout,
            )

        else:
            self.recurrent = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=recurrent_dropout,
            )

        self.classifier = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            num_classes,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)

        lengths = (token_ids != self.padding_idx).sum(dim=1)

        if (lengths == 0).any():
            raise ValueError("Input contains a sequence with no real tokens.")

        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.recurrent(packed)

        if self.rnn_type == "LSTM":
            hidden = hidden[0]

        if self.bidirectional:
            features = torch.cat(
                (hidden[-2], hidden[-1]),
                dim=1,
            )
        else:
            features = hidden[-1]

        return self.classifier(features)


class RNNClassifier(_RecurrentClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, rnn_type="RNN", **kwargs)


class LSTMClassifier(_RecurrentClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, rnn_type="LSTM", **kwargs)


class GRUClassifier(_RecurrentClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, rnn_type="GRU", **kwargs)