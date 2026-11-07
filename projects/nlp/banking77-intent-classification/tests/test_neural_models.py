import pytest
import torch
from models.neural import (
    EmbeddingBagMLP,
    GRUClassifier,
    LSTMClassifier,
    RNNClassifier,
)

models = {
    "EmbeddingBagMLP": EmbeddingBagMLP(100, 16, 32, 77),
    "RNN": RNNClassifier(100, 16, 32, 77),
    "LSTM": LSTMClassifier(100, 16, 32, 77),
    "GRU": GRUClassifier(100, 16, 32, 77),
}

@pytest.mark.parametrize(
    "name, model", list(models.items())
)

def test_neural_models(name, model):
    token_ids = torch.tensor([
        [12, 45, 78, 0, 0],
        [23, 56, 91, 34, 0],
    ])

    logits = model(token_ids)
    assert logits.shape == (2, 77), (
        f"{name}: unexpected shape {logits.shape}"
)
    assert torch.isfinite(logits).all(), (
    f"{name}: output contains NaN or infinity"
    )
    
    
@pytest.mark.parametrize(
    "name, model", list(models.items())
)

def test_neural_models_can_backpropagate(name, model):
    token_ids = torch.tensor([
        [12, 45, 78, 0, 0],
        [23, 56, 91, 34, 0],
    ])

    labels = torch.tensor([3, 10])

    model.zero_grad()

    logits = model(token_ids)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    loss.backward()

    assert loss.ndim == 0, (
        f"{name}: loss should be a scalar"
    )
    assert torch.isfinite(loss), (
        f"{name}: loss contains NaN or infinity"
    )
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    ), f"{name}: no gradients were produced"
 