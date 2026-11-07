import pytest
import torch
from models.transformer import PositionalEncoding, TransformerClassifier

# Banking77 values taken from configs/model.yaml (neural + transformer sections)
VOCAB_SIZE = 5000
NUM_CLASSES = 77
SEQUENCE_LENGTH = 79
MAX_LENGTH = 128
EMBEDDING_DIM = 128


def make_model(**overrides) -> TransformerClassifier:
    """Small, fast model for unit tests; override any argument per test."""
    params = {
        "vocab_size": 100,
        "embedding_dim": 16,
        "num_classes": 5,
        "max_length": 32,
        "num_heads": 4,
        "num_layers": 2,
    }
    params.update(overrides)
    return TransformerClassifier(**params)


def make_batch(batch_size=4, length=10, vocab_size=100) -> torch.Tensor:
    """Random token ids in [1, vocab_size), so id 0 (padding) never appears."""
    return torch.randint(1, vocab_size, (batch_size, length))


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(42)


# ---------- PositionalEncoding ----------

def test_positional_encoding_shape_and_not_trainable():
    pe = PositionalEncoding(embedding_dim=16, max_length=32)
    assert pe.encoding.shape == (1, 32, 16)
    assert len(list(pe.parameters())) == 0       # buffer, not a parameter
    assert "encoding" not in pe.state_dict()     # persistent=False


def test_positional_encoding_first_position_values():
    pe = PositionalEncoding(embedding_dim=16, max_length=32)
    assert torch.allclose(pe.encoding[0, 0, 0::2], torch.zeros(8))  # sin(0)
    assert torch.allclose(pe.encoding[0, 0, 1::2], torch.ones(8))   # cos(0)


def test_positional_encoding_forward_adds_encoding_for_short_sequences():
    pe = PositionalEncoding(embedding_dim=16, max_length=32)
    out = pe(torch.zeros(2, 7, 16))
    assert out.shape == (2, 7, 16)
    assert torch.allclose(out, pe.encoding[:, :7].expand(2, -1, -1))


def test_positional_encoding_supports_odd_dimension():
    pe = PositionalEncoding(embedding_dim=15, max_length=10)
    assert pe.encoding.shape == (1, 10, 15)


# ---------- Constructor validation ----------

@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_dim": 18, "num_heads": 4},  # not divisible
        {"num_heads": 0},
        {"num_layers": 0},
        {"max_length": 0},
    ],
)
def test_invalid_constructor_arguments_raise(overrides):
    with pytest.raises(ValueError):
        make_model(**overrides)


def test_default_feedforward_dim_is_four_times_embedding_dim():
    model = make_model(embedding_dim=16, feedforward_dim=None)
    assert model.encoder.layers[0].linear1.out_features == 64


# ---------- Embedding initialisation ----------

def test_padding_embedding_row_is_zero():
    model = make_model(padding_idx=0)
    assert torch.count_nonzero(model.embedding.weight[0]) == 0


def test_embedding_init_std_matches_inverse_sqrt_dim():
    model = make_model(vocab_size=2000, embedding_dim=64)
    std = model.embedding.weight.std().item()
    assert abs(std - 64 ** -0.5) < 0.01


# ---------- Forward pass ----------

def test_forward_output_shape():
    model = make_model()
    assert model(make_batch(4, 10)).shape == (4, 5)


def test_forward_accepts_sequence_exactly_max_length():
    model = make_model(max_length=32)
    assert model(make_batch(2, 32)).shape == (2, 5)


def test_forward_rejects_sequence_longer_than_max_length():
    model = make_model(max_length=32)
    with pytest.raises(ValueError, match="exceeds max_length"):
        model(make_batch(2, 33))


def test_forward_rejects_all_padding_sequence():
    model = make_model()
    tokens = make_batch(2, 10)
    tokens[1] = 0  # second sequence has no real tokens
    with pytest.raises(ValueError, match="no real tokens"):
        model(tokens)


def test_eval_mode_is_deterministic():
    model = make_model().eval()
    tokens = make_batch()
    with torch.no_grad():
        assert torch.equal(model(tokens), model(tokens))


def test_padding_does_not_change_logits():
    model = make_model().eval()
    tokens = make_batch(1, 6)
    padded = torch.cat([tokens, torch.zeros(1, 9, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(tokens), model(padded), atol=1e-5)


def test_token_order_matters():
    model = make_model().eval()
    tokens = make_batch(1, 8)
    with torch.no_grad():
        assert not torch.allclose(model(tokens), model(tokens.flip(dims=[1])), atol=1e-5)


# ---------- Training behaviour ----------

def test_gradients_reach_all_parameters_and_skip_padding_row():
    model = make_model().train()
    tokens = make_batch(4, 10)
    tokens[:, 8:] = 0  # right-pad the last two positions
    labels = torch.randint(0, 5, (4,))

    loss = torch.nn.functional.cross_entropy(model(tokens), labels)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
    assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0


def test_one_optimizer_step_changes_weights():
    model = make_model().train()
    before = model.classifier[1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = torch.nn.functional.cross_entropy(
        model(make_batch(4, 10)), torch.randint(0, 5, (4,))
    )
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, model.classifier[1].weight)


# ---------- Banking77-sized smoke test ----------

def test_banking77_sized_model_forward():
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        max_length=MAX_LENGTH,
        dropout=0.2,
    ).eval()
    tokens = torch.randint(1, VOCAB_SIZE, (16, SEQUENCE_LENGTH))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (16, NUM_CLASSES)
    assert torch.isfinite(logits).all()