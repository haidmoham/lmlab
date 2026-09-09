import math

import pytest
import torch

from src.language_models import (
    SingleTransformerLanguageModel,
    SinusoidalPositionalEncoding,
    StackedTransformerLanguageModel,
)


@pytest.mark.parametrize("width", [1, 7, 32])
def test_sinusoidal_encoding_matches_paper_formula(width):
    positions = torch.tensor([0, 1, 8, 256, 1024])
    encoder = SinusoidalPositionalEncoding(width)
    actual = encoder(positions)
    expected = torch.empty(len(positions), width)
    for row, position in enumerate(positions.tolist()):
        for feature in range(width):
            pair_index = feature // 2
            angle = position / (10000 ** (2 * pair_index / width))
            if feature % 2 == 0:
                expected[row, feature] = math.sin(angle)
            else:
                expected[row, feature] = math.cos(angle)
    # Float32 frequency rounding accumulates in the angle at larger positions.
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "model_class", [SingleTransformerLanguageModel, StackedTransformerLanguageModel]
)
def test_fixed_positions_allow_longer_inputs_and_training_without_future_leakage(model_class):
    model = model_class(vocab_size=16, n_embd=8, block_size=4, position_encoding="sinusoidal")
    encoder = model.position_embedding_table
    assert list(encoder.parameters()) == [], "fixed positions must not add trainable parameters"
    positions = torch.arange(9)
    original_encodings = encoder(positions).clone()
    token_ids = torch.arange(9).unsqueeze(0)
    logits = model(token_ids)
    assert logits.shape == (1, 9, 16)

    changed_tokens = token_ids.clone()
    changed_tokens[:, 5:] = 15
    torch.testing.assert_close(logits[:, :5], model(changed_tokens)[:, :5])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 16), token_ids.flatten())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}"
    optimizer.step()
    torch.testing.assert_close(encoder(positions), original_encodings, rtol=0, atol=0)

    restored = model_class(vocab_size=16, n_embd=8, block_size=4, position_encoding="sinusoidal")
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(token_ids), model(token_ids))


def test_position_buffer_follows_model_dtype():
    encoder = SinusoidalPositionalEncoding(8).to(dtype=torch.float64)
    assert encoder(torch.arange(4)).dtype == torch.float64


def test_unknown_encoding_fails_clearly():
    with pytest.raises(ValueError, match="position_encoding"):
        StackedTransformerLanguageModel(vocab_size=16, position_encoding="sine")
