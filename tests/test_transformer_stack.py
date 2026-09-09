import pytest
import torch

from src.language_models import TransformerStack


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_stack_composition_and_registered_independent_parameters(depth):
    stack = TransformerStack(depth, 8, 2, 4, 4, 16)
    x = torch.randn(2, 5, 8)
    actual, maps = stack(x)
    expected = x
    for block, weights in zip(stack.blocks, maps, strict=True):
        expected, expected_weights = block(expected)
        torch.testing.assert_close(weights, expected_weights)
        assert weights.shape == (2, 2, 5, 5)
    torch.testing.assert_close(actual, expected)
    parameters = [p for block in stack.blocks for p in block.parameters()]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in stack.parameters()} == {id(p) for p in parameters}
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)


def test_stack_preserves_causality():
    stack = TransformerStack(3, 8, 2, 4, 4, 16)
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 10
    original, _ = stack(x)
    perturbed, _ = stack(changed)
    torch.testing.assert_close(original[:, :3], perturbed[:, :3])


def test_stack_rejects_empty_depth():
    with pytest.raises(ValueError, match="n_layers"):
        TransformerStack(0, 8, 2, 4, 4, 16)
