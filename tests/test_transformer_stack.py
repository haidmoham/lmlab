import pytest
import torch

from src.language_models import TransformerStack


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_stack_composition_and_registered_independent_parameters(depth):
    stack = TransformerStack(n_layers=depth, d_model=8, n_head=2, d_k=4, d_v=4, d_ff=16)
    input_embeddings = torch.randn(2, 5, 8)
    stack_output, layer_attention_weights = stack(input_embeddings)
    expected = input_embeddings
    for block, weights in zip(stack.blocks, layer_attention_weights, strict=True):
        expected, expected_weights = block(expected)
        torch.testing.assert_close(weights, expected_weights)
        assert weights.shape == (2, 2, 5, 5)
    torch.testing.assert_close(stack_output, expected)
    parameters = [parameter for block in stack.blocks for parameter in block.parameters()]
    block_parameter_ids = {id(parameter) for parameter in parameters}
    registered_parameter_ids = {id(parameter) for parameter in stack.parameters()}
    assert len(block_parameter_ids) == len(parameters), "blocks must not share parameters"
    assert registered_parameter_ids == block_parameter_ids, (
        "all block parameters must be registered"
    )
    stack_output.square().mean().backward()
    for parameter in parameters:
        assert parameter.grad is not None, "every parameter must receive a gradient"
        assert torch.isfinite(parameter.grad).all(), "gradients must not contain NaN or infinity"


def test_stack_preserves_causality():
    stack = TransformerStack(n_layers=3, d_model=8, n_head=2, d_k=4, d_v=4, d_ff=16)
    input_embeddings = torch.randn(2, 5, 8)
    changed = input_embeddings.clone()
    changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 10
    original, _ = stack(input_embeddings)
    perturbed, _ = stack(changed)
    torch.testing.assert_close(original[:, :3], perturbed[:, :3])


def test_stack_rejects_empty_depth():
    with pytest.raises(ValueError, match="n_layers"):
        TransformerStack(n_layers=0, d_model=8, n_head=2, d_k=4, d_v=4, d_ff=16)
