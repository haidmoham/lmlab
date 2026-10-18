"""Numerical regression tests; GPU cases skip in the ordinary CPU/Mac suite."""

import pytest
import torch

from kernels.triton.softmax.benchmark import provider_orders
from kernels.triton.softmax.common import (
    adversarial_inputs,
    assert_correct,
    eager_softmax,
    require_cuda,
)


@pytest.fixture
def gpu_kernel():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA GPU required; use the strict softmax.check CLI on the workstation")
    require_cuda()
    from kernels.triton.softmax.kernel import triton_softmax

    return triton_softmax


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (1, 31),
        (32, 128),
        (33, 257),
        (256, 781),
        (256, 1024),
        (32, 2049),
        (256, 4096),
        (4096, 128),
        (32, 8192),
    ],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_softmax_shapes(gpu_kernel, dtype, shape, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    actual = gpu_kernel(inputs)
    assert_correct(inputs, actual)
    output = torch.empty_like(inputs)
    assert gpu_kernel(inputs, output) is output
    assert_correct(inputs, output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_stability_and_tail_masking(gpu_kernel, dtype):
    for _, inputs in adversarial_inputs("cuda", dtype):
        assert_correct(inputs, gpu_kernel(inputs))


def test_rejects_unsupported_gpu_inputs(gpu_kernel):
    for shape in [(0, 31), (3, 0), (1, 8193)]:
        with pytest.raises(ValueError, match="rows|width"):
            gpu_kernel(torch.empty(shape, device="cuda"))
    with pytest.raises(ValueError, match="contiguous 2D"):
        gpu_kernel(torch.randn((3, 31), device="cuda").T)
    with pytest.raises(ValueError, match="autograd"):
        gpu_kernel(torch.randn((3, 31), device="cuda", requires_grad=True))
    backing = torch.randn(4 * 31, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        gpu_kernel(backing[: 3 * 31].reshape(3, 31), backing[31:].reshape(3, 31))


def test_correctness_gate_rejects_an_all_zero_candidate():
    inputs = torch.zeros((2, 31))
    with pytest.raises(AssertionError):
        assert_correct(inputs, torch.zeros_like(inputs))


def test_hardware_gate_never_substitutes_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU and MPS are unsupported"):
        require_cuda()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1, 1), (3, 31), (2, 257), (1, 8192)])
def test_eager_shape_dtype_and_preallocated_output(dtype, shape):
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(shape, dtype=dtype, generator=generator)
    original = inputs.clone()
    assert_correct(inputs, eager_softmax(inputs))
    output = torch.full_like(inputs, float("nan"))
    assert eager_softmax(inputs, output) is output
    assert_correct(inputs, output)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)
    inputs.zero_()
    assert eager_softmax(inputs, output) is output
    torch.testing.assert_close(output, torch.full_like(inputs, 1 / shape[-1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_eager_finite_extreme_logits(dtype):
    for _, inputs in adversarial_inputs("cpu", dtype):
        assert_correct(inputs, eager_softmax(inputs))


def test_eager_half_precision_extremes_round_only_final_probabilities():
    inputs = torch.tensor(
        [
            [65504, 65472, 65440],
            [-65504, -65472, -65440],
            [65504, -65504, 0],
            [1000, 999.5, 999],
            [-1000, -1000.5, -1001],
        ],
        dtype=torch.float16,
    )
    expected = torch.softmax(inputs.double(), dim=-1).half()
    output = torch.empty_like(inputs)
    assert eager_softmax(inputs, output) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert_correct(inputs, output)
    uniform = torch.full((2, 8192), 65504, dtype=torch.float16)
    assert_correct(uniform, eager_softmax(uniform))


@pytest.mark.parametrize("shape", [(0, 31), (3, 0), (1, 8193)])
def test_eager_rejects_unsupported_shapes(shape):
    with pytest.raises(ValueError, match="rows|width"):
        eager_softmax(torch.empty(shape))


def test_eager_rejects_unsupported_inputs_and_outputs():
    with pytest.raises(ValueError, match="contiguous 2D"):
        eager_softmax(torch.randn(3, 31).T)
    with pytest.raises(ValueError, match="contiguous 2D"):
        eager_softmax(torch.randn(31))
    with pytest.raises(ValueError, match="float32 and float16"):
        eager_softmax(torch.ones(3, 31, dtype=torch.float64))
    with pytest.raises(ValueError, match="autograd"):
        eager_softmax(torch.randn(3, 31, requires_grad=True))
    inputs = torch.randn(3, 31)
    for output in (
        torch.empty(3, 30),
        torch.empty(3, 31, dtype=torch.float16),
        torch.empty(31, 3).T,
        torch.empty_like(inputs, requires_grad=True),
    ):
        with pytest.raises(ValueError, match="output must match"):
            eager_softmax(inputs, output)
    backing = torch.randn(4 * 31)
    with pytest.raises(ValueError, match="overlap"):
        eager_softmax(backing[: 3 * 31].reshape(3, 31), backing[31:].reshape(3, 31))


@pytest.mark.parametrize("repeats", [6, 12, 18])
@pytest.mark.parametrize("offset", [-7, 0, 1, 5, 6, 13])
def test_provider_order_balances_positions_and_pair_precedence(repeats, offset):
    orders = provider_orders(repeats, offset)
    providers = {"torch", "eager", "triton"}
    assert len(orders) == repeats
    for order in orders:
        assert len(order) == 3
        assert set(order) == providers
    for provider in providers:
        for position in range(3):
            assert sum(order[position] == provider for order in orders) == repeats // 3
        for other in providers - {provider}:
            preceding = sum(order.index(provider) < order.index(other) for order in orders)
            assert preceding == repeats // 2


def test_provider_order_offset_rotates_schedule():
    baseline = provider_orders(12)
    assert provider_orders(12, 1) == baseline[1:] + baseline[:1]
    assert provider_orders(12, 7) == provider_orders(12, 1)


@pytest.mark.parametrize("repeats", [-6, 0, 1, 5, 7, 13])
def test_provider_order_rejects_unbalanced_repeat_counts(repeats):
    with pytest.raises(ValueError, match="multiple of 6"):
        provider_orders(repeats)


def test_division_into_half_output_rounds_float32_result():
    numerators = torch.tensor([[1.0, 0.60653067, 0.36787945]], dtype=torch.float32)
    denominators = numerators.sum(dim=-1, keepdim=True)
    output = torch.empty_like(numerators, dtype=torch.float16)
    expected = (numerators / denominators).half()
    assert torch.div(numerators, denominators, out=output) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
