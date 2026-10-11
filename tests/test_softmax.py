"""Numerical regression tests; GPU cases skip in the ordinary CPU/Mac suite."""

import pytest
import torch

from kernels.triton.softmax.common import adversarial_inputs, assert_correct, require_cuda


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
