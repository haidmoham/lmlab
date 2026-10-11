"""The input, correctness, and hardware contract for this experiment."""

import torch

ROW_COUNTS = (1, 32, 256, 4096)
ROW_WIDTHS = (1, 31, 128, 257, 781, 1024, 2049, 4096, 8192)
DTYPES = {"float32": torch.float32, "float16": torch.float16}
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-6},
    torch.float16: {"rtol": 2e-3, "atol": 2e-5},
}


def require_cuda(device_index=0):
    """Fail before allocating inputs or recording results; never substitute a CPU."""
    if torch.version.hip is not None or not torch.cuda.is_available():
        raise RuntimeError(
            "this experiment requires an NVIDIA CUDA GPU and CUDA-enabled PyTorch; "
            "run on Linux or WSL2 on the RTX 3080. CPU and MPS are unsupported."
        )
    if not 0 <= device_index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device_index} is unavailable")
    torch.cuda.set_device(device_index)
    try:
        import triton  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Triton is missing; use the repository's locked Linux environment"
        ) from error
    return torch.device("cuda", device_index)


def torch_softmax(inputs, output=None):
    """The optimized PyTorch operator is the baseline, not an unfused Python formula."""
    return torch.softmax(inputs, dim=-1, out=output)


def assert_correct(inputs, actual):
    """Check element values and probability invariants, with dtype-specific tolerances."""
    if actual.shape != inputs.shape or actual.dtype != inputs.dtype:
        raise AssertionError("candidate must preserve input shape and dtype")
    if actual.device != inputs.device:
        raise AssertionError("candidate must preserve the input device")
    if not torch.isfinite(actual).all().item():
        raise AssertionError("candidate produced nonfinite probabilities")
    if not (actual >= 0).all().item():
        raise AssertionError("candidate produced negative probabilities")
    torch.testing.assert_close(actual, torch_softmax(inputs), **TOLERANCES[inputs.dtype])
    # A double-precision oracle also checks the reference's numerical agreement.
    oracle = torch.softmax(inputs.double(), dim=-1).to(inputs.dtype)
    torch.testing.assert_close(actual, oracle, **TOLERANCES[inputs.dtype])
    sums = actual.float().sum(dim=-1)
    sum_tolerance = 1e-5 if inputs.dtype == torch.float32 else 1e-3
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=sum_tolerance)


def adversarial_inputs(device, dtype):
    """Finite extreme logits, tied maxima, and tails exercise stability and masking."""
    shape = (33, 257)
    yield "zeros", torch.zeros(shape, device=device, dtype=dtype)
    yield "positive_offset", torch.full(shape, 1000, device=device, dtype=dtype)
    yield "negative_offset", torch.full(shape, -1000, device=device, dtype=dtype)
    ramp = torch.linspace(-1000, 1000, shape[1], device=device, dtype=dtype)
    yield "wide_range", ramp.expand(shape).contiguous()
    tail_maximum = torch.full(shape, -1000, device=device, dtype=dtype)
    tail_maximum[:, -1] = 1000
    yield "tail_maximum", tail_maximum
