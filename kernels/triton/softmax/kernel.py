"""One Triton program per row, adapted from Triton's fused-softmax tutorial.

https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
This deliberately starts without persistent scheduling or autotuning.
"""

import torch
import triton
import triton.language as tl

from .common import DTYPES


@triton.jit
def _softmax_kernel(input_ptr, output_ptr, width: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    offsets = row * width + columns
    # Padding must not change the maximum or the denominator.
    values = tl.load(input_ptr + offsets, mask=columns < width, other=-float("inf"))
    values = values.to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerators = tl.exp(shifted)
    probabilities = numerators / tl.sum(numerators, axis=0)
    tl.store(output_ptr + offsets, probabilities, mask=columns < width)


def triton_softmax(inputs, output=None, *, num_warps=4):
    """Forward only: contiguous, finite, nonempty 2D CUDA logits, width <= 8192."""
    if not inputs.is_cuda or torch.version.hip is not None:
        raise ValueError("inputs must be on an NVIDIA CUDA GPU")
    if inputs.ndim != 2 or not inputs.is_contiguous():
        raise ValueError("inputs must be a contiguous 2D tensor")
    rows, width = inputs.shape
    if rows < 1 or not 1 <= width <= 8192:
        raise ValueError("rows must be positive and width must be between 1 and 8192")
    if inputs.dtype not in DTYPES.values():
        raise ValueError("only float32 and float16 are supported")
    if inputs.requires_grad:
        raise ValueError("this forward-only experiment does not implement autograd")
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8, or 16")
    if output is None:
        output = torch.empty_like(inputs)
    if (
        output.shape != inputs.shape
        or output.dtype != inputs.dtype
        or output.device != inputs.device
        or not output.is_contiguous()
        or output.requires_grad
    ):
        raise ValueError("output must match inputs and be contiguous without gradients")
    # Both tensors are contiguous, so byte ranges detect even partially overlapping views.
    input_start = inputs.data_ptr()
    output_start = output.data_ptr()
    byte_count = inputs.numel() * inputs.element_size()
    if input_start < output_start + byte_count and output_start < input_start + byte_count:
        raise ValueError("input and output storage must not overlap")
    with torch.cuda.device(inputs.device):
        _softmax_kernel[(rows,)](
            inputs, output, width, BLOCK_SIZE=triton.next_power_of_2(width), num_warps=num_warps
        )
    return output
