"""Strict GPU correctness entry point; absence of a GPU is a failure, not a skip."""

import argparse
from functools import partial

import torch

from .common import (
    DTYPES,
    ROW_COUNTS,
    ROW_WIDTHS,
    adversarial_inputs,
    assert_correct,
    eager_softmax,
    require_cuda,
    torch_softmax,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-warps", type=int, choices=(4, 8, 16), default=4)
    args = parser.parse_args()
    device = require_cuda(args.device)
    from .kernel import triton_softmax

    print(f"checking device {device}: {torch.cuda.get_device_name(device)}")
    case_counts = {"torch": 0, "eager": 0, "triton": 0}

    def check_case(inputs):
        outputs = {provider: torch.empty_like(inputs) for provider in case_counts}
        functions = {
            "torch": partial(torch_softmax, inputs, outputs["torch"]),
            "eager": partial(eager_softmax, inputs, outputs["eager"]),
            "triton": partial(triton_softmax, inputs, outputs["triton"], num_warps=args.num_warps),
        }
        for provider, function in functions.items():
            actual = function()
            if actual is not outputs[provider] or actual.data_ptr() != outputs[provider].data_ptr():
                raise AssertionError(f"{provider} did not reuse its preallocated output")
            assert_correct(inputs, actual)
            case_counts[provider] += 1

    with torch.inference_mode():
        for dtype in DTYPES.values():
            for seed in (0, 42):
                generator = torch.Generator(device=device).manual_seed(seed)
                for rows in ROW_COUNTS:
                    for width in ROW_WIDTHS:
                        inputs = torch.randn(
                            (rows, width), device=device, dtype=dtype, generator=generator
                        )
                        check_case(inputs)
            for _, inputs in adversarial_inputs(device, dtype):
                check_case(inputs)
    torch.cuda.synchronize(device)
    counts = ", ".join(f"{provider}={count}" for provider, count in case_counts.items())
    print(f"passed {sum(case_counts.values())} GPU provider cases ({counts}; float32 and float16)")


if __name__ == "__main__":
    main()
