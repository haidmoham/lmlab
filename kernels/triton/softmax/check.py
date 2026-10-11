"""Strict GPU correctness entry point; absence of a GPU is a failure, not a skip."""

import argparse

import torch

from .common import (
    DTYPES,
    ROW_COUNTS,
    ROW_WIDTHS,
    adversarial_inputs,
    assert_correct,
    require_cuda,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-warps", type=int, choices=(4, 8, 16), default=4)
    args = parser.parse_args()
    device = require_cuda(args.device)
    from .kernel import triton_softmax

    print(f"checking device {device}: {torch.cuda.get_device_name(device)}")
    case_count = 0
    with torch.inference_mode():
        for dtype in DTYPES.values():
            for seed in (0, 42):
                generator = torch.Generator(device=device).manual_seed(seed)
                for rows in ROW_COUNTS:
                    for width in ROW_WIDTHS:
                        inputs = torch.randn(
                            (rows, width), device=device, dtype=dtype, generator=generator
                        )
                        assert_correct(inputs, triton_softmax(inputs, num_warps=args.num_warps))
                        case_count += 1
            for _, inputs in adversarial_inputs(device, dtype):
                assert_correct(inputs, triton_softmax(inputs, num_warps=args.num_warps))
                case_count += 1
    torch.cuda.synchronize(device)
    print(f"passed {case_count} GPU correctness cases (float32 and float16)")


if __name__ == "__main__":
    main()
