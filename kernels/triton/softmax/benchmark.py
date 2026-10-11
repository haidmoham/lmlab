"""Shape sweep with preallocated outputs and synchronized CUDA-event timing."""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch

from .common import DTYPES, ROW_COUNTS, ROW_WIDTHS, assert_correct, require_cuda, torch_softmax


def time_batch(function, iterations):
    """Mean milliseconds per call in one stream batch; host launch gaps may be included."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    milliseconds = start.elapsed_time(end) / iterations
    if milliseconds <= 0:
        raise RuntimeError("nonpositive CUDA-event duration; increase --iterations")
    return milliseconds


def command_output(command, cwd=None):
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"
    return result.stdout.strip()


def environment(device):
    import triton

    root = Path(__file__).resolve().parents[3]
    properties = torch.cuda.get_device_properties(device)
    source_paths = [*Path(__file__).parent.glob("*.py"), root / "pyproject.toml", root / "uv.lock"]
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": properties.name,
        "device_index": device.index,
        "compute_capability": [properties.major, properties.minor],
        "total_vram_bytes": properties.total_memory,
        "multiprocessors": properties.multi_processor_count,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], root),
        "git_status": command_output(["git", "status", "--short"], root),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_paths)
        },
        "nvidia_smi_all_devices": command_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=index,uuid,name,driver_version,pstate,"
                    "temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.used"
                ),
                "--format=csv,noheader",
            ]
        ),
        "runtime_settings": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--rows", nargs="+", type=int, default=ROW_COUNTS)
    parser.add_argument("--widths", nargs="+", type=int, default=ROW_WIDTHS)
    parser.add_argument("--num-warps", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwrite a run"
    )
    args = parser.parse_args()
    if min(args.rows) < 1 or min(args.widths) < 1 or max(args.widths) > 8192:
        parser.error("rows must be positive; widths must be in [1, 8192]")
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 5:
        parser.error("warmup and iterations must be positive; use at least 5 repeats")
    if args.output.exists():
        parser.error("output already exists; choose a new run directory")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        parser.error("unset CUDA_LAUNCH_BLOCKING for representative asynchronous timing")
    device = require_cuda(args.device)
    from .kernel import triton_softmax

    metadata = environment(device)
    print(f"timing {metadata['gpu']} on {device}, {args.dtype}")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    measurements = []
    raw_samples = []
    with torch.inference_mode():
        for rows in args.rows:
            for width in args.widths:
                inputs = torch.randn(
                    (rows, width), device=device, dtype=DTYPES[args.dtype], generator=generator
                )
                torch_output = torch.empty_like(inputs)
                triton_output = torch.empty_like(inputs)
                functions = {
                    "torch": partial(torch_softmax, inputs, torch_output),
                    "triton": partial(
                        triton_softmax, inputs, triton_output, num_warps=args.num_warps
                    ),
                }
                # Compile, verify the actual timed paths, and warm both implementations.
                for function in functions.values():
                    assert_correct(inputs, function())
                    for _ in range(args.warmup):
                        function()
                torch.cuda.synchronize(device)
                samples = {"torch": [], "triton": []}
                for repeat in range(args.repeats):
                    order = ("torch", "triton") if repeat % 2 == 0 else ("triton", "torch")
                    for provider in order:
                        milliseconds = time_batch(functions[provider], args.iterations)
                        samples[provider].append(milliseconds)
                        raw_samples.append(
                            {
                                "rows": rows,
                                "width": width,
                                "repeat": repeat,
                                "provider": provider,
                                "milliseconds_per_call": milliseconds,
                            }
                        )
                torch_ms = statistics.median(samples["torch"])
                triton_ms = statistics.median(samples["triton"])
                # Minimum logical traffic: read input once, write output once.
                logical_bytes = 2 * inputs.numel() * inputs.element_size()
                measurement = {
                    "rows": rows,
                    "width": width,
                    "dtype": args.dtype,
                    "torch_median_us": torch_ms * 1000,
                    "torch_min_us": min(samples["torch"]) * 1000,
                    "torch_max_us": max(samples["torch"]) * 1000,
                    "triton_median_us": triton_ms * 1000,
                    "triton_min_us": min(samples["triton"]) * 1000,
                    "triton_max_us": max(samples["triton"]) * 1000,
                    "torch_effective_gbps": logical_bytes / (torch_ms * 1e6),
                    "triton_effective_gbps": logical_bytes / (triton_ms * 1e6),
                    "speedup_torch_over_triton": torch_ms / triton_ms,
                }
                measurements.append(measurement)
                print(
                    f"{rows:5d} x {width:4d} | torch {torch_ms * 1000:8.3f} us | "
                    f"triton {triton_ms * 1000:8.3f} us | {torch_ms / triton_ms:.3f}x"
                )
                del functions, inputs, torch_output, triton_output
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {**vars(args), "output": str(args.output)}
    report = {
        "environment": metadata,
        "settings": settings,
        "protocol": {
            "timing": "CUDA events; synchronized batch mean, median across interleaved repeats",
            "includes": "forward work and possible host launch gaps in the stream",
            "excludes": "allocation, JIT compilation, correctness, input generation, transfers",
            "cache": "same resident inputs/outputs; warm cache; no explicit cache flush",
            "bandwidth": "2 * rows * width * dtype_bytes / seconds; logical, not measured DRAM",
            "correctness": "every measured shape checked before timing; finite logits only",
        },
        "measurements": measurements,
        "raw_samples": raw_samples,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    print(f"saved {args.output / 'results.json'} and {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
