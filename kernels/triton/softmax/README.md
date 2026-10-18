# three-way softmax comparison

question: how do an explicitly unfused eager softmax, optimized `torch.softmax`, and the existing fused Triton softmax compare over row counts, widths, and dtypes on an RTX 3080?

the original agent-proposed hypothesis was that a register-resident reduction with one input read and one output write might do well at moderate widths and enough rows, while few rows, padding, register pressure, or PyTorch's tuned dispatch might favor PyTorch. no shape-specific researcher prediction was supplied. the three-way extension adds the user-requested eager baseline; it does not assume a winner.

## implementation and numerical policy

- `common.py` contains optimized `torch_softmax`, unfused `eager_softmax`, tolerances, the CUDA gate, and correctness checks.
- `kernel.py` contains the unchanged one-program-per-row fused Triton candidate.
- `check.py` checks all three providers on the same 154 inputs each, 462 provider cases total.
- `benchmark.py` retains raw CUDA-event samples, provider order, summaries, and source/environment metadata.

the eager candidate uses separate, uncompiled tensor operations. it promotes FP16 inputs to FP32 before computing the row maximum, subtraction, exponential, sum, and division. FP32 inputs stay FP32. the final `torch.div` writes directly into the preallocated original-dtype output, combining division and final FP16 rounding without an extra copy kernel. it never calls `torch.softmax`, `torch.compile`, or a JIT compiler.

Triton also computes its reductions and normalization in FP32 before its final output conversion. PyTorch's internal accumulator and dispatch policy are not overridden. the FP16 eager baseline is therefore a stable FP32-promoted reference, not a sequence of FP16 intermediate tensors. differences cannot be attributed solely to fusion.

all three timed paths receive the same input and independent preallocated final outputs. eager casts and intermediate tensors are created inside its callable. the event interval includes their GPU operators and traffic; Python and cached allocator CPU overhead appear only through stream gaps and are not reliably measured in full.

the GPU experiment accepts finite, contiguous, nonempty 2D CUDA tensors in float32 or float16 with width at most 8192. the forward-only candidates reject gradients and overlapping input/output storage. eager also permits CPU tensors for numerical unit tests; benchmark and strict checker still require an NVIDIA CUDA GPU and never fall back to CPU.

element tolerances are FP32 `rtol=1e-4, atol=1e-6` and FP16 `rtol=2e-3, atol=2e-5`. the gate compares each provider with PyTorch and a float64 oracle, requires finite nonnegative probabilities, and checks FP32-accumulated row sums with `rtol=0` and `atol=1e-5` for FP32 or `1e-3` for FP16. seeds 0 and 42 cover every shape. adversarial inputs include tied maxima, large offsets, wide dynamic ranges, and a maximum in the last real lane. exact output object and storage reuse are checked before timing.

## controls and timing

rows are `1, 32, 256, 4096`. widths are `1, 31, 128, 257, 781, 1024, 2049, 4096, 8192`. the largest FP32 input is 128 MiB. input plus three final outputs use 512 MiB, before eager intermediates, correctness oracles, and allocator state. this remains a bounded forward microbenchmark within the RTX 3080's 10 GB budget.

the default uses seed 0, four Triton warps, 25 warmup calls, and 12 interleaved repeats of 100 calls per provider. every block covers all six provider permutations. each provider appears equally often at every position, and each pair precedes the other equally often. `--repeats` must be a positive multiple of six; `--order-offset` rotates the schedule. JSON records full repeat orders and each sample's zero-based order position.

CUDA events measure synchronized stream batch means. generation, transfers, correctness, Triton compilation, and final output allocation are excluded. eager intermediates and promotion are inside its call. caches and allocator are warm, with resident inputs/outputs and no explicit flushing. results can include host launch gaps; they are not isolated kernel durations, cold-call wall times, or request latency percentiles. `CUDA_LAUNCH_BLOCKING` must be unset or zero.

`summary.csv` reports each provider's median/min/max microseconds, logical I/O-equivalent GB/s, and the median ratios `torch/triton`, `eager/triton`, and `eager/torch`. a ratio above one means the denominator had a lower measured median. `2 * rows * width * dtype_bytes / seconds` counts only minimum input/output traffic. it is not measured DRAM bandwidth and understates eager FP32 intermediate traffic, especially with FP16 inputs.

`results.json` also retains every raw batch sample, settings, hardware/software versions, source/lockfile hashes, dirty Git status, runtime flags, and a start-of-run GPU telemetry snapshot. dispersion and repeat-run disagreement must accompany any performance observation. no inference-serving benefit or bottleneck explanation follows from these timings alone.

## reproduce from the repository root

Linux or WSL2, Python 3.12, an existing NVIDIA driver, and CUDA-enabled PyTorch are required. the locked environment supplies the compatible Triton dependency. use new output directories to preserve every prior run.

```bash
uv sync --locked --python 3.12
uv run --locked python -m kernels.triton.softmax.check
uv run --locked pytest -q tests -k softmax -rs
uv run --locked python -m kernels.triton.softmax.benchmark --repeats 12 --order-offset 0 --output artifacts/softmax/three-way-20261018/fp32-1
uv run --locked python -m kernels.triton.softmax.benchmark --repeats 12 --order-offset 3 --output artifacts/softmax/three-way-20261018/fp32-2
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --repeats 12 --order-offset 1 --output artifacts/softmax/three-way-20261018/fp16-1
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --repeats 12 --order-offset 4 --output artifacts/softmax/three-way-20261018/fp16-2
```

the new report is [softmax-three-way.md](../../../reports/experiments/softmax-three-way.md). the earlier [two-provider report](../../../reports/experiments/softmax.md) and its raw files remain historical evidence from the original source revision. the default repeat count and provider set changed for this new experiment.

the Triton implementation follows the [fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html). persistent scheduling and autotuning are not added. timing uses [PyTorch CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).

## scaffold verification — 2026-10-11

on macos arm64 with python 3.12.13 and pytorch 2.13.0, `uv sync --locked --python
3.12`, `make test` (23 passed, 43 gpu tests skipped), `make lint`, and formatting
of the six new python files passed. syntax and local documentation links checked;
all four notebooks remain byte-identical to the committed baseline. the strict
check and benchmark both fail explicitly without cuda and create no result files.
the full repository formatting check has a pre-existing failure in
`notebooks/04_autograd.ipynb`, left unchanged. triton compilation, gpu correctness,
and latency/bandwidth measurements remain unverified until workstation execution.
