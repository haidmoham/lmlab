# three-way softmax experiment

the requested comparison is complete. our stable, unfused eager implementation had a higher measured median than both `torch.softmax` and fused Triton in all 144 shape/run records. these are observations from this warm-allocation CUDA-event harness. they do not establish that fusion alone caused the differences or identify a bottleneck. PyTorch versus Triton remains shape-dependent, with repeat-run reversals and overlapping ranges retained below.

## implementations and numerical policy

all providers receive the same finite input tensor within a run and independent preallocated final outputs. the eager implementation in `kernels/triton/softmax/common.py` is ordinary eager tensor code:

```python
values = inputs.float()
maxima = values.amax(dim=-1, keepdim=True)
shifted = values - maxima
numerators = shifted.exp()
denominators = numerators.sum(dim=-1, keepdim=True)
return torch.div(numerators, denominators, out=output)
```

FP16 is promoted before the maximum and subtraction as well as the exponential, sum, and division. only the final output rounds to FP16. FP32 stays FP32. division writes directly to the supplied output, with no extra final copy kernel. casts and intermediate tensor creation remain inside the eager callable. it does not invoke `torch.softmax`, `torch.compile`, or a JIT compiler.

Triton's kernel is unchanged and also uses FP32 arithmetic before final output conversion. the PyTorch baseline calls `torch.softmax(inputs, dim=-1, out=output)` with its normal internal dispatch and accumulation policy. this is a stable FP32-promoted eager baseline, not a naive FP16-intermediate formula. no provider was selectively tuned, and no researcher prediction was invented.

## checks and bounded protocol

- locked environment sync and Ruff checks passed.
- the small `(1, 128)` GPU smoke passed for all three providers in both dtypes.
- the strict gate passed 462 provider cases, 154 per provider, against identical shapes/seeds/adversarial tensors. it checked the original tolerances, a float64 oracle, finiteness, nonnegativity, row sums, and exact supplied-output object/storage reuse.
- `uv run --locked pytest -q tests -k softmax -rs` passed 86 tests with 21 unrelated tests deselected. tests include FP16 extreme finite values, output reuse, final-only rounding, and order balance.
- each actual preallocated timed callable also passed correctness and output identity checks before warmup/timing on all 144 shape/run records.

the unchanged tolerances are FP32 `rtol=1e-4, atol=1e-6` and FP16 `rtol=2e-3, atol=2e-5`. row sums accumulate in FP32 and use `rtol=0`, with `atol=1e-5` or `1e-3` respectively. the strict gate uses seeds 0 and 42; benchmark seed is 0.

the new namespace is `artifacts/softmax/three-way-20261018/`. two FP32 sweeps and two FP16 sweeps each cover rows `1, 32, 256, 4096` and widths `1, 31, 128, 257, 781, 1024, 2049, 4096, 8192`. controls are four Triton warps, 25 warmup calls, 12 repeats, and 100 calls per provider per repeat. all six provider permutations occur twice per shape. each provider occupies each position four times; each pair precedes the other six times. order offsets are 0, 3, 1, and 4 for the listed runs. full orders and raw positions are saved.

each CUDA-event interval ends with synchronization and produces a batch mean. input generation, transfers, final output allocation, Triton compilation, and correctness are excluded. GPU casts, eager primitive kernels, and their intermediate traffic are included. Python and cached allocator CPU latency are reflected only through stream gaps and are not reliably timed in full. caches and allocator are warm, with no flushing. these are not isolated kernel durations, cold-call wall times, or request latency percentiles.

## measured results

all times below are microseconds. brackets contain the minimum and maximum of 12 batch means, not confidence intervals. ratios divide provider medians within the same run; values above one indicate a lower denominator median. the [complete provider measurements](softmax-results/measurements.csv) retain every shape and all three implementations, including logical I/O-equivalent bandwidth.

| run / dtype | shape | PyTorch median [min, max] us | eager median [min, max] us | Triton median [min, max] us | eager/Triton | eager/PyTorch | PyTorch/Triton |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32-1 | 1 x 128 | 4.879 [3.727, 12.421] | 43.163 [35.729, 82.061] | 16.829 [12.889, 30.044] | 2.565 | 8.846 | 0.290 |
| fp32-1 | 4096 x 128 | 5.089 [4.956, 9.861] | 42.967 [34.108, 44.166] | 14.623 [13.834, 17.367] | 2.938 | 8.443 | 0.348 |
| fp32-1 | 4096 x 1024 | 59.822 [57.416, 73.032] | 253.568 [243.999, 284.076] | 65.111 [59.238, 72.275] | 3.894 | 4.239 | 0.919 |
| fp32-1 | 4096 x 2049 | 163.671 [154.419, 164.813] | 479.016 [476.744, 491.794] | 120.064 [116.562, 121.651] | 3.990 | 2.927 | 1.363 |
| fp32-2 | 1 x 128 | 5.232 [3.318, 12.206] | 55.376 [36.332, 59.055] | 15.181 [12.421, 19.108] | 3.648 | 10.584 | 0.345 |
| fp32-2 | 4096 x 128 | 8.397 [5.120, 15.462] | 67.814 [39.845, 109.394] | 20.639 [9.974, 47.350] | 3.286 | 8.076 | 0.407 |
| fp32-2 | 4096 x 1024 | 59.838 [59.628, 62.925] | 247.787 [242.278, 252.232] | 59.500 [59.116, 78.244] | 4.165 | 4.141 | 1.006 |
| fp32-2 | 4096 x 2049 | 163.697 [158.198, 164.844] | 481.434 [474.788, 489.810] | 119.921 [115.753, 123.269] | 4.015 | 2.941 | 1.365 |
| fp16-1 | 1 x 128 | 6.798 [3.318, 13.158] | 92.759 [44.443, 137.226] | 21.433 [13.253, 42.404] | 4.328 | 13.645 | 0.317 |
| fp16-1 | 4096 x 128 | 5.908 [4.943, 9.832] | 53.003 [49.262, 94.268] | 17.874 [13.701, 25.784] | 2.965 | 8.971 | 0.331 |
| fp16-1 | 4096 x 1024 | 32.553 [28.324, 36.288] | 293.836 [289.587, 297.472] | 32.138 [27.156, 37.734] | 9.143 | 9.026 | 1.013 |
| fp16-1 | 4096 x 2049 | 152.407 [145.062, 154.928] | 556.809 [547.789, 563.722] | 59.407 [59.075, 65.014] | 9.373 | 3.653 | 2.565 |
| fp16-2 | 1 x 128 | 4.075 [3.369, 12.308] | 49.714 [38.954, 94.075] | 13.859 [11.776, 19.414] | 3.587 | 12.200 | 0.294 |
| fp16-2 | 4096 x 128 | 5.089 [4.731, 10.107] | 52.586 [43.889, 113.868] | 16.384 [11.633, 25.743] | 3.210 | 10.333 | 0.311 |
| fp16-2 | 4096 x 1024 | 33.367 [28.488, 41.009] | 291.814 [286.269, 299.141] | 33.357 [27.156, 59.341] | 8.748 | 8.746 | 1.000 |
| fp16-2 | 4096 x 2049 | 148.075 [144.681, 153.405] | 557.195 [549.632, 566.804] | 60.078 [56.484, 60.680] | 9.275 | 3.763 | 2.465 |

for FP32 `(4096, 1024)`, eager/Triton was `3.894` and `4.165`, while PyTorch/Triton was `0.919` and `1.006`. the latter changed direction and the PyTorch/Triton ranges overlap in both sweeps, so no advantage is established between those optimized paths for this shape. FP16 `(4096, 1024)` was also effectively tied between PyTorch and Triton at ratios `1.013` and `1.000`, with overlapping ranges; eager/Triton was `9.143` and `8.748`.

for `(4096, 2049)`, the observed eager/Triton ratios were `3.990` and `4.015` in FP32, and `9.373` and `9.275` in FP16. at the two width-128 focus shapes, optimized PyTorch had the lowest median in every run. no global Triton-over-PyTorch winner follows from these data.

PyTorch/Triton direction reversals across the two same-dtype sweeps occurred at FP32 `(256, 4096)`, `(4096, 781)`, and `(4096, 1024)`, and FP16 `(256, 8192)`. eager/PyTorch and eager/Triton had no direction reversals. the audit recomputed all medians, minima, maxima, bandwidth values, and ratios from 5184 positive finite raw samples and verified every CSV row and the balanced schedules.

## environment, provenance, and limits

the GPU was `NVIDIA GeForce RTX 3080`, device 0, compute capability 8.6, with driver 610.88 and 10240 MiB reported by `nvidia-smi`. software was Python `3.12.13`, PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, and Triton `3.7.1` in WSL2 Ubuntu. the measured source is based on `e107dbf8b7fc25488b74a45436b1ce38bf5bd8e6` with uncommitted experiment changes on `codex/softmax-desktop-20261018`; hashes and dirty status are recorded in each result. the committed [measured-source patch](softmax-results/three-way-measured.patch) reconstructs the exact source from the recorded base revision. source hashes agree across all four runs.

execution started at `2026-10-18T23:04:22+00:00` and completed at `2026-10-18T23:05:36+00:00`. sanitized per-run temperature, clock, power, and memory snapshots are in [provenance.json](softmax-results/provenance.json). snapshots were taken only at the start of each sweep.

desktop load was not isolated, and snapshots do not establish steady clocks or absence of competing work. GPU counters and a limiting-resource profile were not collected. differences can reflect Python launch pacing, implementation dispatch, approximate exponentials, allocations, precision policy, and memory traffic; these runs do not separate their contributions. no speedup is attributed solely to fusion or claimed for end-to-end serving.

reported bandwidth uses `2 * rows * width * input_dtype_bytes / seconds`, the minimum input/output traffic. it is logical I/O-equivalent bandwidth, not measured DRAM bandwidth. it substantially omits eager FP32 intermediates and the FP16-to-FP32 cast traffic, so it cannot establish memory efficiency or hardware bandwidth saturation. warmed allocator/cache state is part of this protocol. repeated same-dtype sweeps are the replication evidence; cross-dtype comparisons do not use identical quantized input values and are not used to claim a precision effect.

Ruff formatting was applied after timing to two source files. exact measured files were saved first, and AST equality was verified for every experiment/test Python file. no behavior changed and no extra sweep was run for whitespace changes. the formatting receipt distinguishes measured hashes from formatted working-source hashes. all six earlier two-provider result-file hashes remain unchanged, and the earlier report is preserved.

## reproduction and artifacts

the repository contains the eager implementation, unchanged Triton kernel, checker, benchmark, tests, and locked dependencies. [compact committed evidence](softmax-results/README.md) includes every provider summary and all raw batch means. its CPU verifier reconstructs the exact measured source and checks AST equivalence with the formatted source. from the repository root, reproduce with fresh output paths:

```bash
uv sync --locked --python 3.12
uv run --locked python reports/experiments/softmax-results/verify.py
uv run --locked python -m kernels.triton.softmax.check
uv run --locked pytest -q tests -k softmax -rs
uv run --locked python -m kernels.triton.softmax.benchmark --repeats 12 --order-offset 0 --output artifacts/softmax/three-way-20261018/fp32-1
uv run --locked python -m kernels.triton.softmax.benchmark --repeats 12 --order-offset 3 --output artifacts/softmax/three-way-20261018/fp32-2
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --repeats 12 --order-offset 1 --output artifacts/softmax/three-way-20261018/fp16-1
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --repeats 12 --order-offset 4 --output artifacts/softmax/three-way-20261018/fp16-2
```

the current defaults are already 12 repeats. the harness rejects existing result directories. use a new namespace or fresh checkout when reproducing. the module commands above are portable; machine-specific launch scripts remain local. no WSL, driver, security, credential, or paid-compute setup was changed.

- [all provider measurements](softmax-results/measurements.csv), including every shape and dispersion.
- [all raw batch means](softmax-results/samples.csv), including provider order positions.
- [sanitized settings, environment, hashes, and source reconstruction](softmax-results/provenance.json).
- [CPU verifier and evidence schema](softmax-results/README.md).
- [earlier two-provider report](softmax.md), retained separately.

## interpretation reserved for the researcher

predictions, bottleneck explanation, and the central research argument remain for the user. the observations above are agent-recorded evidence. no further tuning or follow-up experiment was run.
