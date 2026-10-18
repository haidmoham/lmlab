# softmax experiment worksheet

status: the bounded GPU experiment completed on 2026-10-18. all correctness gates passed. two fp32 sweeps and one fp16 sweep produced measurements for 36 shapes each. these are agent-recorded observations. the performance verdict is inconclusive as a general comparison because several fp32 comparisons reverse direction, batch timings vary, and no bottleneck profile was collected.

the researcher prediction fields below remain as supplied, blank. measurements are now visible, so a prediction written afterward would be retrospective. the interpretation section is unchanged.

## before running — researcher

- question/hypothesis:
- winner predictions for `(1, 128)`, `(4096, 128)`, `(4096, 1024)`, `(4096, 2049)`:
- predicted difference between widths 2048 and 2049, with reason:
- minimum logical bytes for float32 `(4096, 1024)`:
- bandwidth assumption, source/measurement, and latency lower bound:
- likely limiter, confidence, and observation that would change my mind:

## controls and budget

the completed sweep used the existing kernel and benchmark without modification. rows were `1, 32, 256, 4096`; widths were `1, 31, 128, 257, 781, 1024, 2049, 4096, 8192`. each shape used seed 0, four warps, finite seeded inputs, independent preallocated outputs, 25 warmup calls, and seven interleaved batches of 100 calls per provider. the providers alternated order in each batch. both fp32 sweeps used the same settings and source hashes. fp16 changed only dtype and output directory.

CUDA events measured synchronized stream batch means. allocation, input generation, transfers, JIT compilation, and correctness checks were outside timing. the same resident inputs and outputs were reused with warm caches and no explicit flushing. timings can include host launch gaps. minima and maxima describe the seven batch means, not individual request latency tails. the scope is a forward softmax microbenchmark, with no end-to-end inference measurement.

## correctness and provenance

- `uv sync --locked --python 3.12` passed in the isolated worktree. an initial non-login-shell launch could not find `uv`; the failed log was preserved and the existing executable was added to the launch script's PATH before retrying.
- the small `(1, 128)` softmax smoke passed for float32 and float16 before the strict gate. after the user resumed work later on 2026-10-18, that same small smoke passed again in both dtypes. the sweeps were not repeated on resume.
- `uv run --locked python -m kernels.triton.softmax.check` passed all 154 GPU cases, including both dtypes, seeds 0 and 42, and adversarial inputs.
- `uv run --locked pytest -q tests/test_softmax.py -rs` passed all 45 tests, with no GPU skips. benchmark-time checks also validated both actual preallocated timed paths for all 108 measured shape/run combinations.
- element tolerances were fp32 `rtol=1e-4, atol=1e-6` and fp16 `rtol=2e-3, atol=2e-5`, compared with PyTorch and a float64 oracle. outputs also had to be finite and nonnegative, with fp32-accumulated row sums within `1e-5` for fp32 or `1e-3` for fp16. no correctness failure occurred.
- the measured source revision was `e107dbf8b7fc25488b74a45436b1ce38bf5bd8e6`. all three results recorded clean Git status before this report update. the six source/lockfile hashes agree across runs and with the unchanged files at audit time.

| environment | recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080, device 0, compute capability 8.6, 68 multiprocessors |
| memory | driver reports 10240 MiB; PyTorch reports 10736762880 bytes |
| driver | 610.88 |
| Linux | 6.18.33.2-microsoft-standard-WSL2, x86_64, glibc 2.43 |
| Python | 3.12.13 |
| PyTorch / CUDA runtime | 2.13.0+cu130 / 13.0 |
| Triton | 3.7.1 |
| runtime flags | `CUDA_VISIBLE_DEVICES`, `CUDA_MODULE_LOADING`, and `CUDA_LAUNCH_BLOCKING` unset; deterministic algorithms false; TF32 matmul false |

the benchmark captures one GPU telemetry snapshot at the start of each sweep. it does not continuously monitor clocks or other workloads.

| run | UTC timestamp | pstate | temperature C | SM / memory clock MHz | power draw / limit W | memory used MiB |
| --- | --- | --- | --- | --- | --- | --- |
| run-1 | 00:57:07.772424 | P2 | 71 | 1710 / 9501 | 126.45 / 320.00 | 2698 |
| run-2 | 00:57:14.518160 | P2 | 71 | 1710 / 9501 | 126.99 / 320.00 | 2797 |
| fp16-1 | 00:57:20.418940 | P2 | 73 | 1905 / 9501 | 155.57 / 320.00 | 2702 |

all timestamps in that table are on 2026-10-18. concurrent desktop workload was not controlled. these snapshots cannot establish stable clocks or isolated GPU use. no performance-counter permissions or machine settings were changed.

## measured observations

the table contains all four requested focus shapes for every completed run. raw evidence retains the full shape sweep, including every loss and ambiguous result. the ratio is median PyTorch time divided by median Triton time. a value below one means Triton's measured median was higher. ratios are observations for this harness and run, not a general speedup claim. effective bandwidth is minimum logical traffic divided by measured time, in decimal GB/s; it is not measured DRAM traffic or utilization.

| shape / dtype / run | PyTorch median [min, max] us | Triton median [min, max] us | effective GB/s, PyTorch / Triton | PyTorch/Triton ratio | repeat-run observation |
| --- | --- | --- | --- | --- | --- |
| 1 x 128 / fp32 / run-1 | 3.973 [3.359, 7.639] | 14.879 [12.513, 36.823] | 0.258 / 0.069 | 0.267 | higher Triton median in both runs |
| 4096 x 128 / fp32 / run-1 | 9.021 [5.949, 10.690] | 25.989 [19.354, 42.271] | 464.926 / 161.387 | 0.347 | higher Triton median in both runs |
| 4096 x 1024 / fp32 / run-1 | 70.912 [52.091, 99.420] | 64.079 [51.670, 92.733] | 473.184 / 523.644 | 1.107 | direction reverses; inconclusive |
| 4096 x 2049 / fp32 / run-1 | 191.254 [172.858, 248.678] | 153.539 [137.103, 178.967] | 351.060 / 437.295 | 1.246 | lower Triton median in both runs |
| 1 x 128 / fp32 / run-2 | 3.523 [3.297, 9.339] | 13.385 [12.902, 25.846] | 0.291 / 0.077 | 0.263 | higher Triton median in both runs |
| 4096 x 128 / fp32 / run-2 | 5.857 [5.253, 6.840] | 16.526 [14.479, 36.702] | 716.084 / 253.799 | 0.354 | higher Triton median in both runs |
| 4096 x 1024 / fp32 / run-2 | 54.600 [52.183, 81.951] | 77.998 [54.108, 95.027] | 614.554 / 430.196 | 0.700 | direction reverses; inconclusive |
| 4096 x 2049 / fp32 / run-2 | 186.255 [172.943, 193.025] | 138.568 [134.543, 155.863] | 360.482 / 484.540 | 1.344 | lower Triton median in both runs |
| 1 x 128 / fp16 / fp16-1 | 5.079 [3.645, 6.236] | 16.711 [12.104, 34.796] | 0.101 / 0.031 | 0.304 | one sweep only |
| 4096 x 128 / fp16 / fp16-1 | 8.909 [4.987, 13.752] | 28.754 [24.187, 45.844] | 235.402 / 72.934 | 0.310 | one sweep only |
| 4096 x 1024 / fp16 / fp16-1 | 56.617 [28.590, 58.391] | 39.926 [27.352, 42.250] | 296.330 / 420.210 | 1.418 | one sweep only |
| 4096 x 2049 / fp16 / fp16-1 | 179.313 [163.468, 210.964] | 59.105 [51.640, 85.699] | 187.219 / 567.983 | 3.034 | one sweep only |

fp32 `(4096, 1024)` changed from ratio `1.107` in run-1 to `0.700` in run-2. its batch ranges overlap within both runs. this comparison is inconclusive and is not reported as a speedup.

six of the 36 fp32 shape comparisons changed direction between sweeps: `(256, 4096)`, `(4096, 257)`, `(4096, 781)`, `(4096, 1024)`, `(4096, 4096)`, and `(4096, 8192)`. fp32 `(4096, 2049)` had a lower Triton median in both sweeps, with ratios `1.246` and `1.344`; the batch ranges overlap in run-1. the two width-128 focus shapes had higher Triton medians in both fp32 sweeps. fp16 has one sweep only, so repeat-run agreement is untested for that dtype.

## validation and limitations

the independent CPU audit recomputed medians, minima, maxima, logical bandwidth, and ratios from all 1512 raw batch samples. all samples were positive and finite. all 108 JSON measurement rows matched their CSV rows. the audit also checked the complete shape grids, seven samples per provider per shape, settings, recorded commit and clean status, source hashes, and gate logs. the audit receipt includes hashes of the six raw result files so later changes can be detected.

the bounded budget is complete. no additional benchmark sweep was launched on resume. profiling was not part of these completed runs, so the limiting resource remains unverified. the results do not separate kernel execution from host launch gaps, do not measure DRAM traffic, and do not establish a serving benefit. differences between fp16 and fp32 are not a matched repeatability claim. width 2048 is not in the existing sweep, so the 2048-versus-2049 prediction cannot be tested from these data.

the next experimental decision remains with the researcher. a separate follow-up could select a controlled repeat or profile to distinguish timing variability from a kernel effect. no such follow-up was run here.

independent numerical review found no material errors. the committed CSV records and verifier reproduce the numerical audit without host-specific launch scripts. researcher interpretation remains open.

## evidence and reproduction

the original execution ran from `2026-10-18T00:56:45+00:00` through `2026-10-18T00:57:24+00:00`. the compact [measurements and all batch samples](softmax-results/README.md) retain all three runs under `two-way/` identifiers. [provenance.json](softmax-results/provenance.json) records settings, sanitized environment snapshots, exact measured-source hashes, and hashes of the original JSON/CSV files. full raw artifacts remain local and ignored by Git.

verify the committed evidence on CPU:

```bash
uv run --locked python reports/experiments/softmax-results/verify.py
```

the source for this historical two-provider experiment is commit `e107dbf8b7fc25488b74a45436b1ce38bf5bd8e6`. reproduce from that revision in a separate checkout with fresh output directories. that revision used seven repeats, a 154-case strict gate, and only PyTorch/Triton. the current main uses three providers, 12 balanced repeats, and a 462-provider-case gate; running its defaults does not reproduce this historical protocol.

the original sweep commands at the recorded revision were:

```bash
uv sync --locked --python 3.12
uv run --locked python -m kernels.triton.softmax.check
uv run --locked pytest -q tests/test_softmax.py -rs
uv run --locked python -m kernels.triton.softmax.benchmark --output artifacts/softmax/reproduction/run-1
uv run --locked python -m kernels.triton.softmax.benchmark --output artifacts/softmax/reproduction/run-2
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --output artifacts/softmax/reproduction/fp16-1
```

## interpretation — researcher

- which predictions held/failed? cite measured rows.
- possible mechanism for a gain/regression: distinguish evidence from inference.
- alternative explanation and profiler evidence that would discriminate:
- why effective bandwidth does not establish dram utilization:
- what this cannot establish about serving:
- one next experiment, changed variable, controls, budget, and decision criterion:

agent contributions: initial kernel/tests/harness and proposed hypothesis.
researcher prediction, explanation, profiler evidence, and conclusions: pending.
