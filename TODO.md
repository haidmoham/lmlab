# lmlab roadmap

## NOW — first experiment

aim: one focused evening, about two hours after environment setup. produce a
trustworthy softmax measurement on the rtx 3080 (10 gb) and your own explanation.
a triton win is not required. run all commands from the repository root with this
change's branch checked out.

### 1. prerequisites and device verification (15–30 minutes)

- [ ] use linux x86_64, or a linux shell inside wsl2 on the nvidia workstation.
  native windows and macos are not targets. use python 3.12, uv, and the existing
  lockfile. run:

```bash
uv sync --locked --python 3.12
nvidia-smi
uv run --locked python -c "import torch, triton; from kernels.triton.softmax.common import require_cuda; d = require_cuda(); p = torch.cuda.get_device_properties(d); print('device:', d, p.name, 'vram GiB:', p.total_memory / 2**30); print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'triton:', triton.__version__)"
```

- [ ] confirm the selected device is the rtx 3080. if there are multiple gpus,
  pass `--device INDEX` to the check and benchmark. close other gpu workloads;
  record the driver, idle temperature, and power state. resolve driver/runtime
  failures before proceeding; do not silently use cpu or blindly upgrade packages.
  the first triton compilation may take longer than later calls.

### 2. predict before timing (10 minutes)

- [ ] complete the prediction section in `reports/experiments/softmax.md`. predict
  the winner for `(1, 128)`, `(4096, 128)`, `(4096, 1024)`, and `(4096, 2049)`.
  predict what changes between widths 2048 and 2049. estimate minimum logical
  bytes for float32 `(4096, 1024)` and a latency lower bound from a bandwidth
  assumption you name. explain the expected bottleneck, confidence, and what
  observation would change your prediction. these are your hypotheses.
- [ ] trace one row in `kernel.py`: padding mask, subtracting the maximum, fp32
  exponentiation/reduction, normalization, and masked store.

### 3. establish correctness (10 minutes)

```bash
uv run --locked python -m kernels.triton.softmax.check
uv run --locked pytest -q tests/test_softmax.py -rs
```

- [ ] the strict command must report **154 GPU cases passed**: two seeds × 36
  shapes × two dtypes, plus five extreme/tied-logit cases per dtype. pytest also
  covers preallocated outputs and invalid shape/layout/gradient/storage contracts.
  skipped gpu tests are not correctness evidence; stop if the strict command fails.
  element tolerances: fp32 `rtol=1e-4, atol=1e-6`; fp16 `rtol=2e-3, atol=2e-5`.
  row sums and finite/nonnegative probabilities have separate checks.

### 4. measure and repeat (20–30 minutes)

```bash
uv run --locked python -m kernels.triton.softmax.benchmark --output artifacts/softmax/run-1
uv run --locked python -m kernels.triton.softmax.benchmark --output artifacts/softmax/run-2
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --output artifacts/softmax/fp16-1
```

- [ ] inspect each `summary.csv` and `results.json`. defaults: row counts
  1/32/256/4096; widths 1/31/128/257/781/1024/2049/4096/8192; 25 warmups; 100 calls
  per batch; seven interleaved provider repeats. every timed shape is checked.
  retain median/min/max latency in microseconds, effective gb/s, and the ratio
  `torch_latency / triton_latency`. a ratio below one is a triton loss.
- [ ] compare the two fp32 runs. preserve losses and noise-sized differences.
  this is warm-cache forward timing with preallocated outputs. input generation,
  allocation, transfers, and JIT compilation are excluded. cuda events can include
  gaps between host launches. this is not backward, cuda-graph, or serving timing.
- [ ] if time permits, measure the padding boundary:

```bash
uv run --locked python -m kernels.triton.softmax.benchmark --rows 4096 --widths 2048 2049 --output artifacts/softmax/padding-1
```

the largest fp32 input is 128 mib; input plus two outputs use 384 mib, with extra
memory needed for correctness and the runtime. this leaves headroom below 10 gb
on an idle device. width above 8192 is intentionally unsupported: wide-row
occupancy/spilling requires another experiment. every run requires a fresh output
directory to protect prior evidence. raw files stay ignored; copy selected actual
measurements and exact artifact paths into the tracked report.

### 5. explain and finish (20 minutes)

- [ ] answer in your own words:
  1. why does subtracting the maximum preserve softmax and help stability?
  2. why load padded lanes as negative infinity and mask stores?
  3. how do few rows, short rows, and padded wide rows change likely bottlenecks?
  4. which predictions matched the data, and which failed?
  5. does the spread support a win, a loss, or an inconclusive comparison?
  6. why is effective bandwidth not measured physical dram traffic?
  7. what profiler evidence would distinguish your explanation from an alternative?
- [ ] **done:** strict gpu correctness passes; two fp32 sweeps and one fp16 sweep
  are saved with hardware/software/git/source provenance; the worksheet contains
  pre-run predictions, a small measured table retaining any observed loss or
  ambiguous case, and your explanation with an untested alternative. no invented
  speedup or serving claim. choose one next experiment from the evidence.

## NEXT — gpu fundamentals

do one milestone at a time under the same experiment contract. create directories
only when implementing runnable work; keep the question and short report with it.

| milestone | experiment | measurable deliverable and learning objective |
| --- | --- | --- |
| 1. bandwidth and roofline | sweep contiguous copy/vector add below and above cache capacity; compare softmax logical traffic | bytes, latency, effective gb/s, and a roofline sketch with measured bandwidth and a stated compute assumption; distinguish launch/memory/compute limits |
| 2. reductions and coalescing | compare row/column sums, contiguous/strided layouts, padded widths, and equal element counts | correctness, shape/layout sweep and memory-transaction counters; explain adjacent-thread access, reduction work, and masking |
| 3. tiled matmul | compare a small tiled triton/cuda matmul with `torch.matmul` under explicit dtype/tf32 settings | rectangular/aligned/unaligned shape and tile sweeps, latency/flops/tolerances; explain reuse, arithmetic intensity, and tensor-core eligibility |
| 4. registers and occupancy | vary softmax `--num-warps` 4/8/16 around power-of-two boundaries | matched-correctness table, compiler register counts, occupancy/spill counters; explain why more warps or occupancy need not help |
| 5. fusion and profiling | compare separate/fused bias + activation, then profile a transformer block | outputs, launches, memory traffic and operator/block latency; explain which removed intermediates matter end to end |

use nsight compute for kernel counters and nsight systems/pytorch profiler for
timelines when available. profile separately from benchmark timing. do not choose
an explanation from timing alone.

## LATER — inference engineering

budget each experiment on one gpu. fix model revision, tokenizer, prompts, output
lengths, dtype, runtime settings, and request distribution. estimate weights,
workspace, and kv-cache memory before loading; start with a small model leaving
measured headroom within 10 gb.

| milestone | experiment and measurable deliverable | learning objective |
| --- | --- | --- |
| 1. prefill versus decode | split a fixed workload into prefill and single-token decode; sweep prompt length/batch size and record latency/peak memory | distinguish parallel prompt processing from repeated token steps and identify each limiter |
| 2. kv cache and attention | derive bytes per token from layers/kv heads/head dimension/dtype; compare with memory measurements over cached lengths | explain reuse, grouped-query attention, memory growth, and attention work |
| 3. quantization | compare supported fp16 and quantized small models on fixed prompts with a defined quality check | memory, latency/throughput and quality deltas; separate weight savings from speed and activation/cache precision |
| 4. continuous batching and scheduling | compare serial, fixed-batch, and continuous-batch execution on a saved mixed-length trace | queue time, utilization and latency distributions; explain head-of-line blocking and fairness under bounded memory |
| 5. vllm or sglang | choose one compatible runtime and benchmark a pinned small model with a fixed trace | reproducible version/configuration/baseline; distinguish engine changes from changed workloads |
| 6. service metrics | sweep offered load and concurrency against the baseline | ttft, inter-token latency, output tokens/s, completed requests/s, p50/p99, errors and sample counts; distinguish saturation from faster requests |
| 7. ai-generated kernels | request one candidate within a fixed evaluation/time budget; include held-out shapes and adversarial inputs | preserve prompt/model/source/failure provenance, correctness and latency; independently explain accepted changes without an orchestration framework |
| 8. end-to-end evaluation | compare baseline/optimized servers under identical traces, warmup/model/memory budgets | repeatable latency/throughput/quality/errors including tokenization and queueing; test whether kernel gains matter to users |

defer distributed inference until single-gpu measurement, memory budgeting, and
scheduling fundamentals are solid. earlier encoder/cross-attention/warmup/dropout
work remains pending in `experiments/attention-resume.md`; this roadmap does not
implement it. historical pre-norm results are not a current post-norm baseline.

## RESOURCES

read the relevant section while doing its experiment, not everything first.

- **NOW / NEXT 2:** [triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html),
  especially [fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
  trace masking and reductions; this lab starts with one program per row.
- **NEXT 1–5:** [wafer gpu performance engineering resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)
  and [gpu mode lectures](https://github.com/gpu-mode/lectures). choose the memory,
  reduction, or occupancy material for the experiment you are measuring.
- **NEXT 3:** [cuda matmul optimization](https://siboehm.com/articles/22/CUDA-MMM).
  use its incremental tiling comparisons; published numbers are not rtx 3080 evidence.
- **LATER 1–4, 6, 8:** [baseten inference engineering](https://www.baseten.co/inference-engineering/book/00-inference/).
  connect its inference/service concepts to your measurements.
- **LATER 5–6:** [vllm documentation](https://docs.vllm.ai/). verify the chosen
  release's hardware/model support and commands when implementing the experiment.
- **LATER 7:** [kernelbench](https://github.com/ScalingIntelligence/KernelBench).
  borrow correctness and held-out evaluation ideas before adopting tooling.
