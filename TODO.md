# lmlab roadmap

## current status: paused after completed softmax comparisons

the completed work is recorded in the [two-provider report](reports/experiments/softmax.md) and [three-provider report](reports/experiments/softmax-three-way.md). no bandwidth calibration, roofline experiment, or profiler capture has been run. resume curriculum work only after a new request.

- [x] verified the existing Linux/WSL2 CUDA environment on the RTX 3080 with Python 3.12 and the locked dependencies.
- [x] completed the original PyTorch/Triton comparison: two FP32 sweeps and one FP16 sweep, 36 shapes each, seven interleaved repeats per provider, 154 strict GPU cases and 45 pytest tests.
- [x] added stable, unfused eager softmax and compared all three implementations: two FP32 sweeps and two FP16 sweeps, 36 shapes each, 12 balanced repeats per provider. the strict gate passed 462 provider cases, 154 each; 86 softmax tests passed.
- [x] preserved every measured shape, provider, range, and repeat-order observation. compact sanitized [measurement records and CPU verifier](reports/experiments/softmax-results/README.md) are committed beside the reports; full raw JSON and generated bundles stay local.
- [ ] researcher interpretation remains open. original pre-run prediction fields were blank; do not fill them retrospectively as predictions.
- [ ] measured bandwidth ceiling, limiting-resource profile, notebook presentation of softmax evidence, and any causal explanation of fusion remain pending.

## current reproduction protocol

these commands document reproduction, not an instruction to resume the paused experiments. run from the repository root on an existing compatible NVIDIA Linux/WSL2 setup, with new output directories:

```bash
uv sync --locked --python 3.12
uv run --locked python reports/experiments/softmax-results/verify.py
uv run --locked python -m kernels.triton.softmax.check
uv run --locked pytest -q tests/test_softmax.py -rs
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float32 --repeats 12 --order-offset 0 --output artifacts/softmax/reproduction/fp32-1
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float32 --repeats 12 --order-offset 3 --output artifacts/softmax/reproduction/fp32-2
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --repeats 12 --order-offset 1 --output artifacts/softmax/reproduction/fp16-1
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --repeats 12 --order-offset 4 --output artifacts/softmax/reproduction/fp16-2
```

the current checker expects 462 provider cases. the current benchmark uses PyTorch, eager, and Triton with 25 warmups and 100 calls per batch; repeats must be a positive multiple of six. all six provider permutations are used, and offsets rotate their schedule. the default is 12 repeats, not the historical seven. input generation, transfers, final output allocation, compilation, and correctness are outside timing. eager GPU casts/operators and their intermediate traffic are timed; Python and allocator CPU costs appear only through stream gaps. see the [protocol](kernels/triton/softmax/README.md).

eager FP16 inputs promote to FP32 before max/subtract/exp/sum/divide and round only the final output. the old two-provider experiment is reproduced from its recorded base commit, not by running the current three-provider defaults. the reports retain losses, overlapping ranges, and reversals; a Triton win is not required.

assistant implementation is the default for the planned milestones below. flag a rare, narrow user-written kernel piece only when it teaches a concrete mental model. the proposed next lab is device-resident copy/vector-add bandwidth calibration plus a separate small softmax timeline; it remains unimplemented and unmeasured.

## NEXT — gpu fundamentals, paused

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
