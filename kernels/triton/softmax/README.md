# triton softmax versus pytorch

question: how does a simple one-program-per-row kernel compare with optimized
`torch.softmax` over row counts, widths, and dtypes on an rtx 3080?

agent-proposed hypothesis: a register-resident reduction with one input read and
one output write may do well at moderate widths and enough rows. few rows, padded
widths, register pressure, or pytorch's tuned dispatch may favor pytorch. this is
not a measured finding. record your own predictions in the
[worksheet](../../../reports/experiments/softmax.md) before timing.

## run from the repository root

linux or wsl2, python 3.12, an nvidia driver, and cuda-enabled pytorch are required.
the existing lockfile supplies pytorch and its compatible linux triton dependency.
no cpu/mps fallback, native-windows path, or gpu ci job is introduced.

```bash
uv sync --locked --python 3.12
nvidia-smi
uv run --locked python -m kernels.triton.softmax.check
uv run --locked pytest -q tests/test_softmax.py -rs
uv run --locked python -m kernels.triton.softmax.benchmark --output artifacts/softmax/run-1
uv run --locked python -m kernels.triton.softmax.benchmark --dtype float16 --output artifacts/softmax/fp16-1
```

select the actual gpu with `--device INDEX`. use a new output directory for every
run to retain earlier evidence. gpu pytest cases skip on a cpu/mac; **the strict
check command must pass on the workstation**. [TODO.md](../../../TODO.md) includes
repeat runs and the definition of done.

## implementation and correctness

- `common.py`: reference, tolerances, hardware gate, and correctness checks.
- `kernel.py`: triton candidate and input/output validation.
- `check.py`: strict 154-case workstation gate.
- `benchmark.py`: shape sweep, raw cuda-event samples, and environment/source record.

the reference is optimized pytorch, not an unfused Python formula. both timed paths
write to independent preallocated outputs. the candidate rounds width to the next
power of two, masks loads with negative infinity, subtracts the maximum, computes
exponentials and the sum in fp32, divides, then masks stores. four warps are the
default; `--num-warps 8` or `16` is a later controlled treatment, not an assumed win.

supported inputs: finite contiguous nonempty 2d cuda tensors, float32/float16,
width <= 8192. output storage must not overlap input storage. this forward-only
kernel rejects gradient-requiring inputs. noncontiguous layouts, infinities/nans,
wider rows, backward, and bfloat16 are outside this experiment. finite-value
validation is a correctness step outside timing, not a scan on every launch.

element tolerances: fp32 `rtol=1e-4, atol=1e-6`; fp16 `rtol=2e-3, atol=2e-5`.
compare with pytorch and a float64 oracle; require nonnegative finite probabilities
and fp32-accumulated row sums of one (`rtol=0`, `atol=1e-5` for fp32 or `1e-3` for
fp16). seeds 0/42 cover width one, irregular/power-of-two widths and varying row
counts. adversarial cases cover tied maxima, large offsets, wide dynamic range,
and a maximum in the last real lane. tests also cover preallocated writes.

## measurement protocol

rows: 1/32/256/4096. widths: 1/31/128/257/781/1024/2049/4096/8192. the largest fp32
input is 128 mib; input plus two outputs use 384 mib. allow extra correctness and
runtime memory within the 10 gb budget.

compile and check both actual timed paths, warm each for 25 calls, synchronize,
then measure seven interleaved repeats of 100 calls per provider with cuda events
and end-event synchronization. alternate provider order. each raw sample is a
batch mean in milliseconds per call; median/min/max summarize those batch means,
not request latency percentiles. generation, allocation, transfers, JIT compilation,
and correctness are excluded. stream timing can include host launch gaps and is
not isolated kernel duration. disable `CUDA_LAUNCH_BLOCKING`.

inputs/outputs remain resident and reused without explicit cache flushing.
effective bandwidth is `2 * rows * width * element_size / latency`, in decimal
gb/s: minimum logical traffic, not measured dram traffic or bus utilization.
small shapes may be cache-resident. speedup is median pytorch latency divided by
median triton latency; below one is a loss. differences within the observed
spread need more evidence.

`summary.csv` records latency/spread/bandwidth/speedup by shape. `results.json`
retains raw repeats, settings, gpu/vram/capability, software versions, git revision
and status, source/lockfile hashes, selected runtime flags, and an optional
`nvidia-smi` snapshot of driver/clocks/temperature/load. missing telemetry is
labeled. record other workloads or changing clocks in the report; repeat the
run before attributing small differences to the kernel.

no performance results have been measured in this macos environment. profiler
evidence and your explanation remain subsequent work. no serving speed claim
follows from this microbenchmark. implementation reference:
[triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
its persistent scheduling and published results are not copied into this lab.
timing reference: [pytorch cuda events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).

## scaffold verification — 2026-10-11

on macos arm64 with python 3.12.13 and pytorch 2.13.0, `uv sync --locked --python
3.12`, `make test` (23 passed, 43 gpu tests skipped), `make lint`, and formatting
of the six new python files passed. syntax and local documentation links checked;
all four notebooks remain byte-identical to the committed baseline. the strict
check and benchmark both fail explicitly without cuda and create no result files.
the full repository formatting check has a pre-existing failure in
`notebooks/04_autograd.ipynb`, left unchanged. triton compilation, gpu correctness,
and latency/bandwidth measurements remain unverified until workstation execution.
