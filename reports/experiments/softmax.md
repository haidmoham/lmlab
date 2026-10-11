# softmax experiment worksheet

status: unmeasured. agent-prepared worksheet, not a researcher-authored conclusion.
complete predictions before looking at benchmark output.

## before running — researcher

- question/hypothesis:
- winner predictions for `(1, 128)`, `(4096, 128)`, `(4096, 1024)`, `(4096, 2049)`:
- predicted difference between widths 2048 and 2049, with reason:
- minimum logical bytes for float32 `(4096, 1024)`:
- bandwidth assumption, source/measurement, and latency lower bound:
- likely limiter, confidence, and observation that would change my mind:

## controls and budget

same finite seeded inputs, dtype/device, independent preallocated outputs,
warm-cache state, and forward operation. default budget: two fp32 sweeps and one
fp16 sweep; 36 shapes each; 25 warmups; seven interleaved batches of 100 calls per
provider. compilation/correctness are outside timing. record deviations.
commands: [TODO.md](../../TODO.md).

## correctness and provenance

- strict gate command, pass count, tolerances, failures:
- selected gpu/vram, driver, cuda/pytorch/triton/python versions:
- git revision/status and source hashes from results.json:
- exact artifact paths for every sweep:
- other gpu load and temperature/power/clock observations; unavailable telemetry:

## measured observations

retain losses and ambiguous cases. raw files stay in ignored `artifacts/softmax/`;
this table is the durable summary. leave cells blank until measured. units are
microseconds and decimal gb/s; min/max describe batch means, not request tails.

| shape / dtype | pytorch median [min, max] us | triton median [min, max] us | effective gb/s, pytorch / triton | pytorch/triton speedup | repeat-run agreement |
| --- | --- | --- | --- | --- | --- |
| 1 x 128 / fp32 | | | | | |
| 4096 x 128 / fp32 | | | | | |
| 4096 x 1024 / fp32 | | | | | |
| 4096 x 2049 / fp32 | | | | | |

## interpretation — researcher

- which predictions held/failed? cite measured rows.
- possible mechanism for a gain/regression: distinguish evidence from inference.
- alternative explanation and profiler evidence that would discriminate:
- why effective bandwidth does not establish dram utilization:
- what this cannot establish about serving:
- one next experiment, changed variable, controls, budget, and decision criterion:

agent contributions: initial kernel/tests/harness and proposed hypothesis.
researcher prediction, explanation, profiler evidence, and conclusions: pending.
