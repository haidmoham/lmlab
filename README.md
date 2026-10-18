# lmlab

an ai-assisted laboratory for transformer internals, gpu performance engineering,
inference optimization, and serving systems. develop the judgment of a strong
inference engineer through small, reproducible experiments. this is a learning
repository, not an inference framework or production product.

start with [TODO.md](TODO.md) for completed experiments and paused next steps.
the RTX 3080 softmax comparison now includes stable eager tensor operations,
optimized PyTorch, and fused Triton.

## setup and first experiment

```bash
uv sync --locked --python 3.12
uv run --locked python -m kernels.triton.softmax.check
uv run --locked python -m kernels.triton.softmax.benchmark --output artifacts/softmax/reproduction/fp32-1
```

these gpu commands require linux or wsl2, a working nvidia driver, and cuda-enabled
pytorch. the existing `uv.lock` includes triton as pytorch's linux dependency.
cpu and macos/mps are unsupported for this experiment; the commands fail explicitly.
see the [experiment protocol](kernels/triton/softmax/README.md) for tolerances,
timing scope, and limitations. the [three-way report](reports/experiments/softmax-three-way.md)
records measured medians and ranges, and the [CPU evidence verifier](reports/experiments/softmax-results/README.md)
checks compact committed results without launching GPU work. the curriculum is paused.

for cuda c++ work, see [local gpu setup](LOCAL_SETUP.md), the
[setup notebook](notebooks/cuda_setup.ipynb), and
[05 — cuda c++ basics](notebooks/05_cuda_cpp_basics.ipynb). these provide the
existing smoke checks and unexecuted study workspace. the
[pinned gpu curriculum](resources/README.md) retains its upstream attribution.

existing commands remain: `make notebook`, `make test`, `make lint`, `make format`.
use `uv run --locked ruff format --check .` for a read-only formatting check.

## layout

| area | current contents |
| --- | --- |
| `fundamentals/` | [navigation](fundamentals/README.md) to preserved tokenizer, autograd, and transformer work |
| `notebooks/` | existing shared computational surfaces and saved outputs |
| `src/` | tokenizer, dataset, language-model implementations, and gpu smoke primitives |
| `experiments/` | existing ablation runners, plots, research notes, and handoffs |
| `kernels/triton/softmax/` | eager/PyTorch/Triton comparison, correctness gate, and balanced shape-sweep benchmark |
| `tests/` | existing regressions and softmax correctness checks |
| `reports/experiments/` | measured softmax reports, compact sanitized records, and a CPU evidence verifier |
| `artifacts/` | ignored results/checkpoints generated locally |
| `resources/` | pinned external gpu curriculum and provenance |
| `scripts/` | dataset utilities and local cuda setup verifier |
| `public/` | explicit notebook publication manifest; private drafts stay excluded |

inference profiling, serving, scheduling, further kernels, and optimization agents
gain directories when there is runnable work. existing code stays at its current
paths to preserve imports and historical replay.

## experimental and ai contract

each performance experiment states a hypothesis, reference, candidates, numerical
tolerances, shape sweep, hardware/software/dtype/settings, synchronized timing,
and an explanation separating observations from interpretation. retain regressions
and ambiguous results. a microbenchmark alone does not prove faster serving.

assistant implementation is the default within agreed experiments. flag rare,
small user-written pieces only when they teach a concrete mental model. agents
provide kernels, tests, scaffolding, profiler summaries, and report drafts. the
researcher predicts bottlenecks, validates measurements, and explains gains and
regressions in their own words. generated artifacts do not establish mastery.
the [learning contract](AGENTS.md) preserves that boundary. no orchestration
framework is needed.

## companion project

[tcg-agent](https://github.com/haidmoham/tcg-agent) is a separate private companion
project. cross-reference relevant learning experiments here; keep project-specific
implementation and evaluation in its own repository.
