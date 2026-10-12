# preserved foundations

navigation, not duplicate implementations. existing paths preserve notebook
imports, historical replay, and local outputs.

| subject | implementation and shared surface | evidence and pending work |
| --- | --- | --- |
| tokenization/batching | [`src/tokenizer.py`](../src/tokenizer.py), [`src/dataset.py`](../src/dataset.py), [`01_tokenizer.ipynb`](../notebooks/01_tokenizer.ipynb) | byte-level/bpe exercises, existing download script and ignored data |
| bigram models | [`02_bigram.ipynb`](../notebooks/02_bigram.ipynb), [`src/language_models.py`](../src/language_models.py) | matched comparisons in `experiments/` |
| transformers | [`03_attention.ipynb`](../notebooks/03_attention.ipynb), [`src/language_models.py`](../src/language_models.py) | [attention handoff](../experiments/attention-resume.md), depth/context/position reports and tests |
| autograd | [`04_autograd.ipynb`](../notebooks/04_autograd.ipynb) | numpy tensor graph, backward rules and finite-difference checks; outputs preserved |

the attention handoff distinguishes old pre-norm runs from current post-norm
source. do not relabel measurements or treat historical checkpoints as a matched
post-norm control. pending learning mechanisms remain yours unless delegated.

the existing [cuda setup](../LOCAL_SETUP.md),
[setup notebook](../notebooks/cuda_setup.ipynb), and
[cuda c++ study notebook](../notebooks/05_cuda_cpp_basics.ipynb) are preserved,
along with smoke primitives in `src/gpu_primitives/` and the
[credited curriculum](../resources/README.md). no dedicated interpretability or
inference serving implementation is introduced. the untracked local cuda
vector-add exercise in the original checkout remains untouched and uncommitted.
use `kernels/cuda/` when it is deliberately integrated. no existing code was
relocated; only `todo.md` was renamed to `TODO.md` and expanded.
