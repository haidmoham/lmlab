# attention notebook handoff

## completed

`notebooks/03_attention.ipynb` is the shared working surface, with local executed outputs preserved. source implementations live in `src/language_models.py` and the scientific runners in `experiments/`.

implemented: causal scaled dot-product multi-head attention, ReLU feed-forward networks, pre-norm residual blocks, independent configurable stacks, final normalization, learned and sinusoidal positions, and a zero-position ablation. the current model is a causal language model, not the paper's complete encoder-decoder architecture.

completed experiments (all three seeds 42/43/44, 10,000 updates):
- bigram and depths 1/2/4/6/8: `depth-comparison.md`.
- contexts 8/32/128/256 at four blocks, including supported frozen-model cross-evaluation: `context-comparison.md`.
- bigram → six blocks without encoding → six blocks with sinusoidal encoding: `position-encoding-comparison.md`. mean validation losses are 3.8353 / 3.6731 / 3.5363 nats per BPE token; context 32, width 32, matched batches. the user interpreted this as positional encoding helping in this experiment. 36 full checkpoints are saved locally under `artifacts/position-encoding-six-blocks-10k/`.

protocols differ across experiments; do not merge their losses into a single comparison. historical source hashes describe the code at execution; later plotting-only edits do not rewrite provenance.

all nine existing notebook figures now use the dark palette in `experiments/notebook_theme.py`. figures were regenerated from saved measurements/samples; metrics and non-image notebook outputs were preserved. the setup cell applies the palette for future scratch figures. use normal editable code for implementation; tables are welcome, but do not render bright source-code panels. figures follow explicit user requests.

## agreed next scope

the user selected **encoder, cross-attention, warmup, and dropout**, and explicitly deferred the rest of the paper. this is the next implementation/learning pass, not completed work.

- encoder: unmasked source self-attention, with source padding masks when batching variable lengths.
- cross-attention: decoder queries attend to encoder keys/values. choose a meaningful source/target task together; source and target lengths can differ. the current attention always creates a triangular self-attention mask, so accepting three tensors does not make it a correct cross-attention implementation.
- warmup: a learning-rate schedule; it can be investigated in the existing language model independently of the encoder-decoder task.
- dropout: add the agreed sites and distinguish training from evaluation behavior; it can also be tested with the existing language model. establish rates and controls before a substantial experiment.

reference: [attention is all you need](https://arxiv.org/pdf/1706.03762), sections 3.1/3.2.3 (encoder and cross-attention), 5.3 (warmup and inverse-square-root decay), and 5.4 (dropout). local PDF: `artifacts/papers/attention-is-all-you-need.pdf`.

remaining fidelity differences are intentionally deferred: paper post-norm versus current pre-norm, embedding scaling and weight sharing, label smoothing, original optimizer details beyond the selected schedule, translation-scale data/training, beam search and checkpoint averaging. do not describe the current work as a full paper reproduction. section 4 and table 3 still contain useful reasoning and ablations; the user is choosing a narrower scope, not asserting those sections are only history.

## collaboration

preserve the user's authorship of unfamiliar mechanisms unless delegated; own agreed experiment plumbing. explain the computation and propose discriminating controls without silently supplying the user's research argument. preserve the user's context/loss reflection in the notebook. the next four mechanisms have been selected, but task choice and unagreed experiment budgets are still decisions to make together.
