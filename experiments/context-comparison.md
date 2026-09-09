# context-length comparison

## pre-run record

question: does more available history improve held-out prediction at four blocks and fixed training-token exposure? the agent hypothesis is that longer context may help, with possible diminishing returns. the user has not supplied a quantitative prediction.

treatments: context lengths 8, 32, 128, 256; depth 4, width 32, two heads, feed-forward width 128. three seeds (42, 43, 44), 10,000 AdamW updates at learning rate 0.001 and weight decay 0.01. each update trains on exactly the same contiguous 256 targets, reshaped into batches of 32, 8, 2, or 1 sequences. each treatment sees 2,560,000 prediction targets per seed.

sequence boundaries and the amount of preceding information change with context. learned position tables also grow. all other initial parameters match, and position-table prefixes match. the training sampling protocol differs from the earlier depth sweep; rerun all four contexts and do not merge its losses with the earlier table.

evaluation: fixed 128 distinct target positions per split (seed 2026), all at least 256 tokens into the split. score only the last-position prediction using exactly each treatment's context of preceding tokens. train and validation targets are separate; use the same targets for every context and seed. evaluate every 500 updates. this is a sampled held-out estimate, not a full-corpus metric.

checkpoints: full model, optimizer, and RNG states at 1k, 2.5k, 5k, 10k. generation: preselected seed 42, same first 256 training tokens as prompt, model truncates to its context, temperature 1, sampling seed 123, 100 new BPE tokens. preserve all samples. show prompt suffix in panels and retain full prompt in results.

measurements: per-seed and mean train/validation loss, seed range, parameters, update time excluding evaluation/checkpointing. equal tokens and updates are not equal compute. the primary comparison is validation loss on matched target positions; qualitative text is secondary.

```bash
uv run python -m experiments.compare_context_lengths --steps 10000 --output artifacts/context-comparison-10k
```

notebook: expose the run command and execute the result table, curves, and four-panel text grid. original depth results stay intact. no new model mechanism is introduced.

## cross-evaluation and presentation boundary

user authorized both comparisons and reserved interpretation for a joint walkthrough. record and present measurements without rankings, conclusions, explanatory analysis, or follow-up recommendations.

at the final checkpoint, evaluate each frozen model at every supported context in {8,32,128,256}. score identical target positions throughout. cropped inputs use learned positions starting at zero, as in the model's ordinary forward pass; this changes available history and position usage. do not treat it as a fully isolated history intervention. save all ten supported train/evaluation combinations per seed. show means and per-seed tables with unsupported cells blank.

include native-context text panels and a second four-panel display of the 256-trained model sampled at each evaluation context. all panels use seed 42, the same prompt and sampling settings. no output selection based on apparent quality.

## execution receipt

completed 12 training runs (four contexts × three seeds), each with 10,000 updates and 2,560,000 prediction targets. saved 48 full milestone checkpoints and 30 supported cross-evaluation rows, plus fixed samples. all histories contain 21 evaluations and finite losses. matching-context cross-evaluation entries equal final native evaluation exactly. source/corpus hashes were verified.

11 tests, lint, and formatting checks pass. the notebook run/display cells executed against the completed artifacts; original outputs were preserved. visual layout was inspected for readable panels. interpretation and the next research decision remain reserved for the user-led walkthrough.
