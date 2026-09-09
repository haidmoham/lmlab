# bigram, one block, and two blocks

## pre-run record

question: at the same data exposure and update budget, does adding a second transformer block improve held-out next-token prediction?

agent-proposed hypothesis: two blocks may reduce validation loss through additional representation refinement. a smaller training loss with unchanged or worse validation loss would instead indicate a generalization limitation at this budget. the user has not yet supplied a prediction or interpretation.

treatments: bigram, the existing one-block model, and a two-block stack. all use width 32, context 8, batch size 32, and the same 512-token BPE corpus and 90/10 stream split. the stack adds one block; the bigram comparison changes positions, normalization, attention, and feed-forward computation together.

controls: three paired seeds (42, 43, 44), identical training batches per seed, fixed shared train/validation evaluation windows (32 batches each; seed 2026), AdamW with learning rate 0.001 and weight decay 0.01. shared components start at equal values but remain independent parameters. the original one-block construction order is preserved so the previous baseline can be reconciled.

budget: 10,000 updates per treatment and seed, from scratch. equal updates mean equal data exposure, not equal compute. measure parameter count and update time separately.

measurements: train and validation cross-entropy every 100 updates, reported in nats per BPE token. validation loss is primary; report each seed, mean, and seed range. generate the same prompt at temperature 1 and sampling seed 123 at 1k, 2.5k, 5k, and 10k. samples are illustrative, not the primary evidence.

save weights and full optimizer/RNG checkpoints at those milestones and at completion. retain original comparison artifacts. new results: `artifacts/all-treatments-10k/`.

```bash
uv run python -m experiments.compare_bigram_transformer --steps 10000 --include-stack --output artifacts/all-treatments-10k
```

## observed results

all nine runs completed. means over seeds 42, 43, and 44:

| treatment | parameters | train loss | validation loss | mean update seconds |
| --- | ---: | ---: | ---: | ---: |
| bigram | 33,280 | 3.7416 | 3.7752 | 4.53 |
| one block | 46,176 | 3.4156 | 3.5293 | 14.33 |
| two blocks | 58,752 | 3.3196 | 3.4689 | 22.83 |

paired validation differences (two minus one): -0.0527, -0.0729, -0.0557 nats/token. the mean improvement is 0.0604 nats/token, with 27.2% more parameters and approximately 59.3% more update time. timings exclude evaluation and checkpointing and are approximate, machine-specific costs.

the two-block model improved validation loss in every measured seed at this budget. this is evidence for this treatment under these settings, not an isolated effect of depth or a universal result. the central interpretation and choice of follow-up remain open for discussion with the user.

verification: all six baseline loss histories match the prior 10k experiment exactly. recorded source/corpus hashes match the executed files. the two-block 1k checkpoint restored identical next updates when model, optimizer, and RNG state were restored independently. six tests, lint, and formatting checks pass. only the new results cell was executed for notebook reporting; existing outputs were preserved.
