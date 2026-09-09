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

## extension: 4, 6, and 8 blocks

pre-run question: does validation loss keep improving as depth increases from 2 to 4, 6, and 8, or does the benefit flatten or reverse? all previous data, optimizer, evaluation, seeds, and 10k-update settings remain fixed. the agent expects possible diminishing returns; this is a hypothesis, not the user's conclusion.

only the new depths are trained. each depth resets to the same post-baseline initialization RNG state, then copies the same embeddings, first block, final norm, and output head. overlapping additional block prefixes therefore start identically, including the prior two-block treatment. depth changes parameters and compute; this remains an equal-update comparison. prior results are retained and will be loaded alongside the new results.

```bash
uv run python -m experiments.compare_bigram_transformer --steps 10000 --stack-depths 4 6 8 --output artifacts/depth-4-6-8-10k
```

save the same milestone samples, full checkpoints, and source hashes as before. inspect each seed and train/validation curves before interpreting aggregate rankings. timing comparisons across runs are approximate.

## extension results

all nine new runs completed; prior and new controls and model/data source hashes match. means over the three paired seeds:

| treatment | parameters | train loss | validation loss | validation range | mean update seconds |
| --- | ---: | ---: | ---: | --- | ---: |
| bigram | 33,280 | 3.7416 | 3.7752 | 3.7666–3.7805 | 4.53 |
| 1 block | 46,176 | 3.4156 | 3.5293 | 3.5108–3.5406 | 14.33 |
| 2 blocks | 58,752 | 3.3196 | 3.4689 | 3.4582–3.4807 | 22.83 |
| 4 blocks | 83,904 | 3.2163 | 3.3895 | 3.3846–3.3948 | 39.79 |
| 6 blocks | 109,056 | 3.1791 | 3.3645 | 3.3629–3.3673 | 56.28 |
| 8 blocks | 134,208 | 3.1472 | 3.3371 | 3.3344–3.3397 | 73.27 |

each seed has the same final validation ranking across depths 1, 2, 4, 6, and 8: deeper is better at this update budget. the 2→4 improvement is 0.0793 nats/token; 4→6 is 0.0251; 6→8 is 0.0273. gains beyond four blocks are smaller, but not strictly diminishing at every increment. parameter count and compute increase together with depth, so this does not isolate depth or establish the best model at equal compute.

verification: 9 complete finite histories with 101 evaluations each; 36 full milestone checkpoints; source hashes match. the new depth interface reproduces the prior two-block losses at updates 0 and 100 exactly. overlapping block prefixes initialize identically. eight tests, lint, and formatting pass. the new notebook results cell is executed; earlier outputs remain intact.
