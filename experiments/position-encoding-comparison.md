# bigram, transformer, and positional encoding

question: what changes in held-out next-token loss when adding a six-block transformer to the bigram baseline, then adding sinusoidal positional encoding to that transformer? these are two planned contrasts across three arms, paired by seed and batches. exploratory assistant hypothesis: each addition may help, but neither improvement is assumed.

treatments: the existing bigram model; six pre-norm transformer blocks without explicit positional encoding (zero vectors); the identical six-block architecture with section 3.5 sinusoidal encoding. width 32, two heads, feed-forward width 128, context 32. the transformers share final normalization and causal masks. the no-encoding arm can still exploit order cues from masking. token embeddings are not rescaled.

the first contrast changes architecture, parameter count, compute, and access to preceding tokens. the second changes only the added position vectors, with equal trainable parameter counts. sinusoidal encoding has no learned parameters. this is not a learned-versus-sinusoidal comparison.

controls: identical token embeddings and vocabulary head across all arms; identical block and final normalization weights across the transformers; independent parameter objects. identical batches and targets for every arm. seeds 42/43/44, 10,000 updates, 256 target tokens per update (eight sequences of 32). AdamW, learning rate 0.001, weight decay 0.01, betas 0.9/0.999, epsilon 1e-8. CPU, two torch threads. equal updates are not equal compute.

measurements: fixed 1,024 sampled targets per split, final-position predictions with 32 preceding tokens provided to every arm; the bigram uses only the last token. evaluate step zero, every 1,000 updates, and 2,500. report train/validation means, validation seed ranges and paired differences, parameter counts, and update time. planned validation contrasts are transformer minus bigram and sinusoidal transformer minus no-encoding transformer; negative means lower loss. tables only. this evaluation sample differs from earlier runs, so compare within this experiment.

save full model, optimizer, RNG states and history at 1k/2.5k/5k/10k. config records source/corpus hashes, split sizes, evaluation targets and software version. results and weights remain local. interpretation is reserved for the joint walkthrough.

```sh
uv run python -m experiments.compare_position_encodings --steps 10000 --output artifacts/position-encoding-six-blocks-10k
```

## execution receipt

completed all nine runs at 10,000 updates, each with 2,560,000 target-token exposures. saved 36 full checkpoints. all 12 evaluation points per run are finite; source/corpus hashes match the executed code. the three seed-42 final checkpoints reload and reproduce recorded validation losses exactly. the notebook result cell executed with the mean and paired-contrast tables; earlier outputs were preserved. 20 tests, lint, and formatting checks passed. no performance interpretation added.
