# lm lab

A personal laboratory for understanding, building, and researching language models.

I work from constituent operations to complete models, then change them to investigate what matters. Implementation, mathematical reasoning, and ablation are complementary ways to understand the same system.

After the attention foundations, the direction is language-model paper reproduction: implementing published ideas, checking fidelity, and investigating changes through ablation.

the long-horizon aim is research independence strong enough to be competitive for fellowships at labs such as anthropic. i write the reference implementations of mechanisms i am learning; ai contributes scientific plumbing, critique, and agreed treatments and ablations. we use jupyter as a shared computational workspace, with runnable experiments and visible evidence.

progress means increasingly owning the questions, experimental decisions, and arguments—not merely producing working models. the [learning and research contract](AGENTS.md) defines the division of labor and experimental standards.

## setup

From a fresh clone:

```bash
uv sync
```

## common commands

```bash
make setup
make notebook
make test
make lint
make format
```

## repository layout

```text
.
├── configs/       Experiment configurations
├── data/          Local datasets (ignored by Git)
├── experiments/   Experiment code and notes
├── notebooks/     Exploratory notebooks
├── scripts/       Utility scripts
├── src/           Implementations
└── tests/         Tests
```
