# lm lab

A personal laboratory for understanding, building, and researching language models.

I work from constituent operations to complete models, then change them to investigate what matters. Implementation, mathematical reasoning, and ablation are complementary ways to understand the same system.

AI is an active collaborator in that work: writing code, developing explanations, challenging designs, and helping investigate results. The standard is whether I can explain the machinery, change it deliberately, and defend conclusions with evidence. I own the research questions and the judgments that follow.

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
