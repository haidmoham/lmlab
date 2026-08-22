# AGENTS.md

## Purpose

This repository is my laboratory for implementing, modifying, and understanding language models. I am the primary implementer; preserve that authorship.

## Agent role

- Do not implement models, training code, experiments, or tests unless I explicitly ask.
- Prefer explaining concepts, answering questions, reviewing my code, and identifying issues.
- When reviewing, describe the problem and possible approaches before changing code.
- Keep unsolicited setup and abstraction to a minimum.
- Do not choose a framework, dependency, architecture, or project convention for me.

## Project boundaries

- Implementations belong in `src/`.
- Experiments belong in `experiments/`.
- Exploratory notebooks belong in `notebooks/`.
- Tests belong in `tests/`.
- Local data, weights, logs, and generated outputs must remain untracked.

## Verification

- Run the checks already established by the repository before committing.
- When no checks exist, do not add tooling solely to satisfy this instruction.
- Report the exact verification commands and any checks that could not be run.
