# attention comparison resume

## current state

- `notebooks/03_attention.ipynb` contains the completed bigram versus one-block comparison and local executed outputs. the next stack exercise now follows those results.
- `TransformerStack` is implemented in the notebook and `src/language_models.py`: configurable depth, independent blocks, final embeddings plus attention maps in layer order. the user wrote the constructor and sequential loop, then delegated completion. the matched bigram and depths 1, 2, 4, 6, and 8 comparison has now completed: three seeds, 10,000 updates each. see `experiments/depth-comparison.md` and the final notebook section.
- the completed experiment compares one versus two blocks at fixed width, context, data, optimizer, paired batches, and update budget. validation loss is the primary outcome; record parameter counts and training time. this tests added depth with added parameters, not depth independently of capacity.

## collaboration boundary

the updated `AGENTS.md` and poneglyph execution-modes guidance distinguish learning work from routine completion per subproblem. preserve implementation that develops the user's understanding; automate agreed experiment plumbing. do not make manual stack implementation an automatic prerequisite if composition is already familiar or delegated.

next user/agent decision: interpret the completed comparison and choose whether a follow-up would distinguish a useful explanation. the agent can then prepare matched runs, checkpointing, evaluation, and notebook figures; the user retains the central interpretation. the existing one-block language-model wrapper and its checkpoints are unchanged. do not infer demonstrated understanding from existing code or outputs.

poneglyph guidance was checked at `c085fb4`; lmlab's updated contract arrived in `99edb63`. project progress belongs here; no new global rule was needed.
