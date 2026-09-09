# attention comparison resume

## current state

- `notebooks/03_attention.ipynb` contains the completed bigram versus one-block comparison and local executed outputs. the next stack exercise now follows those results.
- `TransformerStack` remains an unimplemented scaffold. no depth comparison was run in this session.
- the next proposed experiment is one versus two blocks at fixed width, context, data, optimizer, paired batches, and update budget. validation loss is the primary outcome; record parameter counts and training time. this tests added depth with added parameters, not depth independently of capacity.

## collaboration boundary

the updated `AGENTS.md` and poneglyph execution-modes guidance distinguish learning work from routine completion per subproblem. preserve implementation that develops the user's understanding; automate agreed experiment plumbing. do not make manual stack implementation an automatic prerequisite if composition is already familiar or delegated.

next user/agent decision: establish whether stack composition is still the learning target and agree on the comparison question and budget. the agent can then prepare matched runs, checkpointing, evaluation, and notebook figures; the user retains the central interpretation. do not infer demonstrated understanding from existing code or outputs.

poneglyph guidance was checked at `c085fb4`; lmlab's updated contract arrived in `99edb63`. project progress belongs here; no new global rule was needed.
