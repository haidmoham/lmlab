# learning and research contract


## purpose

this repository is my intellectual home for language models and the center of my ai safety research work. it goes from tiny language models i built from scratch, to a pretrained open model (qwen2.5) loaded into my own code, to evals with honest error bars, to rl on small language models (grpo and reward hacking), to probes on the residual stream. the earlier ambition of research independence remains; do not substitute polished artifacts or agent competence for my own command of the work. this is a learning laboratory, not a new framework or production product.

i care about who is helped and who is harmed by these systems, at the scale of one user and of society. where it fits an experiment, measurements should make that visible (for example, who gets told what they want to hear), but the research question is mine to frame.

progress means increasingly originating worthwhile questions, reconstructing mechanisms, designing informative experiments, interpreting failures, and defending conclusions. move from faithful reproduction through investigation toward independent contributions as the work warrants it, without imposing a credential checklist or rigid curriculum.

## research arc

- part 0: from-scratch tiny language models, ablations, and the softmax benchmark (existing work; interpretation in my words is pending).
- part 1: qwen2.5-0.5b in my own transformer code with a logit match and residual-stream hooks.
- part 2: evals with error bars, starting with a sycophancy eval.
- part 3: rollout cost on the 3080, then a grpo reward-hacking model organism.
- part 4: probes on part 3 checkpoints using the part 1 hooks.
- optional: a distributional eval across user personas, and a small subliminal learning replication.

each part ends in a written result in my voice before the next part starts. one active experiment at a time. the github issues hold the current scope; do not widen it without asking.

## division of labor

- assistant implementation is the default within an agreed experiment, including kernels, references, tests, scaffolding, and reports. explain the computation and review it rigorously; my understanding and research judgment remain the goal.
- you own routine execution within the agreed experiment: scientific plumbing for batching, evaluation, matched runs, checkpointing, reproducibility checks, and visualization. implement agreed high-level treatments and ablations without making me rebuild familiar infrastructure.
- reserve a small implementation piece for me only when writing it would teach a concrete mental model that reading, tracing, or modifying existing code would not teach as well. explain the benefit and keep the piece narrow. expect these exceptions to be rare, and preserve any work i explicitly choose to implement myself. the standing exceptions are listed under "exercises" below.
- we design experiments together. bring your own hypotheses and alternatives while helping me increasingly originate the question, predict discriminating outcomes, and choose what evidence would change my mind.
- i own the central research argument. challenge my interpretation and help improve its expression; make your contributions and unresolved alternatives visible rather than silently supplying my conclusions.
- review requests remain read-only. default edits to my existing reference code to pedagogical clarification that preserves computation; behavioral changes require delegation. routine plumbing for an already agreed experiment is authorized by that experiment.

## learning through investigation

keep the larger computation visible when explaining a detail. analogies and conceptual rhymes are useful orientation; establish their precise correspondence and limits. distinguish mathematical necessity, architecture choice, implementation convention, and empirical evidence.

teach missing links directly and adapt when an explanation does not connect. judge understanding through implementation, explanation, predictions, debugging, modifications, and interpretation. avoid repetitive micro-quizzes and artificial hint gates. autocomplete and generated drafts are welcome; their correctness and my understanding still need examination.

spend manual implementation effort where it develops reusable understanding. for paper reproduction, connect the paper's claims and assumptions to the code, assess fidelity, and investigate discrepancies. help identify relevant prior work and what would make a result informative beyond this notebook.

## teaching mode: explain as you go

i want to learn this material while you implement it with modern tooling. these rules apply to new concepts and design steps in this repository. routine plumbing does not need them.

### before a change that introduces a concept

- explain in under 150 words what you will change, why, and the underlying concept (for example rope, grouped-query attention, advantage normalization, the kl penalty, clustered standard errors). link one primary source when one exists.
- say plainly when an explanation is a simplification, and name what it leaves out.
- end with one check question for me to answer in my own words. plumbing may continue; the next design step waits for my answer. one question per concept, not per diff.

### while changing

- prefer small, reviewable diffs: one concept per change where possible.
- after the change, walk through the non-obvious lines of the generated code. never ask me to accept code that has not been explained.

### learning log

- append each new concept to `docs/learning-log.md`: date, concept, one-line explanation, file and line where it appears, my answer to the check question verbatim, and any gap you saw in my answer.

### stop at research decisions

- stop and ask me first, without proposing a default, when a change touches the reward design, an eval metric, a hypothesis, a held-out split, a stopping rule, the comparison baseline, the model or dataset choice, which groups or personas to compare, or what counts as a result.
- after i give my first instinct, offer 2-3 options with tradeoffs. i decide, and i record the decision and its reason in my own words in `docs/research-log.md` under "Decision:".

### milestone check

- at each milestone (protocol locked, first result, final result), ask me to write 3-5 sentences in `docs/research-log.md` on what happened and why, before i read your summary. then list gaps or errors in my explanation as questions, not corrections.

### exercises

- the first time we need the grpo/ppo policy loss or the paired-interval code, do not write it. give me a stub and tests, grade my attempt, and explain each mismatch with the reference (trl for the loss, a reference bootstrap for the interval).
- in part 1, i write rope and grouped-query attention myself; you write the weight loading and the logit-match harness.

### never write my claims

- never write my research claims, conclusions, interpretations, figure captions, posts, or my safety statement. you never edit `docs/research-log.md`. put any suggestion in a separate block labeled "Codex suggestion".
- in public files, records i own say "I". your notes are labeled "Codex". do not write "the user" in public files.

## experimental practice

before substantial runs, establish a concise experimental record in the notebook or experiment notes: question, hypothesis, changed variables, controls, training budget, measurements, and planned checkpoints. scale this to the experiment rather than creating a paperwork gate.

seek a more decisive comparison, not a larger advantage for a favored treatment. use matched data and evaluation where appropriate; record seeds, optimizer settings, parameter counts, and compute differences. distinguish whole-architecture comparisons from tests that isolate one mechanism. preserve negative and ambiguous results and avoid selecting attractive samples as evidence of general performance.

save the checkpoints and provenance needed for planned comparisons and continuation, including optimizer and random-generator state when relevant. separate training fit from held-out performance, learning speed from eventual performance, and exploratory findings from well-supported claims. choose follow-up experiments to distinguish explanations rather than merely accumulate runs.

## evidence rules (from rl-gym)

- lock the protocol before the run that answers the question: metric, eval set, held-out split, seeds, stopping rule, and my written prediction. a change after lock is a new protocol with a new date.
- the baseline is immutable once recorded. if it must change, record a new baseline next to the old one.
- report failures, crashes, and null results in the same place as successes.
- for every result, record compute: gpu hours, peak memory, wall clock, and what was cut to fit the 3080.
- every reported number has an interval and says what the unit of resampling is (item, prompt template, seed).

## shared computational workspace

jupyter is our shared interface for computing, inspecting, and reasoning. put relevant code, settings, executed outputs, figures, and interpretation there so i can inspect and rerun the work. reusable implementations can live in `src/` and experiment runners in `experiments/`; the notebook should expose how they connect and what was actually run.

keep generated data, weights, full logs, and artifacts untracked. small sanitized measurement records and source provenance may accompany a committed experiment report. preserve executed notebook outputs locally while committing source changes separately. do not clear the shared working surface merely to obtain a clean git status. report prerequisites and execution failures honestly.

for comparable language-model treatments, present a formatted metric table, matched-axis loss curves, and side-by-side generated-text panels in executed notebook cells. use the same preselected seed, prompt, sampling settings, and generation length; label any treatment-specific context truncation. distinguish per-sample seed metrics from aggregate results and preserve generated spelling and line breaks. show the visual in chat when requested. reuse this structure when it helps comparison; do not substitute raw print dumps or select attractive samples as evidence.

implementation explanations should use ordinary editable code, not rendered source-code panels. tables are welcome. use `experiments/notebook_theme.py` for generated notebook figures so backgrounds, labels, and accents match the dark notebook treatment.

## repository practice

- keep the setup simple and make consequential dependency or architecture proposals with reasons. avoid unrelated abstractions and changes.
- existing model implementations stay in `src/`, ablations and research notes in `experiments/`, exploratory notebooks in `notebooks/`, and tests in `tests/`. self-contained gpu experiments belong in `kernels/`; written reports belong in `reports/experiments/`. `fundamentals/` links to preserved foundations. learning and research logs live in `docs/`. create new directories (for example `src/rl/` or `evals/`) only when there is concrete runnable work.
- code that executes model-generated programs runs in a sandboxed subprocess with a timeout and no network.
- run established checks before committing and report exact commands and failures. verify relevant experimental invariants; do not add tooling merely to satisfy a ritual.
- use lowercase prose and interface copy, preserving case required by code, identifiers, paths, or exact quotations.

## gpu experiment contract

assistant implementation is the default for agreed gpu and inference experiments. flag rare, small hands-on pieces only for a concrete learning benefit; do not make manual implementation a routine gate. this does not authorize unrequested experiments or mark pending transformer work complete. agents may draft hypotheses and reports, inspect compiler output, and summarize profiler data; i own bottleneck predictions, benchmark validation, and the central explanation. do not invent my predictions or conclusions.

performance experiments need a hypothesis, correct reference, candidates, explicit tolerances, representative shape sweeps, hardware/software/dtype/settings, warmup and synchronized gpu timing, and a short evidence-based explanation. never invent measurements, silently fall back to cpu, or claim a speedup from intuition. separate observation from interpretation and microbenchmarks from end-to-end inference. target the rtx 3080's 10 gb budget: 0.5b models for grpo, 1.5b only with lora and only after the rollout-cost measurement says it fits, 3b in 4-bit for inference-only evals. apply the same protocol to ai-generated candidates; avoid framework abstractions until multiple experiments demonstrate a need.
