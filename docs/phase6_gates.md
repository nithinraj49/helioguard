# Phase 6 gates (pre-registered)

Written before any SFT trace was generated and before any training run. Edit this file after training
only to add results under "Outcome", never to change a gate; a changed gate needs a new dated section
that says why.

Registered: 2026-10-26, Phase 6 session 1 (SFT warm start). GVPO is not part of this session.

## What is being tested
A small Qwen model, fine-tuned with LoRA on traces of the tool-informed expert (TRAIN tasks only),
evaluated with the Phase 5 harness (strict parser, env-enforced caps) on the 50 held-out eval tasks
(`tasks/eval.jsonl`, 28 storms, none shared with train). The base 0.5B model scored 0% valid submissions
and a mean final reward of -0.500.

Reference points on the same 50 eval tasks (Phase 5, `results/phase5/`):

| Reference | Mean final reward |
|---|---|
| Tool-informed expert (the teacher) | +0.948 |
| Same plan, no tool calls ("no-tool plan") | +0.743 |
| Rule-based | +0.727 |
| Failed episode | -0.500 |

## Evaluation protocol (fixed)
- Model: the SFT adapter, strict parser unchanged, caps unchanged (10 tool calls, 20 turns).
- Primary condition: greedy decoding (temperature 0), one run per task (greedy has no seed variance).
- Robustness condition: temperature 0.7, 3 seeds (0, 1, 2). Reported, but the gates are decided on greedy.
- Confidence intervals: 95 percent percentile bootstrap over the 50 tasks (10,000 resamples, seed 0),
  resampling tasks and averaging the seeds within a task.
- An episode with no valid submission counts as a final reward of -0.5.
- "Optimal" means final reward >= 0.999 (the exact CP-SAT optimum scores 1.0).
- Metrics: valid-submission rate (primary), mean final reward, percent optimal, mean tool calls per
  episode, and the per-disruption breakdown (a task with several disruptions counts in each one).

## Gates
**Gate A (format).** Valid-submission rate >= 90 percent on the eval tasks (greedy).
Pass: the point estimate is at least 0.90. The CI is reported but not part of the rule.

**Gate B (value).** Mean final reward (greedy) is above the no-tool plan (+0.743) with the 95 percent CI
excluding it: the lower bound of the CI is greater than +0.743.

**Gate C (honesty).** If the mean final reward is within the CI of the rule-based plan (+0.727) or below
it, report that the SFT model only matches (or fails to beat) the rule-based plan, whatever A and B say.
Gate C is a reporting rule, not a pass/fail threshold: it always produces a sentence in the report.

## What happens after
- Pass or fail, the numbers go in the README and in the review summary, including negative results.
- Failing A means the format was not learned: fix the data or training (not the parser).
- A and B passing is the condition for starting GVPO from this adapter. Failing B with A passing means
  the model writes valid submissions but does not use the tools well; that is reported as such, and the
  next step is discussed before any RL.
- The verifier and the strict parser are not changed to make a gate pass (a test pins the parser source).

## Trace-selection rule (also fixed in advance)
Expert traces are kept when the final submission is valid and the final reward is >= 0.90. The reward
distribution and the optimal / non-optimal split of the kept traces are reported in the data card.
This selects easier tasks for training (hard tasks the expert gets wrong are dropped); it is a stated
limitation.

## Outcome
Not run yet. Session 1 prepared traces (800 kept of 1,000), the trainer, the notebook and the evaluation hook;
the 1.5B SFT run and the evaluation against these gates are still to do.

---

## Amendment 1 (dated 2026-10-26, before any training; commit: the commit that adds this section, see `git log`)
Clarifications made before the first training run. No threshold changed.

- **Observation view.** All models are evaluated under the compact tool-observation view (introduced in
  commit 53cfe1d: floats rounded to 3 decimals, hourly altitude series of `simulate_decay` replaced by
  altitudes every 12 h). The scripted baselines and the 0.5B and 1.5B base models were re-run under it;
  results are in `results/phase6/baselines_compact/`. Scripted baselines score the same as under the raw
  view (they use only final altitude and re-entry hour). The base models were run greedy, one attempt,
  50 tasks (no temperature 0.7 run). Earlier base-model figures (raw view) are not comparable.
- **Gate C.** "Only matches" means the model's 95 percent CI contains +0.727 or its point estimate is
  below +0.727.
- **Secondary analysis (not a gate).** Paired bootstrap of the per-task difference in final reward versus
  the no-tool plan (compact view), same 10,000 resamples and seed. If the paired and unpaired Gate B
  verdicts disagree, both are reported.
- **Checkpoint.** The final checkpoint of the fixed-epoch config (`configs/sft_1p5b.json`, 2 epochs) is
  evaluated. No checkpoint is selected using eval results.
- **Registered run.** Qwen2.5-1.5B-Instruct, trainer Unsloth (fp16, Colab T4) if it installs, otherwise
  transformers + PEFT; the trainer actually used is recorded in `train_summary.json` and reported. One
  training seed (seed 0), a stated limitation: no variance across training seeds.
- **"Valid submission".** The episode ended with a `submit_decision` that (1) passed the strict parser and
  (2) was accepted by the environment (every asset in the fleet given a known action, a well-formed launch
  decision when the briefing has one), so the environment returned a final reward. Rule violations
  (for example two raises in one ground window) are still valid submissions and are penalised in the final
  reward. Malformed replies, unplayable submissions and episodes that time out are not valid.
- **Descriptive only.** Per-disruption rows and the quiet-task action rate (on tasks with no disruption
  listed: how often each action is chosen per asset, next to the stored optimum's mix) are descriptive;
  no gate depends on them.
