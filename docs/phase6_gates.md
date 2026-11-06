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
Measured 2026-11, one registered run (Qwen2.5-1.5B-Instruct + LoRA, final checkpoint, 50 eval tasks). No gate, threshold,
parser, verifier or metric was changed; this section only records results. Source files:
`results/phase6/train_summary.json` and `results/phase6/sft_1p5b_eval_summary.json`.

### Gate verdicts (greedy, the condition the gates are decided on)
| Gate | Rule | Measured | Verdict |
|---|---|---|---|
| A, format | valid-submission rate >= 0.90 | 1.0 (50 of 50 episodes) | **Passed** |
| B, value | lower bound of the 95% CI of mean final reward must exceed +0.743 | mean +0.825, 95% CI +0.717 to +0.908; lower bound +0.717 | **Failed** |
| C, honesty | compare with the rule-based plan (+0.727) | mean +0.825, CI +0.717 to +0.908 contains +0.727 | The SFT model **only matches** the rule-based plan |

Gate B fails because the lower bound (+0.717) is below the no-tool plan's +0.743, even though the point estimate is above it.
Gate C is not a pass: the CI contains the rule-based plan's +0.727, so the data do not separate the two.

### Other greedy measures (secondary)
Percent optimal 24% (95% CI 12% to 36%). Tool calls per episode 9.86 (CI 9.72 to 9.96). No failure episodes.

### Temperature 0.7, seeds 0, 1, 2 (150 episodes, secondary)
Valid-submission rate 1.0. Mean final reward +0.636 (95% CI +0.525 to +0.739). Percent optimal 22% (CI 12.7% to 32%).
Tool calls per episode 9.86.

### Paired analysis against the no-tool plan (secondary, not a gate)
Per-task difference in final reward, model minus the no-tool plan (compact view), 50 tasks, same bootstrap.
| Condition | Mean difference | 95% CI | CI excludes zero above? |
|---|---|---|---|
| Greedy | +0.082 | -0.029 to +0.169 | No |
| Temperature 0.7 (3 seeds) | -0.107 | -0.220 to 0.0 | No |

The paired and unpaired Gate B verdicts agree (both: not shown to beat the no-tool plan).

### Per-disruption breakdown (descriptive, greedy; a task can carry several disruption types, so groups overlap)
Small groups (3 to 5 tasks) are anecdotes, not estimates.
| Disruption | Tasks | Valid rate | Mean final reward | Percent optimal |
|---|---|---|---|---|
| forecast_error | 25 | 1.0 | +0.905 | 28.0% |
| storm_stronger | 17 | 1.0 | +0.839 | 23.5% |
| storm_longer | 20 | 1.0 | +0.785 | 25.0% |
| thruster_failure | 15 | 1.0 | +0.785 | 20.0% |
| launch_slip | 11 | 1.0 | +0.799 | 9.1% |
| none | 5 | 1.0 | +0.858 | 20.0% |
| ground_outage | 3 | 1.0 | +0.205 | 0.0% |

### Quiet-task action mix (descriptive, greedy, not a gate)
On the 5 tasks with no disruption listed, per-asset actions chosen by the model against the stored optimum's mix:
| Action | Model | Stored optimum |
|---|---|---|
| safe_mode_edge_on | 12 | 10 |
| raise_orbit | 5 | 7 |

All 5 tasks received a submission. The model chose safe mode on 2 more assets than the optimum did.

### Run details
- Model Qwen/Qwen2.5-1.5B-Instruct, trainer `peft` (transformers + PEFT), training seed 0, config `configs/sft_1p5b.json`
  (LoRA r 16, alpha 32, lr 2e-4, 2 epochs, 8 examples per step, gradient checkpointing). 800 traces, none skipped for length.
  200 of 200 steps done; loss 0.120 at the first logged step and 0.0031 at the last.
- **Resume.** The run was resumed from step 22 (`resumed_from` 22) after the Google Drive sync stopped partway through the
  first session and the later checkpoints were lost. This is a protocol event recorded for transparency, not a change to
  the registered run: same config, same data order (a pure function of seed and step), same learning-rate schedule.
- **Code between the two parts.** The Hub-mirroring and resume-from-Hub code (robust saving) was added after the first
  part, and a fix for resuming from an empty Hub repo (commit 31fa584) was applied before the second part. None of this
  touches the parser, verifier, gates, metrics or the training maths. `train_summary.json` records `git_commit`
  31fa584afb534551d9d88d6287f7458f7c4ff4c3 with a clean tree; that is the commit of the second part (steps 23 to 200).
  The commit of the first part (steps 1 to 22) is not recorded in the file.
- **Packages.** torch 2.11.0+cu130, transformers 5.22.0, peft 0.21.2, accelerate 1.15.0, torchao 0.18.0 (as reported by
  the user from the Colab run; `train_summary.json` does not store them). The notebook pin for transformers was moved from
  5.20.2 to 5.22.0 to match.
- The evaluation ran without `--greedy-only`: greedy once plus temperature 0.7 with seeds 0, 1, 2, as in Amendment 3.

### What this does and does not show
SFT taught the format completely (Gate A) and the model calls tools about ten times per episode, but its reward is not shown
to be better than the no-tool plan (Gate B failed) and is statistically indistinguishable from the rule-based plan
(Gate C). Only 24% of greedy episodes reach the optimum. These are results from one run, one seed, and 50 tasks; the
confidence intervals are wide. Nothing here shows the model learned to use the tool outputs beyond what a simple plan achieves.

**Next step: discuss before any RL.** This is not a pass, and no RL phase starts from it unprompted.

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

---

## Amendment 2 (dated 2026-10-26, before any training; commit: the commit that adds this section, see `git log`)
No threshold changed. Fixes an information loss in the compact tool-observation view of Amendment 1.

- **Change.** The compact `simulate_decay` result now includes `min_alt_km` and `min_alt_hour`: the minimum
  of the full raw hourly altitude series and the hour it occurs (rounded to 3 decimals like every float),
  for all actions. Everything else in the view is as in Amendment 1.
- **Why.** Raise trajectories dip and then recover, and the 12-hourly samples can miss the dip. A scan of
  459 `simulate_decay` results on the 50 eval tasks (all assets, three actions, 48 h) found 154 non-monotone
  raw trajectories and 18 raise trajectories whose true minimum was not visible in the compact view; the
  largest gap was 1.52 km (for example `eval_storm_20220203_0`, asset b2, raise: true minimum 209.529 km at
  hour 10, the samples showed 210.0 km). Safe mode and hold decay monotonically and lost nothing. The
  minimum, and the margin to re-entry (minimum altitude minus 150 km), are what an agent needs to judge a
  raise.
- **Unchanged.** The environment, rewards, verifier and strict parser (the parser hash test passes); the
  decision fields `final_alt_km`, `reentered`, `reentry_hour`, `can_gain_altitude`, `thrust_slows_decay`
  and `min_alt_km` of `min_safe_altitude` are identical after rounding. Tests cover every `simulate_decay`
  result on all eval tasks and (slow test) all train tasks.
- **Scripted baselines unchanged.** They still score +0.727 (rule-based), +0.743 (no-tool plan), +0.948
  (tool-informed), +1.000 (oracle) and -1.353 (always raise) on the 50 eval tasks.
- **Base models.** The 0.5B and 1.5B base-model compact-view runs started under Amendment 1 were stopped
  and discarded; they are re-run under this view (greedy, one attempt, 50 tasks) into
  `results/phase6/baselines_compact/`.
- **SFT traces regenerated.** 800 of 1,000 kept as before (same tasks, same order). Assistant turns and
  step and final rewards are byte-identical to the previous set; only tool-observation text differs.
  Tokens per trace: mean 3,392, max 3,595 (before: 3,214 and 3,415).
- **Trainer field.** The training config has a `trainer` field (`peft` or `unsloth`), default `peft`, and
  there is no silent fallback: the registered trainer is the one that runs. It is set after the T4 smoke test
  and the choice is committed before the real run; this refines the "Registered run" line of Amendment 1,
  and `train_summary.json` records the trainer used.

---

## Amendment 3 (dated 2026-10-28, before the registered run; commit: the commit that adds this section, see `git log`)
No threshold changed. Registers the run and records the T4 smoke test.

- **Registered run.** Qwen2.5-1.5B-Instruct + LoRA, trainer `peft` (transformers + PEFT; Unsloth is not
  used), training seed 0, 2 epochs (200 steps; 800 traces, 8 examples per step, 100 steps per
  epoch), config `configs/sft_1p5b.json`. The **final checkpoint** (step 200) is the one evaluated; no
  checkpoint is selected using eval results. This settles the "Registered run" line of Amendments 1 and 2:
  the trainer is `peft`.
- **T4 smoke test (before registering).** 20 steps on the Colab T4: no NaN losses, fits in memory at the
  longest trace (3,595 tokens), about 54 s per step. Resume check passed: the run was stopped, resumed with
  resumed_from 11, and steps 12 to 20 were identical to the uninterrupted run. At about 54 s per step the
  200-step run is roughly three hours of compute plus checkpointing.
- **Provenance.** `train_summary.json` records the git commit hash and whether the working tree had
  tracked changes, together with the trainer and the training seed.
- **Evaluation protocol.** `python -m helioguard.sft_eval` is run without `--greedy-only`: greedy once and
  temperature 0.7 with seeds 0, 1 and 2, on the 50 eval tasks, as in the protocol above. Gates are decided
  on greedy.
- **Environment pins.** The notebook pins peft, transformers and accelerate and requires torchao>=0.16.0;
  the versions in use are printed in the notebook's check cell and are to be copied into the run notes.

## Note 2026-10-30 (Phase 7 session 1): QUBO penalty rule; no gate or threshold changed
Phase 7 found that `build_qubo`'s original penalty (range + 1) can be too weak when rewards are large and
similar, and added an explicit `penalty_rule`: "sufficient" (new engine default) and "legacy" (the original).
**Phase 6 uses "legacy"**, so every Phase 6 QUBO, `qaoa_schedule` output, trace and evaluation reproduces
exactly: a test (`tests/test_phase6_qubo_unchanged.py`) pins the fleet QUBO of all 1,050 train and eval tasks to
digests built with the original code. On those 1,050 tasks the legacy penalty gave a **feasible exact optimum
on all of them**, and exact and OR-Tools objectives agree on all of them under both rules. For reference, the
two rules give different penalty values on 887 tasks, 6 train tasks sit below the sufficiency bound (still
feasible), and 10 train tasks have a different but equally good exact plan (ties). `data/sft/train_traces.jsonl`
holds no QAOA outputs (no `approx_ratio` or `expected_energy`). No Phase 6 data was modified.
