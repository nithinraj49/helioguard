# Phase 6 diagnosis: why the SFT model misses the optimum

Read-only analysis of the greedy evaluation of the 1.5B SFT model (50 held-out tasks). No training, no GPU, no gate,
metric, parser or verifier changed. Everything below is descriptive: one run, one seed, 50 tasks, so groups are small
and nothing here is a tested hypothesis. Gate verdicts are in `docs/phase6_gates.md` (Outcome); this note explains them.

**Inputs.** `results/phase6/sft_1p5b_eval_summary.json` and the per-turn greedy log `results/phase6/local/sft_1p5b_greedy.jsonl`
(local only, git-ignored, not committed). The "tool-informed expert" is `ToolInformedPolicy` (`helioguard/policies.py`)
replayed on the same 50 tasks under the same compact observations, so its tool calls and plan can be compared one to one.
The analysis scripts were run from the scratchpad and are not committed; the definitions below are what they computed.

**Definitions.**
- *Optimal* is the harness definition: valid submission with final reward >= 0.999 (12 of 50 tasks, the 24% in the Outcome).
- *Stored optimum* is `task["optimal"]["plan"]` (exact solve). *Expert plan* is what the tool-informed policy submits. The expert sees
  only the forecast storm, so it is itself not optimal on every task.
- *Own-simulation plan*: the plan the expert's decision rule gives when fed the values from the model's own `simulate_decay`
  results (best simulated action per asset, one raise per ground window, none in an outage window, launch from `min_safe_altitude`).
- *Explicit contradiction*: the submission says the opposite of a tool statement (raise where `check_feasibility` says thrust does
  not slow decay or the ground window is unavailable; deploy below `min_safe_altitude`).

## 1. Submissions against the expert and the stored optimum

| | Tasks (of 50) |
|---|---|
| Model plan identical to the expert plan (all assets and launch) | 15 |
| Model plan identical to the stored optimum | 10 (plus 2 more tasks reach optimal reward with a different, tied plan) |
| Expert plan identical to the stored optimum | 16 (expert reward >= 0.999 on 18) |
| Model optimal | 12 |
| Model not optimal | 38 |

Reward, same 50 tasks: expert mean +0.948 (95% bootstrap CI +0.926 to +0.969); model mean +0.825 (CI +0.717 to +0.908);
paired model minus expert -0.124 (CI -0.233 to -0.035). The model beats the expert by more than 0.01 on 12 tasks, matches it on 17
and falls short on 21. The expert's CI lower bound (+0.926) is above the Gate B line (+0.743); that is a reference, not a gated result.

**The 38 non-optimal tasks split in three by what the expert does:**
- 10 tasks: model plan equals the expert plan, and the expert is not optimal either. The shortfall is in the information the tools
  give (they show the forecast storm only; stronger or longer storms are invisible), not in the model.
- 15 tasks: the expert is also not optimal, and the model differs from the expert as well.
- 13 tasks: the expert reaches the optimum and the model does not. All 13 differ from the expert. This is the model's own error.

**Launch decisions.** On the 25 tasks with a launch decision, the model's launch choice is identical to the expert's on all 25
(and to the own-simulation plan). It differs from the stored optimum on 24 of 25. 11 non-optimal tasks have every asset action equal
to the optimum and miss only on the launch. So the launch part of the shortfall comes from the expert rule itself (lowest option at or
above `min_safe_altitude` from the forecast storm), which the model copies faithfully.

**Which assets differ (asset actions, 169 decisions over 50 tasks).** On the 38 non-optimal tasks, 11 have no asset difference from the
optimum (launch only, above), 16 differ in one asset, 8 in two and 3 in three. Against the optimum the differences are 27 times
`raise_orbit` -> `safe_mode_edge_on` and 14 times `safe_mode_edge_on` -> `raise_orbit`: the model mostly raises too few assets.
Against the expert, 61 of 169 asset actions differ (in all 50 tasks combined). The per-task list:

| Task | Model | Expert | Assets differing from expert | Assets differing from optimum | Launch vs optimum | Disruptions | Class |
|---|---|---|---|---|---|---|---|
| 20220203_0 | +0.85 | +1.00 | b3 | b3 | n/a | FE | differs from own simulations |
| 20220203_1 | +0.44 | +1.00 | b3 | b3 | n/a | - | differs from own simulations |
| 20240510_1 | +1.00 | +0.86 | b1, b3 | - | differs | FE GO LS SL SS TF | differs from own simulations |
| 20121008_0 | +0.97 | +0.99 | b1, b2, b3 | b1, b2, b3 | differs | SL TF | differs from own simulations |
| 20121008_5 | +0.86 | +0.83 | b1, b3, b4 | b4 | n/a | SS | differs from own simulations |
| 20050902_5 | +0.99 | +0.99 | - | - | differs | FE SL TF | follows own simulations |
| 20050902_11 | +0.99 | +0.99 | - | - | differs | - | follows own simulations |
| 19971001_2 | +0.85 | +1.00 | b1 | b1 | differs | - | differs from own simulations |
| 19971001_11 | +1.00 | +1.00 | b2, b3 | b2, b3 | n/a | SL | differs from own simulations |
| 20151214_3 | +0.76 | +1.00 | b3, b4 | b3, b4 | n/a | FE SL | differs from own simulations |
| 20181104_4 | +1.00 | +0.71 | b3, b4 | - | differs | - | differs from own simulations |
| 19970921_5 | +0.99 | +0.98 | b2 | - | differs | SL | differs from own simulations |
| 19970921_10 | +0.46 | +0.99 | b1, b2, b3 | b1, b2, b3 | differs | LS | differs from own simulations |
| 20130629_0 | +0.86 | +0.86 | - | b3 | n/a | SS | follows own simulations |
| 19961022_4 | +0.85 | +1.00 | b3 | b3 | n/a | FE SL | differs from own simulations |
| 20020911_2 | +0.85 | +1.00 | b1 | b1 | n/a | FE SL TF | differs from own simulations |
| 20020911_3 | +0.99 | +0.99 | - | - | differs | TF | follows own simulations |
| 20161016_7 | +0.51 | +1.00 | b2, b3, b4 | b2, b3, b4 | n/a | FE SL SS TF | differs from own simulations |
| 20000128_10 | +0.99 | +0.99 | - | b2, b3 | differs | FE SS | follows own simulations |
| 20110429_5 | -0.12 | +0.99 | b3 | b3 | differs | GO LS SL | broke window rule |
| 20110429_7 | +0.99 | +0.99 | - | - | differs | LS SS | follows own simulations |
| 20020110_6 | +0.99 | +0.99 | - | - | differs | LS | follows own simulations |
| 20151110_5 | +0.71 | +0.99 | b3 | b3 | differs | LS | differs from own simulations |
| 20240303_9 | +0.88 | +1.00 | b4 | b4 | differs | SS TF | differs from own simulations |
| 20221130_9 | +0.85 | +1.00 | b1 | b1 | n/a | SS | differs from own simulations |
| 20010512_2 | -0.85 | +0.90 | b3 | b2 | differs | SL TF | broke window rule |
| 20010512_11 | +0.65 | +1.00 | b2, b3 | b2, b3 | n/a | FE TF | differs from own simulations |
| 20011028_6 | +0.99 | +1.00 | b3, b4 | b3, b4 | n/a | FE SL | differs from own simulations |
| 20120714_0 | -0.26 | +0.92 | b2, b3 | b2, b3 | differs | GO SS | broke window rule |
| 20120714_3 | +0.83 | +0.99 | b3 | b3 | differs | FE LS TF | differs from own simulations |
| 20121001_6 | +1.00 | +0.89 | b1, b2 | b2, b3 | n/a | TF | differs from own simulations |
| 20121001_7 | +0.81 | +1.00 | b3 | b3 | n/a | FE | differs from own simulations |
| 20001126_4 | +0.81 | +0.95 | b1 | b2 | differs | FE SS | differs from own simulations |
| 20190805_11 | +0.99 | +0.99 | - | - | differs | FE LS SS | follows own simulations |
| 20010923_0 | +0.96 | +0.97 | b2, b3 | b2, b3 | differs | FE LS SL TF | differs from own simulations |
| 20240628_3 | +0.99 | +0.99 | - | - | differs | FE LS | follows own simulations |
| 20210425_2 | +0.99 | +0.99 | - | - | differs | FE SL SS | follows own simulations |
| 20250502_9 | +0.80 | +0.82 | b1, b2, b3 | b3 | n/a | FE SL SS | differs from own simulations |
Disruption codes: FE forecast_error, GO ground_outage, LS launch_slip, SL storm_longer, SS storm_stronger, TF thruster_failure, - none.
Rewards are final rewards; "differs" under launch means the launch decision differs from the stored optimum.

## 2. Does the submission contradict what the model's own tools returned?

| Class (non-optimal tasks) | Tasks |
|---|---|
| Tool-call error (error observation, duplicate call, wrong horizon) | **0** |
| Ignored tool output: explicit contradiction of a tool statement | **0** |
| Ignored tool output: submission differs from what its own simulations imply (no tool statement contradicted) | **25** |
| Wrong planning despite correct tool use: plan follows its own simulations, shortfall is the expert's/information gap | **10** |
| Wrong planning: broke a window rule stated in the briefing, no tool said so | **3** |

Reading this honestly:
- **No tool-call errors.** Every call was well formed, none was a duplicate, all simulations used the 48 h horizon.
- **No explicit contradictions.** The specific failure the environment penalises (raising an asset after `check_feasibility` said thrust
  does not slow its decay) did not occur, and neither did deploying below `min_safe_altitude`.
- **The "ignored" class is an inference, not a quoted contradiction.** In 25 tasks the submission differs from the plan its own simulation
  values imply. The model never wrote a reason, so "ignored the tool output" is shorthand for "did not apply the decision rule to the
  numbers it had". It could equally be called imitation error. Across all 50 tasks 35 submissions deviate (28 non-optimal, 7 optimal)
  and 61 of 169 asset actions deviate.
- **Shape of the deviations.** 38 asset actions are `safe_mode_edge_on` where the own-simulation plan raises, and 23 are `raise_orbit`
  where it does not. In 20 tasks a deviation is an in-window swap (the model raises a different asset than the one with the largest
  gain in the same ground window). 12 tasks show only under-raising, 2 only over-raising, 1 both in different windows. The swap
  pattern says the model learned "raise one asset per window" but not reliably which one.
- **Window rule.** In 3 tasks the model raised an asset in an outage window (2) or raised two assets in one window (1). These are the three
  worst rewards in the run (-0.12, -0.26 and -0.85; mean -0.41 against the expert's +0.935 on the same tasks). `check_feasibility` was either not
  called for that asset or did not contradict, so no tool output was ignored; the briefing stated the windows.
- **Following its own numbers works.** On the 15 tasks where the submission follows the own-simulation plan, the model's mean reward is
  +0.985, equal to the expert's on those tasks. On the 35 tasks where it deviates, the model averages +0.756 against the expert's +0.932.
- **Deviation is not always a loss.** On 12 tasks the model's reward is above the expert's, because the expert plan is itself short of
  the optimum there.

## 3. Was the 10-call cap the limit?

**No.** In all 50 tasks the model's tool calls were the expert's calls: the same set of calls with the same arguments, in the same order
(50 of 50 identical sequences; 9.86 calls per task for both). The expert's essential calls are `simulate_decay` for `raise_orbit` and
`safe_mode_edge_on` on every asset, plus `min_safe_altitude` on launch tasks; that is at most 9 calls (4 assets and a launch), so the
cap of 10 never prevents them. The model made every essential call in every task. The cap was reached in 45 of 50 episodes, but only
by the expert's optional calls (`simulate_decay` for `hold`: 110 calls; `check_feasibility`: 20 calls in 15 tasks). The tool-use pattern is
learned exactly; the errors come after the information is in hand.

## 4. Where the failures sit (descriptive)

38 of 50 tasks are non-optimal. A task can carry several disruption types, so the rows overlap and cannot be added.

| Disruption type | Tasks | Non-optimal | Expert also non-optimal | Submission deviates from own simulations | Broke a window rule |
|---|---|---|---|---|---|
| forecast_error | 25 | 18 | 15 | 18 | 0 |
| storm_longer | 20 | 15 | 13 | 17 | 2 |
| storm_stronger | 17 | 13 | 13 | 11 | 1 |
| thruster_failure | 15 | 12 | 11 | 13 | 1 |
| launch_slip | 11 | 10 | 11 | 7 | 1 |
| none (no disruption listed) | 5 | 4 | 2 | 3 | 0 |
| ground_outage | 3 | 3 | 3 | 3 | 2 |

- Non-optimal rates are 72% (forecast_error), 75% (storm_longer), 76% (storm_stronger), 80% (thruster_failure), 91% (launch_slip) and
  100% (ground_outage, 3 tasks), and 80% on the 5 tasks with no disruption listed against 76% overall:
  **no disruption type isolates the failures.** launch_slip is highest among the larger groups, which follows from the launch rule gap
  below; the groups of 3 to 5 tasks carry no weight.
- Launch tasks are worse than no-launch tasks: 23 of 25 non-optimal against 15 of 25. Part of that is the launch rule gap above (11 tasks miss only on the launch).
- The only specific signal is ground_outage: 2 of its 3 tasks contain the window-rule violations, the lowest rewards in the run
  (-0.12, -0.26). Three tasks cannot support a rate.
- Task complexity also tracks failure: tasks with none or one disruption type are non-optimal 16 of 24, with two 12 of 13, with three or more 10 of 13.

## What this does and does not show
- The model reproduced the expert's tool-use policy exactly and its launch rule exactly. It did not reproduce the asset decision rule
  reliably: 36% of asset actions differ from what its own simulations imply, mostly choosing the wrong asset to raise in a ground window
  and under-raising. That is where the 0.124 reward gap to the expert comes from.
- A faithful imitation of the expert would have cleared Gate B on these tasks (expert CI +0.926 to +0.969), so the Gate B miss is a shortfall of
  the learned decision step, not of the tool information. The information gap is real but separate: the expert itself is optimal on only 18
  of 50 tasks, and 25 of the 38 non-optimal model tasks are also non-optimal for the expert.
- Not shown: why the model deviates (no reasoning is recorded), whether more steps, other seeds or a different data mix would fix it, or
  whether the pattern holds beyond these 50 tasks. The per-class counts depend on my definitions above, in particular the own-simulation plan.
- The summary's claim that the model "only matches the rule-based plan" (Gate C) stands; this note does not alter any gate.

**Next step: discuss before any RL.** This diagnosis does not select a method; it only narrows where the error is.
