# Phase 6, session 2: GVPO design and pre-registration

Registered 2026-11-11, **before any GVPO code was written and before any GVPO run or evaluation**. This file is committed
first; the implementation follows in later commits. After this commit, nothing in "Gates" (section 7) changes. A change needs a
new dated section that says why, written before the run it affects. Results go under "Outcome" (section 9) and nowhere else.

Not changed by this phase: the strict parser, the verifier, the environment, the rewards, and the SFT gates
(`docs/phase6_gates.md`). HelioGuard recommends; humans decide.

## 1. Why this experiment, and what it can and cannot show

The SFT model (`docs/phase6_gates.md`, Outcome) learned the tool-use policy and the launch rule exactly but applies the asset
decision rule unreliably (`docs/phase6_diagnosis.md`): 61 of 169 asset actions differ from what its own simulations imply, the
model's mean final reward is +0.825 against +0.948 for the tool-informed expert with identical tool calls, and Gate B failed.
**The target of RL is therefore the decision step**: choosing which asset to raise in each ground window and respecting the
window rules. It is not tool use.

Honest limits, stated now:
- The environment gives no reward for the decision step other than the final reward (section 2). RL here is outcome-driven.
- The expert is itself optimal on only 18 of 50 eval tasks because tools show the forecast storm only. RL cannot recover
  information the tools do not give; a result above the expert is possible but not expected.
- Budget is small (section 5). A null result is a real possibility and will be reported as null.

## 2. Which per-step rewards exist, and do any cover the decision step?

From `helioguard/env.py` and `helioguard/env_tools.py` (constants as in the code; none is changed):

| Step | Reward |
|---|---|
| `tool_call`, executed | +0.05 if the verifier marks the call useful, minus the tool cost (0.02; `qaoa_schedule` 0.10) |
| duplicate call, or a tool that errors | -0.05 minus the tool cost |
| call beyond the 10-call cap | -0.10 |
| claim (`claim_decay` and similar) | +0.10 if correct, -0.10 if wrong; repeated claim -0.05 |
| malformed or unknown action | -0.05 |
| a reply the strict parser rejects (harness) | -0.5, ends the episode |
| turn limit | -0.5 added to the last step |
| `submit_decision` | **the final reward** (`score_plan` minus 0.2 per ignored tool result), or -0.5 if rejected |

**None of these covers the decision step except the final reward itself.** The submit step's reward is the outcome; there is no
per-asset or per-window shaping, and nothing scores the choice of which asset to raise before the final score arrives. The shaping
terms reward and price tool use and claims.

**What this means for "step+final versus outcome-only".** The two reward signals are `total_return` (sum of all step rewards plus
the final reward) and `final_reward`. In the SFT greedy run the shaping part, `total_return - final_reward`, is +0.30 in 45 of 50
episodes, +0.27 in 3 and +0.24 in 2: ten useful calls at +0.03 each, differing only by the number of calls made. After group
centring, a constant cancels, so the two signals can differ only through call count, claims and failures. **This ablation cannot test whether
process rewards help the decision step, because no process reward exists for it.** It tests the much narrower question of whether
the tool-call shaping changes training, and it is expected to show no difference. A null there will not be read as "process
rewards are unnecessary". Building decision-step rewards would change the environment and rewards, which is out of scope here.
The precheck (section 4) measures how often the two signals differ within a group, so this expectation is checked on data.

## 3. Arms, starting point, seeds

| Arm | Algorithm | Reward used per episode | Role |
|---|---|---|---|
| A | GVPO | `total_return` (step+final) | primary |
| B | GRPO | `total_return` (step+final) | baseline for the algorithm |
| C | GVPO | `final_reward` (outcome-only) | ablation (section 2) |

GRPO with outcome-only reward is not run (budget); the algorithm comparison is made at the step+final reward.

- **Starting point and reference.** The SFT adapter of the registered run (Qwen2.5-1.5B-Instruct, final step 200, from the Drive or
  Hub copy). It is merged into the base weights and a fresh LoRA (r 16, alpha 32, same target modules, zero-initialised) is trained on
  top. At step 0 the policy equals the SFT model exactly. The reference policy is the same model with the new adapter disabled,
  so **the reference is the SFT model** (a test checks the log-ratio is 0 at step 0). Disabling the SFT adapter itself would give the
  base model, which cannot produce the action format, so that is not used.
- **Seeds.** Training seeds 0, 1, 2 for every arm: 9 runs. For a given seed all arms see the same task sequence. The seed fixes the task
  order and the sampling seeds, both pure functions of (seed, step), so a resumed run equals an uninterrupted one.
- **Tasks.** Training tasks only (the Phase 4 train split; eval tasks and eval storms are refused). No curriculum this session.

## 4. Precheck (run before any training; go/no-go)

`scripts/gvpo_precheck.py`, on Colab: 100 train tasks (fixed sample, seed 0), 8 attempts each at temperature 0.7 (top_p 0.95, as
the evaluation protocol) with the SFT adapter. It reports, over the 100 groups:
- the fraction of groups whose 8 `final_reward` values are all identical, and the same for `total_return`;
- the within-group spread of both signals, the valid-submission rate, and how often the step+final and outcome-only signals rank
  or centre a group differently.

GVPO and GRPO both need groups whose rewards differ. **Gate G0 (go/no-go): at least 40% of the 100 groups must have non-identical
`final_reward` values.** If fewer, no training is run; the finding is reported and any redesign is a new dated section.
(40% means about 24 informative groups in a 60-group run.)

## 5. Training budget (planning figures, labelled estimates)

| Setting | Value |
|---|---|
| Group size | 8 attempts per task |
| Tasks per step | 2 (16 episodes per optimizer step) |
| Steps | **30** (60 groups, 480 episodes per run; no early stopping, final checkpoint evaluated) |
| Optimizer | AdamW, weight decay 0, learning rate 5e-5 constant after 3 warm-up steps, gradient clip 1.0 |
| GVPO beta | 0.1 |
| GRPO | advantage divided by group standard deviation, KL coefficient 0.04 to the reference |
| Sampling | temperature 0.7, top_p 0.95, 256 new tokens per turn, fp16, gradient checkpointing, T4 |
| Replay | **off**: every update uses only samples from the current policy, none stored |

These hyperparameters are guesses, fixed now and not tuned on eval results. Illustrative timing, from the measured sequential greedy
episodes of the SFT eval (28.9 s per episode on the T4): sampling the 8 attempts of a task in one batch is estimated at about
1 minute, an update at about 8 s per episode (as in SFT). That gives roughly 4 minutes per step, about 2 hours per run, **about 18 T4 hours
for the 9 runs**, plus about 3.5 hours for the 9 greedy evaluations and about 1.5 hours for the precheck: **about 23 T4 hours in
total, with an uncertainty of roughly a factor of 1.5 either way**. The precheck and the first run replace these with measured
numbers. The number of steps stays 30 whatever the timing shows.

## 6. Evaluation protocol

- The same 50 held-out eval tasks as Phase 6, **greedy, one attempt**, the final (step 30) checkpoint of every run, no checkpoint selection.
  Evaluation uses the existing resumable `sft_eval` path (per-episode files, Hub mirror).
- Per-task final rewards are averaged over the 3 seeds of an arm, then a 95% percentile bootstrap over tasks (10,000 resamples, seed 0),
  as in Phase 6. Percent optimal is the harness definition (valid and final reward >= 0.999).
- **Paired comparison against the SFT result**: per-task difference (arm minus SFT greedy, the 50 per-task rewards of the SFT
  evaluation), same bootstrap. Paired comparisons between arms use the same method.
- **Reference points:** SFT mean +0.825 (95% CI +0.717 to +0.908), SFT percent optimal 24%; tool-informed expert +0.948 (CI +0.926 to
  +0.969, a reference and not a gate); no-tool plan +0.743.

## 7. Gates

Applied to the **primary arm (A)**. Decided on greedy, seed-averaged per task unless stated.

| Gate | Rule |
|---|---|
| G0 precheck | at least 40% of the 100 precheck groups have non-identical `final_reward` (section 4) |
| G1 format | valid-submission rate >= 0.90 in each of the 3 seeds |
| G2 improvement (primary) | the paired mean difference in final reward, arm A minus SFT, has a 95% CI **lower bound above 0** |
| G3 value | mean final reward 95% CI lower bound above **+0.743** (the no-tool plan; the Phase 6 Gate B line) |
| G4 consistency | the paired mean difference, arm A minus SFT, is above 0 in each seed taken alone (point estimates) |
| G5 percent optimal | seed-averaged percent optimal above SFT's 24% (point estimate; the CI is reported) |

**GVPO is said to improve on the SFT model only if G0 to G5 all hold.** Otherwise the outcome states which gates failed, and the result is
"no demonstrated improvement". Further rules:
- No gate, threshold, metric, seed count, step count or hyperparameter changes after any GVPO result has been seen. No extra seeds, steps
  or arms are added to rescue a failed gate. Any extension is a new dated section, labelled exploratory, and cannot change a verdict.
- **Secondary, not gates:** arm A minus arm B (GVPO against GRPO) and arm A minus arm C (step+final against outcome-only), paired, with
  CIs. "GVPO is better than GRPO" is claimed only if the CI lower bound of A minus B is above 0; with 3 seeds and 50 tasks that
  comparison has little power and an indistinguishable result is the expected one. Also reported: distance to the expert (+0.948), tool
  calls per episode, rate of window-rule violations and of explicit tool contradictions, per-disruption breakdown (descriptive), and the
  training curves (mean group reward per step, fraction of groups with identical rewards, KL to the reference).
- Training uses only samples from the current policy (replay off); no result from a replayed or stored-trace variant is reported as GVPO.
- Never claim quantum advantage; nothing in this phase uses a quantum component.

## 8. Method, as implemented (checked by tests)

For a group of k samples with rewards R_i, sequence log-probabilities log pi_theta(y_i) over all assistant tokens of the episode (tool results are
context), and reference log-probabilities log pi_ref(y_i) from the disabled adapter:
- centred reward c_i = R_i - mean(R), **never divided by the group standard deviation**;
- weight w_i = c_i - beta * (d_i - mean(d)), d_i = log pi_theta(y_i) - log pi_ref(y_i); the weights **sum to zero** in a group (asserted);
- loss = (1/2) * sum_i (c_i - beta * (d_i - mean(d)))^2 averaged over groups and members, whose gradient is -beta * sum_i w_i * grad log pi_theta(y_i).

The loop checkpoints (adapter, optimizer, step, RNG states, config hash, log) on the SFT pattern, reads the step back from the written
file, and mirrors every checkpoint to a private Hub repo with a verified listing; it resumes from Drive or the Hub, whichever is further along.

## 9. Outcome

Not run yet.

## 10. Clarifications (2026-11-13, before the precheck and any GVPO run; no gate changed)

- **Reference model.** The SFT adapter is merged into the base weights and a fresh zero-initialised LoRA is trained on top, so the
  reference policy (new adapter disabled) is the SFT model. Disabling the SFT adapter itself would give the base model, which cannot
  produce the action format. The test that checks the log-ratio is 0 at step 0 is `tests/test_gvpo.py::test_reference_is_the_model_with_the_adapter_disabled` (it asserts that the fresh-adapter policy log-probability equals the reference to 1e-4, and that the reference does not move when the adapter is changed); the loop tests also assert `mean_log_ratio` is 0 at step 1.
- **Budget.** 30 steps of 2 tasks x 8 attempts is 60 tasks per run. G2 to G5 may come out null because training is short, not because
  GVPO cannot help; such a result is reported as "no demonstrated improvement", not as evidence against the method. This risk is
  accepted before the precheck.
- **Order of runs.** Arm A (3 seeds) first, then arm B, then arm C. Arm C is lowest priority and runs last; if units run out, it is
  reported as not run. Arms A and B are not dropped to make room for C.
- **Ablation.** No per-step reward covers the decision step (section 2), so A minus C is expected to be null and is not evidence
  about process rewards.
- **Precheck reporting.** The precheck also reports groups split by whether the task was in the SFT trace set or was one of the 200
  dropped tasks. Descriptive only; G0 is unchanged.
- **Expert optimality figure.** Section 1 states that the expert is optimal on 18 of 50 eval tasks. That figure has not been re-checked
  against `docs/phase6_diagnosis.md`; where the two differ, the diagnosis document is authoritative. No gate depends on it.
