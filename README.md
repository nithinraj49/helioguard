# HelioGuard: Space-Weather Decision Engine

Physics-verified AI decisions for spacecraft operators facing solar storms.
Stack: LLM policy trained with **GVPO** + **Verifiable Process Rewards (VPR)**,
physics verifiers (orbit decay, drag, feasibility), and a quantum/QUBO layer for
fleet-wide response scheduling.

## Architecture
```
Solar data (SunPy / NOAA SWPC) + Fleet state
   -> LLM policy: step-by-step decision trace (JSON steps)
   -> VPR verifier (this repo): each step checked by physics -> dense rewards
   -> GVPO training on group-centred rewards
   -> Quantum/QUBO: fleet-wide action schedule (vs OR-Tools baseline)
   -> Operator briefing (plain English) -> human approves
```

## Phase 0 (done): setup and locked results
Tooling: `pyproject.toml`, a virtual environment, and pytest. 8 tests in
`tests/test_phase1_baseline.py` lock the Phase 1 results, so any physics change shows up as a failure.
Spacecraft values are illustrative, not operator specifications.

| Feb 2022 replay, 210 km, 48 h | Quiet atmosphere | Storm (density x1.5) |
|---|---|---|
| Raise orbit | Survives, about 243 km | Infeasible (drag 85.1 mN > 70 mN thrust); lost at hour 31.8 |
| Edge-on safe mode | Survives, about 166 km | Lost at hour 35.3 (best available) |
| Hold | Lost at hour 19.9 | Lost at hour 13.2 |

Verifier traces on the storm: naive 0/4, physics-aware 4/4; GVPO group-centred rewards -0.5 and +0.5.

Setup and tests:
```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

## Phase 1 engine core (done): `verispace_engine/`
Domain-free package shared with the Navigator project; a test fails if it ever mentions HelioGuard.
- `tools.py`: `Tool` protocol (name, JSON schema, `run()`) and `ToolRegistry` (validates arguments, turns tool errors into failed results)
- `schema.py`: pydantic `TraceStep`, `DecisionTrace`, `ToolCall`, `ToolResult`
- `verifier_base.py`: `VerifierBase` with `add_rule()`, `score_step()`, `score_episode()`; unknown, malformed and empty steps earn 0
- `env_base.py`: `EnvBase` with `reset()` / `step()`

HelioGuard's verifier now runs on `VerifierBase` (`HelioGuardVerifier`); the Feb 2022 results are unchanged (naive 0/4, physics-aware 4/4). 25 tests pass.

## Phase 7, session 1 (done): QAOA scaling benchmark

Pre-registered gates: `docs/phase7_gates.md`. Full tables, CIs and verdicts: `results/phase7/results.md` and `summary.json`.
Spacecraft values are ILLUSTRATIVE. QAOA ran on classical statevector simulators; **no quantum advantage is claimed or
tested for** (G6). `solve_qaoa` ("engine arm") was benchmarked unchanged against exact enumeration, OR-Tools CP-SAT and
uniform random sampling, on HelioGuard scheduling QUBOs: family B (2 actions per asset) at 6, 8, 10, 12, 14, 16 qubits and
family A (the real 3-action fleet) at 6, 9, 12, 15; 10 instances per size, 3 optimiser seeds, depth p = 1 to 4. A numpy
simulator pinned to the PennyLane circuit (`verispace_engine/qaoa_bench.py`) added 3 optimisers x 2 starts, and a reduced
run at 18 and 20 qubits (5 instances).

Family B, engine defaults (COBYLA, random start), p = 2; mean over instances [95% bootstrap CI]:

| qubits | P(optimum), one sample | uniform sampling | approx. ratio | top-64 readout finds optimum | 64 shots find optimum | uniform, 64 samples |
|---|---|---|---|---|---|---|
| 6 | 0.114 [0.092, 0.133] | 0.016 | 0.74 | 1.00 | 0.93 | 0.64 |
| 8 | 0.049 [0.042, 0.056] | 0.004 | 0.67 | 1.00 | 0.89 | 0.22 |
| 10 | 0.024 [0.017, 0.031] | 0.001 | 0.65 | 0.87 [0.67, 1.00] | 0.70 | 0.06 |
| 12 | 0.010 [0.008, 0.012] | 2.4e-4 | 0.67 | 0.80 [0.67, 0.93] | 0.43 | 0.02 |
| 14 | 0.005 [0.004, 0.006] | 6.1e-5 | 0.70 | 0.63 [0.50, 0.77] | 0.28 | 0.004 |
| 16 | 0.002 [0.002, 0.003] | 1.5e-5 | 0.63 | 0.60 [0.47, 0.77] | 0.14 | 0.001 |

Exact enumeration and OR-Tools solved every instance (110 of 110, OR-Tools within its own reward rounding) in a few
milliseconds; serial QAOA at 16 qubits took 9 s (360 simulator evaluations) against 1 ms for OR-Tools.

- **G1, beats chance: pass in every cell** (7x to 162x more optimum probability than uniform sampling), and the gap grows with size.
- **G2, solves the problem: mostly fails.** Single-shot P(optimum) >= 0.5 fails at every size. The top-64 readout (simulator-only) passes
  its reliability gate at 6 and 8 qubits (family B) and 6 and 9 (family A) and fails from 10 and 12 qubits respectively. The top-64 readout needs
  the full probability vector, which hardware does not give; the 64-shot column is the hardware-style reading and is lower.
- **G3, scaling: P(optimum) decays with size**, about -0.15 to -0.17 log10 per added qubit (family B), -0.18 to -0.22 (family A); every CI is below zero.
- **G4, depth: mostly no detectable change** (21 of 30 paired p to p+1 comparisons; 9 "helps", none "hurts"). Four layers do not rescue the large sizes.
- **G5, optimiser and start (same evaluation budget):** L-BFGS-B from the ramp start helps in 16 of 20 cells and hurts in none; Nelder-Mead from the ramp
  start hurts in 7 and helps in 8; COBYLA/ramp and Nelder-Mead/random are mostly indistinguishable from the default. Gains are small in absolute terms.
- **Engine vs numpy harness (COBYLA/random):** paired differences about 0.000 to 0.004; the harness is a valid stand-in.
- **Limit.** Run up to 20 qubits (harness arm, p = 2): P(optimum) about 5e-4 at 20 qubits against 1e-6 uniform. Serial cost per evaluation is
  0.04 s at 16, 0.13 s at 18 and 0.69 s at 20 qubits for the PennyLane engine; larger sizes were not tried.
- **Negative results and caveats.** 10 instances per cell is small. Family B and A are different problems; do not compare across them. Wall times from the
  parallel run were contended (the serial table is the one of record). At 6 qubits, plain uniform random sampling with the same number of samples as QAOA's circuit
  evaluations (about 330) finds the optimum more often (0.994) than 64 QAOA shots (0.93; family A: 0.994 vs 0.89), so QAOA is no cheaper there; from 8 qubits up
  QAOA's 64 shots do better (8 qubits: 0.89 vs 0.74; 12 qubits: 0.43 vs 0.08), but that is shots against a different sample count, not a cost comparison.
- **Engine bug fixed.** `build_qubo`'s penalty (range + 1) was too weak for large, similar rewards, making the QUBO minimum infeasible on 7 of 50 probe
  instances. The engine default is now `penalty_rule="sufficient"`; Phase 6 paths pass `"legacy"` and reproduce exactly (golden digests for all 1,050 tasks;
  `data/sft/train_traces.jsonl` regenerates byte-identical).

Run: `python scripts/phase7_run.py run`, `python scripts/phase7_run.py timing`, `python scripts/phase7_report.py`. Raw per-run caches are git-ignored.

## Phase 6, session 1 (done): SFT warm start, set up but not yet trained
Goal: teach a small Qwen the tool-call and submit format and the tool-informed behaviour, so RL starts from a policy that produces valid submissions (the base 0.5B had 0 percent). **No GVPO and no 1.5B training happened in this session**; the 1.5B run is a Colab T4 job (notebook below), so there is no SFT result yet. The gates were written first and pushed before any trace was generated: `docs/phase6_gates.md` (A: valid-submission rate >= 90 percent; B: mean final reward above the no-tool plan +0.743 with the 95 percent bootstrap CI excluding it; C: say so if it only matches rule-based +0.727; gates are decided on greedy decoding).

**Traces** (`helioguard/sft_traces.py`, `scripts/generate_sft_traces.py`, data card in `data/sft/data_card.json`). The tool-informed expert was run on the 1,000 TRAIN tasks only (generation refuses eval tasks; a test checks no task id or storm is shared with eval). Kept: valid submission and final reward >= 0.90 (the same "near-optimal" line used elsewhere). 800 of 1,000 survive: 283 exactly optimal, 517 between 0.90 and 0.999 (mean 0.989). The 200 dropped scored 0.0 to 0.9. Disruption mix of the kept set: forecast error 372, storm longer 312, thruster failure 254, storm stronger 245, ground outage 194, launch slip 187, none 77. About 9.75 tool calls and 3,200 tokens per trace (max 3,415). Generation is deterministic; the seed in the card is the training and shuffle seed.

Traces use the same functions as the harness (system prompt, briefing, observation messages), and every assistant turn is the compact JSON of the action, which the unchanged strict parser accepts (a test pins the parser source hash and round-trips every turn). Replaying a trace's calls through the environment reproduces its recorded rewards. **One change to the harness view:** raw tool observations were about 1,000 tokens each (a 10-call episode was about 12,000 tokens, too long for a T4 and for the CPU smoke test), so the model now sees a compact observation (floats rounded to 3 decimals, and the hourly altitude series of `simulate_decay` replaced by altitudes every 12 h; final altitude and re-entry hour are kept). The environment and rewards are untouched; the Phase 5 numbers for the base Qwen used the raw view. Limitations: dropping traces below 0.90 over-represents tasks the expert solves, and the expert never makes claims or recovers from errors, so the model sees no recovery examples.

**Training** (`verispace_engine/sft.py`, `scripts/train_sft.py`, `configs/sft_smoke_0p5b.json`, `configs/sft_1p5b.json`). LoRA through PEFT, or Unsloth when installed on a GPU (fp16 on a T4); loss on assistant tokens only (system prompt, briefing and tool outputs are context; a test checks the mask with a fake and the real Qwen tokenizer). Resumable: a checkpoint (adapter, optimizer, GradScaler, step, all RNG states) is written every N minutes to a folder you point at Drive, and rerunning the same command continues; a test shows a stopped-and-resumed run gives the same losses and adapter as an uninterrupted one. The CPU smoke test (10 steps on the 0.5B model) passes (`pytest -m slow`, about 10 minutes; slow tests are skipped by default). `notebooks/phase6_sft_colab.ipynb` clones the repo (token from a Colab secret, not stored), mounts Drive, installs, trains, resumes and evaluates; it contains no secrets.

**Evaluation hook** (`python -m helioguard.sft_eval --model Qwen/Qwen2.5-1.5B-Instruct --adapter <adapter folder>`): the Phase 5 harness on the 50 held-out tasks, greedy once and temperature 0.7 with 3 seeds, 95 percent bootstrap CIs over tasks; valid-submission rate (primary), mean final reward, percent optimal, tool calls per episode, per-disruption breakdown, and the gate verdicts. Not run yet (no adapter exists).

## Phase 5 (done): the harness and a first evaluation
`verispace_engine/harness/` (domain-neutral, reusable by the Navigator): `prompts.py` (system prompt with tool schemas, observation messages), `parser.py` (strict: the whole reply must be one JSON action, optionally in one code fence; no repair), `rollout.py` (N attempts per task, env-enforced turn and tool-call caps, JSONL log of every turn, tool call, step reward and final reward), `backends.py` (local Hugging Face transformers on CPU or GPU, an OpenAI-compatible HTTP endpoint with the key read from the environment or `.env`, and a scripted backend for tests), `eval.py`. HelioGuard plugs in through `helioguard/harness_env.py`. One command:

    python -m helioguard.harness_cli --model Qwen/Qwen2.5-0.5B-Instruct --split eval --n 50
    python -m helioguard.harness_cli --model baseline:tool-informed --split eval --n 50

A reply the parser rejects ends the episode as a failure (-0.5). Logs and summaries are in `results/phase5/`. Tests: `tests/test_harness.py`, `tests/test_tool_informed.py`.

**Tool-informed baseline first.** `ToolInformedPolicy` decides only from the briefing and tool results (a test locks the stored answers so touching them fails). It simulates raise and safe mode for every asset over the 48 h horizon, then hold and `check_feasibility` while the 10-call cap allows, asks `min_safe_altitude` for the launch, and submits the best plan that respects the ground windows.

| Policy (final reward) | Train (1,000) | Eval (50) |
|---|---|---|
| Same plan, no tool calls | +0.702 | +0.743 |
| Tool-informed | +0.945 | +0.948 |

It clearly beats the no-tool plan, so Phase 4 is unchanged. One caution for training: it reaches the exact optimum on 80 percent of train and 74 percent of eval tasks, because the forecast-storm simulation is nearly the whole story; only the stronger and longer storm cases and the launch altitude are not visible to any tool. That leaves limited headroom for RL. If you want it harder, the change to consider is hidden information that only tools reveal (for example a tool-only disclosure of the true storm peak or a ground outage not in the briefing). I have not made that change.

**First model evaluation** (50 eval tasks, Qwen2.5-0.5B-Instruct on CPU, 1 attempt, greedy):

| Policy | Mean final reward | Valid submissions | Tool calls / episode |
|---|---|---|---|
| Oracle (upper bound) | +1.000 | 100% | 3.4 |
| Tool-informed | +0.948 | 100% | 9.9 |
| Same plan, no tool calls | +0.743 | 100% | 0 |
| Rule-based | +0.727 | 100% | 3.4 |
| Always raise | -1.353 | 100% | 0 |
| **Qwen2.5-0.5B-Instruct** | **-0.500** | **0%** | 2.1 |

Failure types for Qwen (50 episodes): 25 malformed output (a valid JSON action followed by extra text, so the strict parser rejects it), 1 other malformed output (invalid JSON), 23 rejected submissions (for example an asset set to null, or a missing asset), 1 never submits. 31 of 50 first replies were a legal tool call. In short, the model cannot yet follow the output format; no episode reached a scored decision, so -0.5 is the failure reward, not a measure of its physics skill.

**Limitations.** The model is the smallest practical one (0.5B), run on CPU in float32 at about 1 minute per episode (a 50-task pass takes about an hour), as a single greedy attempt, so there is no variance estimate; the default of 8 attempts per task was not run for cost. The strict parser is deliberate, and a more lenient one would score this model differently (the extra-text failures dominate). Prompt wording was written once and not tuned. Larger models and the OpenAI-compatible endpoint path are implemented and unit-tested with a fake endpoint, but not run here. Nothing here is a claim about any trained agent: no training has happened.

## Phase 4 (done): multi-turn RL environment
`helioguard/env.py` (`HelioGuardEnv`, on the engine `EnvBase`), with `env_tools.py`, `env_verifier.py`, `tasks.py`, `storms.py`, `policies.py`. Spacecraft, altitudes and the Kp-to-density mapping are illustrative; storm dates and Kp series are real. HelioGuard recommends; humans decide.

**Episode.** `reset()` returns a briefing: fleet, storm forecast with its uncertainty, ground windows, an optional planned launch, and the disruptions in force. The agent can call the five tools (`simulate_decay`, `check_feasibility`, `min_safe_altitude`, `qaoa_schedule`, and `qnn_density_correction`, which works while disabled and returns the plain density with its reason), state checkable claims, and end with `submit_decision`. Tool calls are capped at 10 and turns at 20.

**Raising and `check_feasibility`.** The tool returns two separate fields. `can_gain_altitude` is thrust above drag (now, and at the forecast storm peak). `thrust_slows_decay` is whether raising outlives both safe mode and hold in simulation. They can disagree: raising beats safe mode until drag is about 1.6x thrust, so the optimum sometimes raises although the satellite cannot climb. Raising is therefore always a legal action, valued by simulation; only `thrust_slows_decay = false` followed by a raise is penalised. Claims use the same two names (`claim_can_gain_altitude`, `claim_thrust_slows_decay`). A test (`test_optimal_trace_with_drag_above_thrust_gets_no_step_penalty_and_rewards_agree`) takes a task whose stored optimum raises with drag above thrust and checks that the optimal trace has no penalty and that step and final rewards are both positive.

**Rewards (no LLM judge).** Step rewards come from the verifier: a new, useful tool call +0.05, minus a cost per call (0.02, and 0.10 for QAOA, which only counts as useful when assets compete for a ground window); invalid or duplicate calls -0.05; a true claim +0.10, a false one -0.10 (a claim must be within 10 percent or 0.5 km of the physics, so vague predictions fail); raising an asset although `check_feasibility` said `thrust_slows_decay` is false -0.20. The final reward comes from the gap to the exact optimum (OR-Tools CP-SAT) of the worst-case total over three storm realisations (forecast, stronger, longer): 1.0 at the optimum, 0 when the gap is half a point per decision, floor -1, and -0.3 per rule violation (raise in an outage window, two raises in one ground window). A malformed submission ends the episode at -0.5, and so does running out of turns without submitting.

**Tasks.** Each task is built from a real storm (Kp series since 1996) with random fleets, altitudes near the point where raising stops working, and 0 to 3 disruptions from: forecast error, storm stronger, storm longer, thruster failure, ground-station outage, launch slip. Each task stores its exact optimum. Tasks the simple rule-based policy already solves (final reward 0.9 or more) were filtered out. A storm is in exactly one split; Feb 2022 and May 2024 are eval storms. Regenerate with `python scripts/generate_tasks.py`.

| | Train | Eval |
|---|---|---|
| Tasks | 1,000 | 50 |
| Storms | 240 | 28 |
| Storm class (G1 / G2 / G3 / G4 / G5) | 484 / 303 / 143 / 70 / 0 | 27 / 10 / 11 / 0 / 2 |
| With a planned launch | 656 | 25 |
| Assets contend for a ground window (QAOA relevant) | 859 | 47 |
| Optimum proven optimal by CP-SAT | 1,000 | 50 |

**Baseline scores** (`python scripts/evaluate_baselines.py`, `tasks/baselines.json`). Return is the episode total; final is the end-of-episode reward. The rule-based policy raises whenever `can_gain_altitude` is true and the window is free, otherwise safe mode, and deploys at the lowest altitude where raising can gain altitude at deployment; by construction it is not near-optimal on these tasks, so its score is a floor, not a typical result. The oracle reads the stored optimum and is an upper bound.

| Policy | Train return | Train final | Eval return | Eval final |
|---|---|---|---|---|
| Oracle (upper bound) | +1.095 | +1.000 | +1.101 | +1.000 |
| Rule-based | +0.782 | +0.687 | +0.828 | +0.727 |
| Same plan, no tool calls | +0.702 | +0.702 | +0.743 | +0.743 |
| Random | -0.059 | -0.059 | -0.170 | -0.170 |
| Always raise | -1.332 | -1.332 | -1.353 | -1.353 |
| Vague claims, then the rule-based plan | +0.228 | +0.702 | +0.236 | +0.743 |
| Endless tool calls, never decides | -1.200 | none | -1.200 | none |
| Never submits | -1.500 | none | -1.500 | none |

**Loophole tests** (`tests/test_env_loopholes.py`): always-raise, never calling tools, endless tool calls, never submitting, vague predictions, random play and farming step rewards each lose reward against the proper policies. **Limits:** the Kp-to-density mapping and spacecraft are illustrative; tasks use the reference atmosphere with a storm profile (not NRLMSIS replays, for speed); `can_gain_altitude` is judged at the start of a burn and at the forecast peak, while the value of raising comes from the simulation; the rule-based policy with tools scores slightly below the same plan without tools because it follows its rule even after `thrust_slows_decay` says false and takes the 0.20 penalty; the `qnn_density_correction` tool returns the reference-atmosphere density because the correction is disabled (Phase 3); the optimum is for the three modelled storm realisations, not every possible storm.

## Phase 3 (done): tools
Five tools on the engine `Tool` protocol, in `helioguard/tools/`. Spacecraft values are illustrative. HelioGuard recommends; humans decide.

| Tool | What it does | Tested against |
|---|---|---|
| `simulate_decay` | hourly altitude for an action under the scenario's storm | the Phase 1 physics (reentry hour 35.3 for edge-on, Feb 2022 multiplier case) |
| `check_feasibility` | thrust versus drag, both in mN | 85.1 mN drag against 70 mN thrust |
| `min_safe_altitude` | lowest altitude where raising works | the project document: about 218, 228 and 243 km at x1.5, x2, x3 density |
| `qaoa_schedule` | one action per asset, one raise per ground window, as a QUBO solved by PennyLane QAOA on `lightning.qubit` | OR-Tools CP-SAT and the exact optimum, always returned alongside |
| `qnn_density_correction` | NRLMSIS density times a learned correction (6 qubits, AngleEmbedding + StronglyEntanglingLayers + linear head, 61 parameters) | a same-size MLP (57 parameters) and uncorrected NRLMSIS; **disabled until it passes the held-out check** |

**QAOA versus OR-Tools** (Feb 2022 scenario, 3 actions per asset, simulator, seed 0; QAOA picks the lowest-energy feasible bitstring among its 64 most probable samples, a classical post-processing step):

| Fleet | QAOA depth | QAOA objective | OR-Tools / exact | Probability on the optimum | QAOA time | OR-Tools time |
|---|---|---|---|---|---|---|
| 3 assets, 9 qubits | 1 | 2.517 | 2.636 | 0.3 % | 0.6 s | 5 ms |
| 3 assets, 9 qubits | 2 | 2.636 | 2.636 | 2.4 % | 1.4 s | 1 ms |
| 3 assets, 9 qubits | 3 | 2.636 | 2.636 | 2.3 % | 2.4 s | 1 ms |
| 4 assets, 12 qubits | 1 | 2.779 | 4.426 | 0.0 % | 0.5 s | 2 ms |
| 4 assets, 12 qubits | 2 | 3.652 | 4.426 | 0.2 % | 2.3 s | 1 ms |
| 4 assets, 12 qubits | 3 | 4.276 | 4.426 | 0.0 % | 3.3 s | 1 ms |

OR-Tools solves every instance to the exact optimum in milliseconds. Shallow QAOA reaches it on the 9-qubit fleet only at depth 2 or more and only through the sample selection above; at 12 qubits it stays below the optimum. No quantum advantage is claimed, and none is seen.

**QNN density correction: tested on real Space-Track data, did not pass, stays disabled.**

Setup (`configs/qnn_events.json`, `train_qnn_density.py`, summary in `reports/qnn_density_report.json`): the target is the log ratio of decay-derived effective density to NRLMSIS (storm-time mode). Training uses passive rocket bodies (no thrusters hiding drag) from five storms: Sep 2017, Apr 2023, Mar 2024, Oct 2024 and Jan 2026. **May 2024 is the primary held-out storm. Feb 2022 (Starlink Group 4-7) is a quiet-time "do no harm" check across the ballistic band 0.071 to 0.192 m^2/kg and is never trained on.** Objects tracked in a test storm were removed from training: 8 unique objects (10 object-storm pairs), cutting the training samples from 801 to 649. Each number is the mean of 3 seeds, 100 epochs, QNN with 61 parameters beside a same-size MLP with 57. Error is the mean absolute error of the altitude change over each element-set interval of at least 3 hours, starting from the observed altitude.

| Test | n intervals | Uncorrected NRLMSIS | QNN | Same-size MLP |
|---|---|---|---|---|
| **May 2024 (held-out)** | 175 | 0.061 km | 0.062 km (-2.3 %) | 0.055 km (+10.2 %) |
| Leave-one-storm-out, mean of 5 folds | 113 to 184 per fold | | -8.5 % | -0.3 % |
| LOSO fold: Oct 2024 | 184 | 0.042 km | 0.058 km (-37.0 %) | 0.053 km (-25.7 %) |
| LOSO fold: Jan 2026 | 172 | 0.053 km | 0.042 km (+21.1 %) | 0.043 km (+18.9 %) |
| LOSO fold: Sep 2017 | 113 | 0.013 km | 0.015 km (-18.0 %) | 0.014 km (-5.4 %) |
| LOSO fold: Apr 2023 | 150 | 0.077 km | 0.079 km (-2.6 %) | 0.067 km (+12.8 %) |
| LOSO fold: Mar 2024 | 182 | 0.021 km | 0.022 km (-6.0 %) | 0.022 km (-2.1 %) |
| Feb 2022 quiet check, B = 0.071 | 14 | 114.8 km | 49.8 km | 73.9 km |
| Feb 2022 quiet check, B = 0.125 | 14 | 599.0 km | 398.2 km | 479.5 km |
| Feb 2022 quiet check, B = 0.192 | 14 | 1421.0 km | 979.1 km | 1166.7 km |

(Percentages are improvement over uncorrected; positive is better.)

**Gates (all must pass; the MLP never counts):** at least 5 % better than uncorrected on May 2024 (**fail**, -2.3 %), at least 5 % better on average across the leave-one-storm-out folds (**fail**, -8.5 %), and no worse than uncorrected on Feb 2022 at both ends of the band (formally pass). **The tool stays disabled.** The QNN did not beat uncorrected NRLMSIS on the held-out storm, and it did worse than the same-size MLP on May 2024 and across the folds. The MLP was 10.2 % better on May 2024, but it is not a gate and was not trained to be deployed; one fold of five (Jan 2026) shows a clear gain for both models and four show losses or near-zero change. No quantum advantage is claimed, and none is seen.

**The Feb 2022 check is not informative, and its "pass" should not be read as evidence.** The errors are hundreds of km because the check is dominated by one object (51470, at 158 to 222 km, near re-entry) whose uncorrected prediction runs away (single-interval errors of 4007, 2007 and 1520 km), plus a 358 km error on the last interval of object 51456. Only 14 intervals qualified. Any correction that lowers density reduces that blow-up, so "no worse" here mostly says the QNN lowers density at low altitude. A robust version of the metric (stop at re-entry, cap or use the median) is a follow-up, and the gates were not changed after seeing these results.

**Limitations:**
- **Altitude gap:** training objects are at about 280 to 430 km; Group 4-7 was at about 160 to 280 km, outside what the model saw.
- **Tumbling rocket bodies:** their ballistic coefficient is uncertain and can change over time. Each was estimated from its own pre-storm element sets with NRLMSIS, so it absorbs any density bias.
- **Feb 2022 data starts after the storm:** the first element sets are 5 Feb 16:10 UTC, so that event can only test quiet-time behaviour. The 0.071 to 0.192 band rests on 2 intervals of 1 object plus an assumed 15 % density uncertainty.
- **Jan 2026 is recent data** that may be incomplete or revised. It is also the fold where both models gain most.
- **Small signal:** the per-interval errors are about 0.01 to 0.08 km, comparable to TLE fitting noise, and the targets are noisy. TLE mean motion is treated as circular-orbit mean motion. Seed-to-seed spread is 0.0003 to 0.0046 km.
- **Single configuration:** one architecture, 100 epochs, learning rate 0.05, no tuning. A different setup might do better; this result only says this one did not pass.

## Phase 2 (done): real physics and data
- `helioguard/data/noaa.py`: Kp, Ap and F10.7 from the CelesTrak SW-All file (NOAA/GFZ data), with caching (`data/cache.py`)
- `helioguard/data/celestrak.py`: GP elements, TLE history through sgp4, and SATCAT decay records. CelesTrak serves current elements only; real orbit histories need Space-Track, so the TLE reader takes text from any source
- `helioguard/atmosphere_msis.py`: NRLMSIS density (pymsis) driven by the real indices, orbit-averaged over a small lat/lon grid (illustrative). Falls back to the old reference atmosphere and says which one it used
- `helioguard/replays.py`, `demo_replays.py`: 3 to 5 Feb 2022 (210 km) and 10 to 12 May 2024 (350 km, illustrative). Offline fixtures in `tests/fixtures/`; no test touches the network
- `physics.py` takes an optional `density_fn`; with none, results are identical to Phase 1 (the Phase 0 tests still pass)

| Real-index replay, 48 h (illustrative spacecraft) | Raise orbit | Edge-on | Hold |
|---|---|---|---|
| Feb 2022 at 210 km (G1, Kp 5.3) | survives, 259 km | survives, 181 km | lost at hour 24.4 |
| May 2024 at 350 km (G5, Kp 9.0) | survives, 428 km | survives, 347 km | survives, 343 km |

**Finding.** With real indices, NRLMSIS gives a modest density rise for the Feb 2022 storm and no loss of the satellite when raising. The Phase 1 result (raise orbit infeasible, every action lost) came from the illustrative x1.5 multiplier, not from the model. Nothing was tuned to recover it. The replays drive NRLMSIS in storm-time mode (3-hourly ap); `build_replay(..., storm_time=False)` gives the daily-Ap mode that pymsis uses by default.

Density change at 210 km, 3 to 5 Feb 2022 versus the quiet baseline of 2 Feb 2022 (24 h mean; 5 lat x 4 lon grid; `storm_density_change()`):

| NRLMSIS input mode | Peak at one grid point | Peak, orbit-averaged | Mean over the 72 h, orbit-averaged |
|---|---|---|---|
| Daily Ap (pymsis default) | x1.30 | x1.09 | x1.05 |
| Storm-time (3-hourly ap) | x1.34 | x1.13 | x1.07 |

An earlier version of this note reported about x1.02; that compared against a start time already inside the storm onset, and the quiet-day baseline above is the right comparison. May 2024 peaks at about x1.32 at 350 km relative to the replay start (not the quiet-day baseline).

**How to read the "up to 50 percent".** SpaceX's figure compared the storm with earlier launches, not with the preceding quiet day, and part of that difference reflects rising solar activity over the previous months (observed F10.7 averaged about 75 in Feb to Apr 2021 and about 105 in Jan 2022, from the SW-All file used here). Peer-reviewed comparisons also find that NRLMSIS underestimates this storm: Lin et al. (2022) report an NRLMSIS 2.0 peak enhancement about half of what Swarm measured, and He et al. (2023) find MSIS2.0 underestimates the density enhancement by about 50 percent. Berger et al. (2023) find 20 to 30 percent enhancement at the 210 km staging altitude from NRLMSIS, JB08 and HASDM. These figures come from different altitudes and locations and are not one number. This underestimate is what the Phase 3 density-correction tool is meant to address. The spacecraft and altitudes here remain ILLUSTRATIVE, and the 3 Feb 2022 SATCAT listing shows only 17 payloads, so it is not used as a loss count.

References: [Lin et al. 2022](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022SW003254), [He et al. 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023SW003521), [Berger et al. 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022SW003330).

## Phase 1 (done): the physics verifier
- `atmosphere.py`: reference density model plus storm multiplier and NOAA G-scale
- `physics.py`: drag force, orbit decay under drag/thrust, raise feasibility
- `verifier.py`: VPR step checks (storm class, decay prediction, feasibility, action)
- `scenarios.py`: Feb 2022 Starlink storm replay (illustrative spacecraft params)

Run: `python demo.py`

## Roadmap
See docs/ROADMAP.md for the current plan (quantum-tool agent, multi-turn RL environment).

## Original roadmap notes
1. Swap the reference atmosphere for `pymsis` (NRLMSIS 2.0) driven by real F10.7/Ap
2. Add the pre-launch decision (deploy altitude / launch delay), the decision that
   could actually have changed the Feb 2022 outcome
3. More storms: May 2024 (G5), GEO charging cases, Artemis crew radiation
4. LLM policy: GVPO training (github.com/jszkc/GVPO) with this verifier as reward
5. QUBO fleet scheduler (Qiskit Optimization / D-Wave Ocean) vs OR-Tools
6. Phase 2 module: post-storm anomaly attribution ("was it the Sun?")

Note: spacecraft parameters are illustrative, not operator specifications.
