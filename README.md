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

## Phase 3 (done, with one open item): tools
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
