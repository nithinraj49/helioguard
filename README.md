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

**QNN density correction: not yet shown to help.** The success criterion is that the corrected density reproduces the observed decay of Starlink Group 4-7 better than uncorrected NRLMSIS, with Feb 2022 held out and the model trained on other storms. That needs real TLE histories from Space-Track and I have no credentials here, so it has **not been run**. What exists and is tested on synthetic histories only: the decay-to-density pipeline (`decay_density.py`), the held-out comparison against the same-size MLP, the two gates (held-out RMSE margin and decay-prediction error margin, both at least 5 percent better), and `density_pipeline.py`, which refuses to train on the held-out event. The tool therefore returns uncorrected NRLMSIS density, with the reason, by default. To run it: put credentials in `.env`, copy `configs/qnn_events.example.json`, add training storms and satellites, and run `python train_qnn_density.py <config>`. Caveats for that run: the Group 4-7 satellites have no quiet period before the storm, so their ballistic coefficient (Cd*A/m) must be supplied and justified; and TLE mean motion is treated as circular-orbit mean motion, an approximation.

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
