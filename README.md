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

## Phase 2 (done): real physics and data
- `helioguard/data/noaa.py`: Kp, Ap and F10.7 from the CelesTrak SW-All file (NOAA/GFZ data), with caching (`data/cache.py`)
- `helioguard/data/celestrak.py`: GP elements, TLE history through sgp4, and SATCAT decay records. CelesTrak serves current elements only; real orbit histories need Space-Track, so the TLE reader takes text from any source
- `helioguard/atmosphere_msis.py`: NRLMSIS density (pymsis) driven by the real indices, orbit-averaged over a small lat/lon grid (illustrative). Falls back to the old reference atmosphere and says which one it used
- `helioguard/replays.py`, `demo_replays.py`: 3 to 5 Feb 2022 (210 km) and 10 to 12 May 2024 (350 km, illustrative). Offline fixtures in `tests/fixtures/`; no test touches the network
- `physics.py` takes an optional `density_fn`; with none, results are identical to Phase 1 (the Phase 0 tests still pass)

| Real-index replay, 48 h (illustrative spacecraft) | Raise orbit | Edge-on | Hold |
|---|---|---|---|
| Feb 2022 at 210 km (G1, Kp 5.3) | survives, 259 km | survives, 182 km | lost at hour 24.3 |
| May 2024 at 350 km (G5, Kp 9.0) | survives, 428 km | survives, 348 km | survives, 343 km |

**Finding.** With real indices, NRLMSIS gives only a small density rise for the Feb 2022 storm (orbit-averaged peak about x1.02 of the start value at 210 km), far below the "up to 50 percent" SpaceX reported. So the Phase 1 result (raise orbit infeasible, every action lost) came from the illustrative x1.5 multiplier, not from the model. This is the gap the Phase 3 density-correction work targets. Nothing was tuned to recover the Phase 1 outcome. May 2024 peaks at about x1.32 at 350 km. The 3 Feb 2022 SATCAT listing shows only 17 payloads, so it is not used as a loss count.

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
