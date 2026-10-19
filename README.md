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
