# CLAUDE.md: HelioGuard (Earth-orbit storm decisions)

HelioGuard turns solar-storm forecasts into physics-verified decisions for satellite fleets.
It is a quantum-augmented agent: Qwen reasons and calls tools (physics, PennyLane QAOA,
PennyLane hybrid QNN) inside a multi-turn RL environment, trained with GVPO on
Verifiable Process Rewards (VPR).

Read first, every session: `docs/PROJECT.md` (design), `docs/ROADMAP.md` (phases).
UI reference: `docs/prototype/helioguard_console.html` (open in a browser).

## Current state
- Phase 1 physics verifier exists: `helioguard/atmosphere.py`, `physics.py`, `verifier.py`, `scenarios.py`.
- `python demo.py` replays the Feb 2022 Starlink storm and scores two traces.

## Rules
1. One phase per session, in ROADMAP order. Never start the next phase unprompted.
2. Tests first (pytest). Each phase starts with tests that encode its "done when" criteria.
3. The verifier is independent physics code. Never weaken a check to make a test or model pass.
   The agent never grades itself.
4. Every verifier rule gets loophole tests: deliberately wrong or vague answers must fail.
5. Every quantum result (QAOA, QNN, quantum RL) is reported next to a classical baseline
   (OR-Tools, same-size classical network). Never claim quantum advantage.
6. Label illustrative values as illustrative in code comments and outputs.
7. HelioGuard recommends; humans decide. Never describe it as controlling spacecraft.
8. Shared code lives in `verispace_engine/` so the Navigator project can reuse it unchanged.
9. Python 3.11+, type hints, small modules. Update README.md with results after each phase.
10. Don't put the tech stack or jargon in user-facing UI text; keep it in docs and "Under the hood".

## Commands
- Demo: `python demo.py`   - Tests: `pytest -q`
