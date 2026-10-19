# HelioGuard roadmap (Claude Code)

Build order: HelioGuard first (it creates the shared `verispace_engine`), then the Navigator.
One session per phase. Paste the prompt, review the tests, run them, commit.

| Phase | Deliverable | Done when |
|---|---|---|
| 0 Setup | pyproject, venv, pytest; tests locking Phase 1 results | Tests pass, physics unchanged |
| 1 Engine core | `verispace_engine`: tool registry, trace + tool-call schema, verifier base, env base | Any tool or rule plugs in via one interface |
| 2 Real physics + data | pymsis density from real F10.7/Ap, sgp4/Orekit, CelesTrak history | Feb 2022 and May 2024 replays use real data |
| 3 Tools | simulate_decay, check_feasibility, min_safe_altitude, qaoa_schedule, qnn_density_correction | Each tested; QAOA vs OR-Tools, QNN vs same-size MLP |
| 4 Multi-turn RL env | scenario generator, tool execution, step verifier, tool-use cost, held-out sets, loophole tests | Thousands of scenarios; wrong answers fail every rule |
| 5 Harness | prompts, strict parser, group rollouts, logging, eval vs rule baseline and base Qwen | One command runs a full evaluation |
| 6 GVPO training | Qwen 1.5B to 4B + LoRA, curriculum, ablations (no quantum tools; outcome-only reward) | Trained agent beats base model and rules on held-out storms |
| 7 Quantum benchmarks | larger QAOA, QNN on real decay, quantum RL sub-policy experiment | Benchmark tables published |
| 8 Product app | FastAPI backend, live NOAA/CelesTrak, console UI from prototype, deployment guardrail | Operator loads a fleet, gets verified recommendations |
| 9 Release | README, notebooks, preprint draft, LinkedIn | Public repo and first post |

## Compute notes
- Phases 0 to 5 and 7 run on a laptop (QAOA and QNN on PennyLane simulators, `lightning.qubit`).
- Phase 6 needs a GPU. Use LoRA (Unsloth) and a fast rollout engine (vLLM). Start with a 1.5B model.
- Optional later: run QAOA on real quantum hardware through a PennyLane device plugin.

## Session prompts
See `FIRST_PROMPTS.md` in the repository root.
