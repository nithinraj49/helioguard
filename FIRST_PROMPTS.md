# Claude Code prompts, one per session

## Phase 0: setup
Read CLAUDE.md, docs/PROJECT.md and docs/ROADMAP.md, then the code in helioguard/.
Create pyproject.toml, a virtual environment and pytest. Write tests that lock the current
results (Feb 2022: raise_orbit infeasible at 210 km; edge-on safe mode lost near hour 35;
naive trace 0/4; physics-aware trace 4/4). Run them. Do not change any physics.

## Phase 1: engine core
Create the package verispace_engine/ with: a Tool protocol and registry (name, JSON schema,
run()), a pydantic schema for decision traces and tool calls, a VerifierBase with
score_step()/score_episode(), and an EnvBase with reset()/step(). Refactor helioguard's verifier
onto VerifierBase without changing results. Keep the engine free of HelioGuard-specific code.

## Phase 2: real physics and data
Add data/noaa.py and data/celestrak.py (download with caching; offline fixtures for tests).
Replace the reference atmosphere with pymsis driven by real F10.7 and Ap, keeping the old model
as a fallback. Replay 3-5 Feb 2022 and 10-12 May 2024. Tests first.

## Phase 3: tools
Implement the five tools in docs/PROJECT.md section 8.12 on the engine Tool protocol.
qaoa_schedule: build the QUBO, solve small cases with PennyLane QAOA and ALWAYS return the
OR-Tools solution too. qnn_density_correction: hybrid QNN with qml.qnn.TorchLayer
(AngleEmbedding + StronglyEntanglingLayers), trained on decay-derived density ratios, compared
with a same-size classical network. Only enable it after it beats the uncorrected model on
held-out storms.

## Phase 4: multi-turn RL environment
Build helioguard/env.py on EnvBase: reset() returns a scenario prompt; step(action) executes a
tool call and returns the result, or scores a final decision. Rewards per step from the
verifier (correct tool, correct use, physically sound decision), small cost per tool call,
fixed held-out sets, and loophole tests (vague predictions, always-raise, never-call-tools).

## Phase 5: harness
Prompt templates, strict parser for tool calls and decisions, group rollouts (8 per scenario),
JSONL logging, and an evaluation command comparing a rule-based policy and base Qwen.

## Phase 6: GVPO training
Integrate the GVPO reference code (github.com/jszkc/GVPO) with TRL or verl; Qwen 1.5B with LoRA;
curriculum quiet -> severe storms. Report held-out results and two ablations: without quantum
tools, and with outcome-only rewards instead of step rewards.

## Phase 7 to 9
Follow docs/ROADMAP.md: quantum benchmarks (including the quantum RL sub-policy experiment),
the product app built from docs/prototype/helioguard_console.html, then release material.
