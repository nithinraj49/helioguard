"""Phase 6 stays on the legacy penalty: the fleet QUBO of every train and eval task is bit-identical to the
pre-change one (golden digests, built with the original build_qubo from git 9b4b555^).

Needs the project venv (torch is imported by the toolbox): .venv\Scripts\python -m pytest tests/test_phase6_qubo_unchanged.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
from helioguard.env_tools import FleetToolbox, _QaoaSchedule                            # noqa: E402
from verispace_engine import qubo as engine                                 # noqa: E402
from verispace_engine.qubo import solve_exact, solve_ortools                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "fixtures" / "phase6_qubo_golden.json").read_text(encoding="utf-8"))


def tasks():
    for split in ("train", "eval"):
        for line in (ROOT / "tasks" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            yield json.loads(line)


def test_every_train_and_eval_task_builds_the_same_qubo_as_before():
    n = 0
    for t in tasks():
        box = FleetToolbox(t)
        _, _, q = _QaoaSchedule(box).fleet_qubo()
        got = hashlib.sha256(q.Q.tobytes() + repr(float(q.const)).encode()).hexdigest()
        assert got == GOLDEN["digests"][t["id"]], t["id"]
        n += 1
    assert n == len(GOLDEN["digests"]) == 1050


@pytest.mark.parametrize("case", GOLDEN["insufficient_legacy"], ids=lambda c: c["id"])
def test_sufficient_rule_gives_a_feasible_ortools_matching_optimum_on_the_insufficient_cases(case):
    acts = ["raise_orbit", "safe_mode_edge_on", "hold"]
    rw, w = case["rewards"], case["windows"]
    good = engine.build_qubo(rw, w, acts, "raise_orbit", "sufficient")
    allr = [r[a] for r in rw for a in acts]
    assert (max(allr) - min(allr)) + 1 <= max(max(allr), -min(allr))          # legacy bound really is violated here
    ex = solve_exact(good)
    assert ex["feasible"]
    assert ex["objective"] == pytest.approx(solve_ortools(rw, w, acts, "raise_orbit")["objective"], abs=0.5e-3 * len(rw))
