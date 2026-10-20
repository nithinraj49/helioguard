"""qaoa_schedule: fleet QUBO solved by PennyLane QAOA (simulator) next to OR-Tools.

No quantum advantage is claimed or tested for. Spacecraft values are ILLUSTRATIVE.
"""
import random

import numpy as np
import pytest

from helioguard.scenarios import FEB_2022, STARLINK_LIKE as SC
from helioguard.tools.qaoa_schedule import (ACTION_NAMES, QaoaSchedule, action_rewards, build_qubo,
                                            qubo_energy, solve_exact, solve_ortools, solve_qaoa)
from helioguard.verifier import best_action
from verispace_engine.schema import ToolCall
from verispace_engine.tools import Tool


def random_instance(rng, n_assets):
    rewards = [{a: rng.uniform(-0.2, 1.6) for a in ACTION_NAMES} for _ in range(n_assets)]
    windows = [rng.randint(0, 1) for _ in range(n_assets)]
    return rewards, windows


# --- rewards come from physics -------------------------------------------------
@pytest.mark.parametrize("alt", [200.0, 210.0, 230.0, 260.0, 300.0])
def test_best_reward_matches_the_verifiers_best_action(alt):
    scen = {**FEB_2022, "alt_km": alt}
    r = action_rewards(SC, scen)
    assert max(r, key=r.get) == best_action(SC, scen)


def test_infeasible_raise_is_never_attractive():
    r = action_rewards(SC, {**FEB_2022, "alt_km": 210.0})       # raising infeasible at 210 km
    assert r["raise_orbit"] < 0 < max(r.values())


# --- QUBO ---------------------------------------------------------------------
def test_qubo_minimum_equals_the_constrained_optimum_found_by_ortools():
    rng = random.Random(7)
    for _ in range(25):
        n = rng.randint(2, 5)
        rewards, windows = random_instance(rng, n)
        qubo = build_qubo(rewards, windows)
        exact = solve_exact(qubo)
        ort = solve_ortools(rewards, windows)
        assert exact["objective"] == pytest.approx(ort["objective"], abs=1e-3)
        assert exact["feasible"]


def test_penalties_make_constraint_violations_worse_than_the_best_feasible_plan():
    rewards = [{"raise_orbit": 1.5, "safe_mode_edge_on": 1.0, "hold": 0.1},
               {"raise_orbit": 1.5, "safe_mode_edge_on": 1.0, "hold": 0.1}]
    qubo = build_qubo(rewards, windows=[0, 0])
    best = solve_exact(qubo)
    x_double = np.zeros(6)
    x_double[[0, 1]] = 1                       # asset 0 takes two actions
    x_none = np.zeros(6)                       # nobody does anything
    x_clash = np.zeros(6)
    x_clash[[0, 3]] = 1                        # both raise in the same ground window
    for x in (x_double, x_none, x_clash):
        assert qubo_energy(qubo, x) > best["energy"]


def test_two_assets_sharing_a_window_cannot_both_raise():
    rewards = [{"raise_orbit": 1.5, "safe_mode_edge_on": 1.0, "hold": 0.1}] * 2
    assert sum(a == "raise_orbit" for a in solve_exact(build_qubo(rewards, [0, 0]))["assignment"]) == 1
    assert sum(a == "raise_orbit" for a in solve_exact(build_qubo(rewards, [0, 1]))["assignment"]) == 2


# --- QAOA ----------------------------------------------------------------------
def test_qaoa_improves_on_a_random_guess_and_reports_honestly():
    rewards = [{"raise_orbit": 1.5, "safe_mode_edge_on": 1.0, "hold": 0.1},
               {"raise_orbit": 1.4, "safe_mode_edge_on": 0.9, "hold": 0.1}]
    qubo = build_qubo(rewards, windows=[0, 0])
    res = solve_qaoa(qubo, layers=2, seed=0)
    uniform_mean = np.mean([qubo_energy(qubo, np.array(b)) for b in np.ndindex(*(2,) * 6)])
    assert res["expected_energy"] < uniform_mean
    assert res["n_qubits"] == 6 and res["layers"] == 2
    assert res["optimum_probability"] > 1 / 2 ** 6               # beats uniform sampling
    assert 0 < res["approx_ratio"] <= 1.0 + 1e-9
    assert res["simulator"]


def test_qaoa_is_deterministic_for_a_seed():
    rewards = [{"raise_orbit": 1.5, "safe_mode_edge_on": 1.0, "hold": 0.1}] * 2
    qubo = build_qubo(rewards, [0, 0])
    a, b = solve_qaoa(qubo, layers=1, seed=3), solve_qaoa(qubo, layers=1, seed=3)
    assert a["assignment"] == b["assignment"]
    assert a["expected_energy"] == pytest.approx(b["expected_energy"])


# --- the tool ------------------------------------------------------------------
def make_tool():
    return QaoaSchedule(SC, FEB_2022)


def run(tool, **args):
    from verispace_engine.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register(tool)
    return reg.call(ToolCall(name="qaoa_schedule", arguments=args))


def test_tool_protocol():
    assert isinstance(make_tool(), Tool) and make_tool().name == "qaoa_schedule"


def test_tool_always_returns_the_ortools_solution_next_to_qaoa():
    res = run(make_tool(), assets=[{"id": "b1", "alt_km": 260, "window": 0},
                                   {"id": "b2", "alt_km": 260, "window": 0},
                                   {"id": "b3", "alt_km": 210, "window": 1}])
    assert res.ok, res.error
    out = res.output
    assert set(out["ortools"]["assignment"]) == {"b1", "b2", "b3"}
    assert set(out["qaoa"]["assignment"]) == {"b1", "b2", "b3"}
    assert out["exact"]["objective"] == pytest.approx(out["ortools"]["objective"], abs=1e-3)
    assert out["qaoa"]["n_qubits"] == 9
    assert "no quantum advantage" in out["note"].lower()
    for key in ("solve_time_s", "objective"):
        assert key in out["qaoa"] and key in out["ortools"]
    assert "feasible" in out["qaoa"]


def test_ortools_plan_respects_physics_and_ground_windows():
    out = run(make_tool(), assets=[{"id": "b1", "alt_km": 260, "window": 0},
                                   {"id": "b2", "alt_km": 260, "window": 0},
                                   {"id": "b3", "alt_km": 210, "window": 1}]).output
    plan = out["ortools"]["assignment"]
    assert [plan["b1"], plan["b2"]].count("raise_orbit") == 1       # one ground pass
    assert plan["b3"] != "raise_orbit"                              # infeasible at 210 km


def test_tool_rejects_bad_fleets():
    t = make_tool()
    assert not run(t, assets=[]).ok
    assert not run(t, assets=[{"id": "a", "alt_km": 260, "window": 0}] * 2).ok      # duplicate ids
    assert not run(t, assets=[{"id": "a", "alt_km": -5, "window": 0}]).ok
    assert not run(t, assets=[{"id": "a"}]).ok
    too_many = [{"id": f"s{i}", "alt_km": 260, "window": i % 2} for i in range(6)]
    assert not run(t, assets=too_many).ok                           # 18 qubits > limit
    assert not run(t, assets=[{"id": "a", "alt_km": 260, "window": 0}], layers=0).ok
