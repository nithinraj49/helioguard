"""Lock the Phase 1 results (Feb 2022 replay). Spacecraft values are ILLUSTRATIVE.

If one of these fails, the physics changed. Do not loosen a check to make it pass.
"""
import pytest

from helioguard.physics import can_raise, drag_force, simulate
from helioguard.scenarios import FEB_2022, FEB_2022_QUIET, STARLINK_LIKE as SC
from helioguard.verifier import ACTIONS, best_action, feasible, group_centered, verify_trace


def _run(scen, action):
    att, thr = ACTIONS[action]
    return simulate(SC, scen["alt_km"], att, scen["storm_multiplier"], 48, thrusting=thr)


def test_feb_2022_raise_orbit_infeasible_at_210km():
    assert not can_raise(SC, 210.0, "raise", 1.5)
    assert not feasible(SC, FEB_2022, "raise_orbit")
    drag = drag_force(210, SC.area_m2["raise"], SC.cd, 1.5)
    assert drag * 1e3 == pytest.approx(85.1, abs=0.1)
    assert drag > SC.thrust_n  # 70 mN thrust


def test_quiet_orbit_raise_is_feasible():
    assert drag_force(210, SC.area_m2["raise"], SC.cd, 1.0) * 1e3 == pytest.approx(56.7, abs=0.1)
    assert feasible(SC, FEB_2022_QUIET, "raise_orbit")


def test_feb_2022_edge_on_safe_mode_lost_near_hour_35():
    out = _run(FEB_2022, "safe_mode_edge_on")
    assert out["reentered"]
    assert out["reentry_hour"] == pytest.approx(35.3, abs=0.2)


def test_feb_2022_every_action_lost_and_edge_on_is_best():
    hours = {a: _run(FEB_2022, a) for a in ACTIONS}
    assert all(o["reentered"] for o in hours.values())
    assert hours["raise_orbit"]["reentry_hour"] == pytest.approx(31.8, abs=0.2)
    assert hours["hold"]["reentry_hour"] == pytest.approx(13.2, abs=0.2)
    assert best_action(SC, FEB_2022) == "safe_mode_edge_on"


def test_quiet_atmosphere_outcomes():
    raise_ = _run(FEB_2022_QUIET, "raise_orbit")
    edge = _run(FEB_2022_QUIET, "safe_mode_edge_on")
    hold = _run(FEB_2022_QUIET, "hold")
    assert not raise_["reentered"] and raise_["final_alt_km"] == pytest.approx(243, abs=1)
    assert not edge["reentered"] and edge["final_alt_km"] == pytest.approx(166, abs=1)
    assert hold["reentered"] and hold["reentry_hour"] == pytest.approx(19.9, abs=0.2)
    assert best_action(SC, FEB_2022_QUIET) == "raise_orbit"


NAIVE = [
    {"type": "classify_storm", "g_scale": "G3"},
    {"type": "predict_decay", "attitude": "raise", "hours": 24, "altitude_loss_km": 10},
    {"type": "check_feasibility", "action": "raise_orbit", "feasible": True},
    {"type": "choose_action", "action": "raise_orbit"},
]
PHYSICS_AWARE = [
    {"type": "classify_storm", "g_scale": "G1"},
    {"type": "predict_decay", "attitude": "raise", "hours": 12, "altitude_loss_km": 40},
    {"type": "check_feasibility", "action": "raise_orbit", "feasible": False},
    {"type": "choose_action", "action": "safe_mode_edge_on"},
]


def test_naive_trace_scores_0_of_4():
    res = verify_trace(NAIVE, SC, FEB_2022)
    assert res["step_rewards"] == [0, 0, 0, 0]
    assert res["process_reward"] == 0.0


def test_physics_aware_trace_scores_4_of_4():
    res = verify_trace(PHYSICS_AWARE, SC, FEB_2022)
    assert res["step_rewards"] == [1, 1, 1, 1]
    assert res["process_reward"] == 1.0


def test_gvpo_group_centred_rewards():
    totals = [verify_trace(t, SC, FEB_2022)["process_reward"] for t in (NAIVE, PHYSICS_AWARE)]
    assert group_centered(totals) == [-0.5, 0.5]
