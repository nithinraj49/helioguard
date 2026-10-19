"""Feb 2022 and May 2024 replays driven by real indices. Spacecraft and altitudes stay ILLUSTRATIVE."""
from pathlib import Path

import pytest

from helioguard.physics import simulate
from helioguard.replays import FEB_2022_REAL, MAY_2024_REAL, build_replay
from helioguard.scenarios import STARLINK_LIKE as SC
from helioguard.verifier import ACTIONS, best_action, verify_step, verify_trace

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def feb():
    return build_replay(FEB_2022_REAL, FIX / "sw_feb2022.csv")


@pytest.fixture(scope="module")
def may():
    return build_replay(MAY_2024_REAL, FIX / "sw_may2024.csv")


def test_replays_use_real_data_and_say_so(feb, may):
    assert feb["data_source"] == "pymsis" and may["data_source"] == "pymsis"
    assert feb["storm_multiplier"] == 1.0           # the density is in density_fn, no fudge factor
    assert feb["kp"] == pytest.approx(5.3) and may["kp"] == pytest.approx(9.0)
    assert feb["illustrative_spacecraft"] and may["illustrative_spacecraft"]


def test_replay_dates_match_the_events(feb, may):
    assert str(feb["start"]).startswith("2022-02-03") and feb["horizon_h"] == 48
    assert str(may["start"]).startswith("2024-05-10") and may["horizon_h"] == 48


def test_may_storm_is_stronger_than_feb(feb, may):
    assert may["kp"] > feb["kp"]


def test_every_action_has_a_physical_outcome(feb, may):
    for scen in (feb, may):
        for act, (att, thr) in ACTIONS.items():
            out = simulate(SC, scen["alt_km"], att, scen["storm_multiplier"], scen["horizon_h"],
                           thrusting=thr, density_fn=scen["density_fn"])
            assert out["reentered"] or out["final_alt_km"] > 0
        assert best_action(SC, scen) in ACTIONS


def test_verifier_runs_on_real_replays_and_catches_loopholes(feb, may):
    for scen in (feb, may):
        good = best_action(SC, scen)
        assert verify_step({"type": "choose_action", "action": good}, SC, scen)[0] == 1
        res = verify_trace([{"type": "classify_storm", "g_scale": "G0"},
                            {"type": "predict_decay", "attitude": "raise", "hours": 12,
                             "altitude_loss_km": 0}], SC, scen)
        assert res["step_rewards"] == [0, 0]          # wrong storm class, "no decay" claim


# --- locked outputs of the real-data replays (illustrative spacecraft) ---------------------
# These record what NRLMSIS + real indices give today. They are NOT tuned to match the event:
# the Phase 1 illustrative x1.5 multiplier is what made raising infeasible in the Feb replay.
def _outcome(scen, action):
    att, thr = ACTIONS[action]
    return simulate(SC, scen["alt_km"], att, 1.0, scen["horizon_h"], thrusting=thr,
                    density_fn=scen["density_fn"])


def test_feb_2022_real_indices_locked(feb):
    from helioguard.physics import drag_force
    assert drag_force(210, SC.area_m2["raise"], SC.cd, 1.0, feb["density_fn"]) * 1e3 == pytest.approx(44.9, abs=0.5)
    assert best_action(SC, feb) == "raise_orbit"
    hold = _outcome(feb, "hold")
    assert hold["reentered"] and hold["reentry_hour"] == pytest.approx(24.3, abs=0.3)
    assert not _outcome(feb, "raise_orbit")["reentered"]
    assert _outcome(feb, "raise_orbit")["final_alt_km"] == pytest.approx(259, abs=2)


def test_may_2024_real_indices_locked(may):
    assert best_action(SC, may) == "raise_orbit"
    assert not any(_outcome(may, a)["reentered"] for a in ACTIONS)
    assert _outcome(may, "hold")["final_alt_km"] == pytest.approx(343, abs=2)
