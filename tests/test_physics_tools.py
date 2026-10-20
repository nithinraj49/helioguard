"""Phase 3 physics tools on the engine Tool protocol. Spacecraft values are ILLUSTRATIVE."""
import pytest

from helioguard.physics import simulate
from helioguard.scenarios import FEB_2022, FEB_2022_QUIET, STARLINK_LIKE as SC
from helioguard.tools.physics_tools import CheckFeasibility, MinSafeAltitude, SimulateDecay
from verispace_engine.schema import ToolCall
from verispace_engine.tools import Tool, ToolRegistry


def registry(scen=FEB_2022):
    reg = ToolRegistry()
    for tool in (SimulateDecay(SC, scen), CheckFeasibility(SC, scen), MinSafeAltitude(SC, scen)):
        reg.register(tool)
    return reg


def call(reg, name, **args):
    return reg.call(ToolCall(name=name, arguments=args))


def test_tools_satisfy_the_engine_protocol_and_register():
    reg = registry()
    assert reg.names() == ["simulate_decay", "check_feasibility", "min_safe_altitude"]
    assert all(isinstance(t, Tool) for t in (SimulateDecay(SC, FEB_2022),))
    assert all(s["description"] and s["parameters"]["type"] == "object" for s in reg.specs())


# --- simulate_decay ----------------------------------------------------------
def test_simulate_decay_matches_the_phase1_physics():
    res = call(registry(), "simulate_decay", action="safe_mode_edge_on", hours=48)
    assert res.ok
    out = res.output
    assert out["reentered"] and out["reentry_hour"] == pytest.approx(35.3, abs=0.2)
    ref = simulate(SC, 210.0, "edge_on", 1.5, 48)
    assert out["reentry_hour"] == ref["reentry_hour"]


def test_simulate_decay_returns_an_hourly_altitude_series():
    out = call(registry(FEB_2022_QUIET), "simulate_decay", action="raise_orbit", hours=24).output
    assert out["times_h"][0] == 0 and out["altitudes_km"][0] == pytest.approx(210.0)
    assert len(out["times_h"]) == len(out["altitudes_km"]) == 25
    assert out["final_alt_km"] == pytest.approx(out["altitudes_km"][-1], abs=0.5)


def test_simulate_decay_series_stops_at_reentry():
    out = call(registry(), "simulate_decay", action="hold", hours=48).output
    assert out["reentered"] and out["altitudes_km"][-1] < 160
    assert len(out["altitudes_km"]) <= int(out["reentry_hour"]) + 2


def test_simulate_decay_rejects_bad_arguments():
    reg = registry()
    assert not call(reg, "simulate_decay", action="teleport", hours=24).ok     # enum
    assert not call(reg, "simulate_decay", action="hold").ok                   # missing hours
    assert not call(reg, "simulate_decay", action="hold", hours=-5).ok
    assert not call(reg, "simulate_decay", action="hold", hours=100000).ok     # runaway horizon


# --- check_feasibility -------------------------------------------------------
def test_feasibility_reports_both_forces_for_the_feb_storm():
    out = call(registry(), "check_feasibility", action="raise_orbit").output
    assert out["feasible"] is False
    assert out["drag_mN"] == pytest.approx(85.1, abs=0.1)
    assert out["thrust_mN"] == pytest.approx(70.0)


def test_feasibility_quiet_orbit_allows_raising():
    out = call(registry(FEB_2022_QUIET), "check_feasibility", action="raise_orbit").output
    assert out["feasible"] is True and out["drag_mN"] == pytest.approx(56.7, abs=0.1)


def test_feasibility_of_passive_actions_is_true_and_unknown_is_rejected():
    reg = registry()
    assert call(reg, "check_feasibility", action="safe_mode_edge_on").output["feasible"] is True
    assert not call(reg, "check_feasibility", action="warp_drive").ok


# --- min_safe_altitude -------------------------------------------------------
@pytest.mark.parametrize("mult,expected", [(1.5, 218), (2.0, 228), (3.0, 243)])
def test_min_safe_altitude_matches_the_project_document(mult, expected):
    out = call(registry(), "min_safe_altitude", storm_multiplier=mult).output
    assert out["min_alt_km"] == pytest.approx(expected, abs=1.0)


def test_min_safe_altitude_is_exactly_the_feasibility_boundary():
    out = call(registry(), "min_safe_altitude", storm_multiplier=1.5).output
    from helioguard.physics import can_raise
    assert can_raise(SC, out["min_alt_km"] + 0.2, "raise", 1.5)
    assert not can_raise(SC, out["min_alt_km"] - 0.2, "raise", 1.5)


def test_min_safe_altitude_defaults_to_the_scenarios_own_storm():
    out = call(registry(), "min_safe_altitude").output
    assert out["min_alt_km"] == pytest.approx(217.2, abs=0.5)     # FEB_2022 is x1.5
    assert out["scenario_alt_km"] == 210.0 and out["scenario_alt_feasible"] is False


def test_min_safe_altitude_reports_when_raising_is_never_possible():
    out = call(registry(), "min_safe_altitude", storm_multiplier=1e6).output
    assert out["min_alt_km"] is None


def test_min_safe_altitude_rejects_nonsense_multiplier():
    assert not call(registry(), "min_safe_altitude", storm_multiplier=0).ok
    assert not call(registry(), "min_safe_altitude", storm_multiplier=-2).ok
