"""Physics tools: simulate_decay, check_feasibility, min_safe_altitude.

Each tool is built for one spacecraft and one scenario (the situation the agent is shown).
Spacecraft values are ILLUSTRATIVE. These tools recommend; they never command a spacecraft.
"""
from __future__ import annotations

from typing import Any

from ..atmosphere import density as reference_density
from ..physics import Spacecraft, can_raise, drag_force, simulate
from ..verifier import ACTIONS, feasible

_ACTION_ENUM = list(ACTIONS)
MAX_HOURS = 24 * 14
_SEARCH_LO_KM, _SEARCH_HI_KM = 150.0, 800.0


class _ScenarioTool:
    def __init__(self, sc: Spacecraft, scen: dict) -> None:
        self.sc, self.scen = sc, scen


class SimulateDecay(_ScenarioTool):
    name = "simulate_decay"
    description = ("Altitude over time for the scenario's satellite under one action "
                   "(raise_orbit, safe_mode_edge_on or hold), with the storm in the scenario.")
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": _ACTION_ENUM},
                       "hours": {"type": "number"}},
        "required": ["action", "hours"],
    }

    def run(self, action: str, hours: float) -> dict[str, Any]:
        if not 0 < hours <= MAX_HOURS:
            raise ValueError(f"hours must be in (0, {MAX_HOURS}]")
        attitude, thrusting = ACTIONS[action]
        out = simulate(self.sc, self.scen["alt_km"], attitude, self.scen["storm_multiplier"],
                       hours, thrusting=thrusting, density_fn=self.scen.get("density_fn"),
                       sample_every_s=3600.0)
        return {"action": action, **out}


class CheckFeasibility(_ScenarioTool):
    name = "check_feasibility"
    description = ("Whether an action is physically possible now. For raise_orbit it returns "
                   "drag and thrust in mN: raising needs thrust greater than drag.")
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": _ACTION_ENUM}},
        "required": ["action"],
    }

    def run(self, action: str) -> dict[str, Any]:
        attitude, _ = ACTIONS[action]
        drag = drag_force(self.scen["alt_km"], self.sc.area_m2[attitude], self.sc.cd,
                          self.scen["storm_multiplier"], self.scen.get("density_fn"))
        return {"action": action, "feasible": feasible(self.sc, self.scen, action),
                "drag_mN": drag * 1e3, "thrust_mN": self.sc.thrust_n * 1e3,
                "attitude": attitude}


class MinSafeAltitude(_ScenarioTool):
    name = "min_safe_altitude"
    description = ("Lowest altitude (km) at which raising the orbit works, i.e. thrust exceeds "
                   "drag. Optional storm_multiplier asks a what-if with the reference atmosphere "
                   "scaled by that factor; without it the scenario's own atmosphere is used.")
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"storm_multiplier": {"type": "number"}},
        "required": [],
    }

    def run(self, storm_multiplier: float | None = None) -> dict[str, Any]:
        if storm_multiplier is not None:
            if storm_multiplier <= 0:
                raise ValueError("storm_multiplier must be positive")
            fn, mult = (lambda alt, t: reference_density(alt, storm_multiplier)), storm_multiplier
        else:
            fn, mult = self.scen.get("density_fn"), self.scen["storm_multiplier"]

        def ok(alt: float) -> bool:
            return can_raise(self.sc, alt, "raise", mult, fn)

        result: float | None
        if not ok(_SEARCH_HI_KM):
            result = None
        elif ok(_SEARCH_LO_KM):
            result = _SEARCH_LO_KM
        else:                                   # drag falls with altitude, so bisect
            lo, hi = _SEARCH_LO_KM, _SEARCH_HI_KM
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            result = hi
        alt0 = self.scen["alt_km"]
        return {"min_alt_km": result, "scenario_alt_km": alt0,
                "scenario_alt_feasible": ok(alt0),
                "storm_multiplier": storm_multiplier if storm_multiplier is not None else mult}
