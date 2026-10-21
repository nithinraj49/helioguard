"""The five tools as the agent sees them inside an episode: fleet-aware, on the engine Tool protocol.

Each tool reuses the Phase 3 physics, QAOA and QNN code, applied to the *forecast* storm and to
the facts in the briefing (thruster failures, ground outages). Spacecraft values are ILLUSTRATIVE.
HelioGuard recommends; humans decide.
"""
from __future__ import annotations

from typing import Any

from .atmosphere import density as reference_density
from .physics import drag_force
from .scenarios import STARLINK_LIKE
from .storms import Storm, StormProfile
from .tasks import (ACTION_NAMES, HORIZON_H, RAISE, _scen, action_rewards, density_fn_for,
                    spacecraft_for)
from .tools.physics_tools import CheckFeasibility, SimulateDecay
from .tools.qaoa_schedule import NOTE, build_qubo, solve_exact, solve_ortools, solve_qaoa
from verispace_engine.tools import ToolRegistry

ACTION_ENUM = {"type": "string", "enum": list(ACTION_NAMES)}
MAX_SIM_HOURS = 96
TOOL_COSTS = {"qaoa_schedule": 0.10}
DEFAULT_TOOL_COST = 0.02


class FleetToolbox:
    """Builds the tool registry for one task and answers physics questions about it."""

    def __init__(self, task: dict) -> None:
        self.task = task
        self.assets = {a["id"]: a for a in task["fleet"]}
        self.fn = density_fn_for(task, 0)                       # forecast storm
        self.registry = ToolRegistry()
        for tool in (_SimulateDecay(self), _CheckFeasibility(self), _MinSafeAltitude(self),
                     _QaoaSchedule(self), _QnnDensity(self)):
            self.registry.register(tool)

    # -- helpers shared by the tools and the verifier ---------------------------------------
    def asset(self, asset_id: str) -> dict:
        if asset_id not in self.assets:
            raise ValueError(f"unknown asset '{asset_id}'; assets are {sorted(self.assets)}")
        return self.assets[asset_id]

    def spacecraft(self, asset_id: str):
        return spacecraft_for(self.asset(asset_id)["thrust_fraction"])

    def scen(self, asset_id: str) -> dict:
        return _scen(self.asset(asset_id)["alt_km"], self.fn)

    def peak_drag_mN(self, asset_id: str, fn=None) -> float:
        """Largest raise-attitude drag over the horizon at the asset's current altitude."""
        a, sc = self.asset(asset_id), STARLINK_LIKE
        fn = fn or self.fn
        return max(drag_force(a["alt_km"], sc.area_m2["raise"], sc.cd, 1.0, fn, float(h))
                   for h in range(0, HORIZON_H + 1)) * 1e3

    def feasibility(self, asset_id: str, action: str) -> dict[str, Any]:
        a = self.asset(asset_id)
        base = CheckFeasibility(self.spacecraft(asset_id), self.scen(asset_id)).run(action)
        through = True
        peak = base["drag_mN"]
        if action == RAISE:
            peak = self.peak_drag_mN(asset_id)
            through = base["thrust_mN"] > peak
        return {"asset_id": asset_id, "action": action, "feasible_now": base["feasible"],
                "feasible_through_storm": bool(through), "drag_mN": base["drag_mN"],
                "peak_drag_mN": peak, "thrust_mN": base["thrust_mN"], "ground_window": a["window"],
                "ground_window_available": a["window"] not in self.task["windows"]["outage"]}

    def decay(self, asset_id: str, action: str, hours: float) -> dict[str, Any]:
        out = SimulateDecay(self.spacecraft(asset_id), self.scen(asset_id)).run(action, hours)
        out["asset_id"] = asset_id
        out["altitude_loss_km"] = self.asset(asset_id)["alt_km"] - out["final_alt_km"]
        return out


class _Base:
    def __init__(self, box: FleetToolbox) -> None:
        self.box = box


class _SimulateDecay(_Base):
    name = "simulate_decay"
    description = ("Altitude over time for one asset under one action (raise_orbit, safe_mode_edge_on "
                   "or hold) in the forecast storm. Thruster failures are included.")
    schema = {"type": "object",
              "properties": {"asset_id": {"type": "string"}, "action": ACTION_ENUM,
                             "hours": {"type": "number"}},
              "required": ["asset_id", "action", "hours"]}

    def run(self, asset_id: str, action: str, hours: float) -> dict[str, Any]:
        if not 0 < hours <= MAX_SIM_HOURS:
            raise ValueError(f"hours must be in (0, {MAX_SIM_HOURS}]")
        return self.box.decay(asset_id, action, hours)


class _CheckFeasibility(_Base):
    name = "check_feasibility"
    description = ("Whether an action is possible for one asset: thrust versus drag in mN, now and "
                   "at the forecast storm peak, and whether its ground window is available.")
    schema = {"type": "object",
              "properties": {"asset_id": {"type": "string"}, "action": ACTION_ENUM},
              "required": ["asset_id", "action"]}

    def run(self, asset_id: str, action: str) -> dict[str, Any]:
        return self.box.feasibility(asset_id, action)


class _MinSafeAltitude(_Base):
    name = "min_safe_altitude"
    description = ("Lowest altitude (km) at which raising keeps thrust above drag throughout the "
                   "forecast horizon. Optional asset_id uses that asset's thrust; optional "
                   "peak_multiplier asks a what-if with a different storm peak.")
    schema = {"type": "object",
              "properties": {"asset_id": {"type": "string"}, "peak_multiplier": {"type": "number"}},
              "required": []}

    def run(self, asset_id: str | None = None, peak_multiplier: float | None = None) -> dict[str, Any]:
        box, task = self.box, self.box.task
        thrust = (box.spacecraft(asset_id) if asset_id else STARLINK_LIKE).thrust_n
        fn = box.fn
        if peak_multiplier is not None:
            if peak_multiplier < 1:
                raise ValueError("peak_multiplier must be at least 1")
            profile = StormProfile(Storm.from_dict(task["storm"]), task["forecast"]["arrival_h"],
                                   peak_multiplier, 1.0)
            fn = lambda alt, t: reference_density(alt, profile(t))
        sc = STARLINK_LIKE

        def ok(alt: float) -> bool:
            return thrust > max(drag_force(alt, sc.area_m2["raise"], sc.cd, 1.0, fn, float(h))
                                for h in range(0, HORIZON_H + 1))

        lo, hi = 150.0, 800.0
        result = None
        if ok(hi):
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            result = hi
        return {"min_alt_km": result, "asset_id": asset_id,
                "peak_multiplier": peak_multiplier if peak_multiplier is not None
                else task["forecast"]["peak_multiplier"],
                "basis": "thrust exceeds drag at every hour of the forecast horizon"}


class _QaoaSchedule(_Base):
    name = "qaoa_schedule"
    description = ("Best action for every asset in the forecast storm (one action per asset, at most "
                   "one raise per ground window, no raise in a window that is out), solved by QAOA on "
                   "a simulator and always returned next to the OR-Tools and exact solutions. "
                   "Slow and costly: use only when assets compete for ground windows. It does not "
                   "cover forecast error or the launch.")
    schema = {"type": "object", "properties": {"layers": {"type": "integer"}}, "required": []}

    def run(self, layers: int = 2) -> dict[str, Any]:
        box, task = self.box, self.box.task
        if not 1 <= layers <= 4:
            raise ValueError("layers must be between 1 and 4")
        if len(task["fleet"]) * len(ACTION_NAMES) > 12:
            raise ValueError("fleet too large for the simulator (at most 4 assets)")
        ids = [a["id"] for a in task["fleet"]]
        rewards = []
        for a in task["fleet"]:
            r = action_rewards(box.spacecraft(a["id"]), box.scen(a["id"]))
            if a["window"] in task["windows"]["outage"]:
                r = {**r, RAISE: -1.0}
            rewards.append(r)
        windows = [a["window"] for a in task["fleet"]]
        qubo = build_qubo(rewards, windows)
        out = {"qaoa": solve_qaoa(qubo, layers=layers), "ortools": solve_ortools(rewards, windows),
               "exact": solve_exact(qubo)}
        for key, res in out.items():
            out[key] = {**{k: v for k, v in res.items() if k not in ("assignment", "x")},
                        "assignment": dict(zip(ids, res["assignment"]))}
        out["note"] = NOTE
        out["basis"] = "forecast storm only"
        return out


class _QnnDensity(_Base):
    name = "qnn_density_correction"
    description = ("NRLMSIS-style density for the given conditions, corrected by a hybrid quantum "
                   "neural network. The correction is disabled until it beats uncorrected density on "
                   "held-out storms, so this returns the plain density with the reason.")

    def __init__(self, box: FleetToolbox) -> None:
        super().__init__(box)
        from .tools.qnn_density import FEATURE_NAMES, DensityCorrector, QnnDensityCorrection
        arrival = box.task["forecast"]["arrival_h"]
        self._tool = QnnDensityCorrection(DensityCorrector(None, enabled=False),
                                          lambda alt, hours_since_onset: box.fn(alt, arrival + hours_since_onset))
        self.schema = self._tool.schema

    def run(self, **features: float) -> dict[str, Any]:
        return self._tool.run(**features)
