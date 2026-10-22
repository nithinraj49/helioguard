"""qaoa_schedule: fleet-wide protective schedule as a QUBO, solved by PennyLane QAOA.

Every result is reported next to the classical solutions: OR-Tools CP-SAT and a brute-force
optimum. QAOA runs on a classical simulator (lightning.qubit); no quantum advantage is claimed.
Spacecraft values are ILLUSTRATIVE. The schedule is a recommendation; humans decide.

QUBO (x[i,a] = 1 if asset i takes action a, action 0 is raise_orbit):
    E(x) = - sum R[i,a] x[i,a]
           + P * sum_i (sum_a x[i,a] - 1)^2                  one action per asset
           + P * sum_{i<j, same window} x[i,raise] x[j,raise]  one raise per ground window
The window term is the capacity-1 case of "load over capacity" without slack variables.
"""
from __future__ import annotations

from typing import Any


from verispace_engine import qaoa as _engine_qaoa
from verispace_engine import qubo as _engine
from verispace_engine.qubo import (Qubo, all_bitstrings, all_energies, decode, qubo_energy, qubo_to_ising,
                                   solve_exact)

from ..physics import Spacecraft, simulate
from ..verifier import ACTIONS, feasible

ACTION_NAMES = list(ACTIONS)            # raise_orbit first: it needs a ground contact
RAISE = 0
MAX_QUBITS = 12
NOTE = ("QAOA was run on a classical simulator and compared with OR-Tools and the exact optimum. "
        "No quantum advantage is claimed. Recommendation only; a human operator decides.")


# --- rewards from physics -------------------------------------------------------------
def action_rewards(sc: Spacecraft, scen: dict) -> dict[str, float]:
    """Higher is better; ordering matches the verifier: survive first, then altitude or time left.
    An infeasible raise (drag above thrust) gets -1 so it is never attractive."""
    rewards: dict[str, float] = {}
    for action, (attitude, thrusting) in ACTIONS.items():
        if not feasible(sc, scen, action):
            rewards[action] = -1.0
            continue
        out = simulate(sc, scen["alt_km"], attitude, scen["storm_multiplier"], scen["horizon_h"],
                       thrusting=thrusting, density_fn=scen.get("density_fn"))
        if out["reentered"]:
            rewards[action] = 0.5 * out["reentry_hour"] / scen["horizon_h"]
        else:
            rewards[action] = 1.0 + (out["final_alt_km"] - 150.0) / 1000.0
    return rewards


# --- QUBO (the generic code lives in verispace_engine.qubo) -----------------------------------
def build_qubo(rewards: list[dict[str, float]], windows: list[int]) -> Qubo:
    """Fleet QUBO: items = assets, groups = ground windows, exclusive action = raise_orbit."""
    return _engine.build_qubo(rewards, windows, ACTION_NAMES, ACTION_NAMES[RAISE])


def solve_ortools(rewards: list[dict[str, float]], windows: list[int]) -> dict[str, Any]:
    return _engine.solve_ortools(rewards, windows, ACTION_NAMES, ACTION_NAMES[RAISE])


# Old private names, kept as thin aliases for one release; import from verispace_engine.qubo instead.
_all_bitstrings = all_bitstrings
_all_energies = all_energies
_ising = qubo_to_ising


# --- QAOA ------------------------------------------------------------------------------
solve_qaoa = _engine_qaoa.solve_qaoa            # moved to verispace_engine.qaoa; old name kept as an alias


# --- the tool --------------------------------------------------------------------------
class QaoaSchedule:
    name = "qaoa_schedule"
    description = ("Best fleet-wide protective schedule: one action per asset, and at most one "
                   "orbit-raise per ground window. Solved with QAOA (simulator) and always "
                   "returned next to the OR-Tools solution and the exact optimum.")
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "assets": {"type": "array",
                       "description": "[{id, alt_km, window}], at most 4 assets"},
            "layers": {"type": "integer", "description": "QAOA depth, 1 to 4 (default 2)"},
        },
        "required": ["assets"],
    }

    def __init__(self, sc: Spacecraft, scen: dict) -> None:
        self.sc, self.scen = sc, scen

    def run(self, assets: list[dict], layers: int = 2) -> dict[str, Any]:
        if not 1 <= layers <= 4:
            raise ValueError("layers must be between 1 and 4")
        if not assets:
            raise ValueError("assets must not be empty")
        if len(assets) * len(ACTION_NAMES) > MAX_QUBITS:
            raise ValueError(f"at most {MAX_QUBITS // len(ACTION_NAMES)} assets "
                             f"({MAX_QUBITS} qubits) on the simulator")
        ids = [a.get("id") for a in assets]
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("each asset needs a unique string id")
        for a in assets:
            if not isinstance(a.get("alt_km"), (int, float)) or a["alt_km"] <= 0:
                raise ValueError("each asset needs a positive alt_km")
            if not isinstance(a.get("window"), int) or isinstance(a["window"], bool) or a["window"] < 0:
                raise ValueError("each asset needs a non-negative integer window")
        rewards = [action_rewards(self.sc, {**self.scen, "alt_km": float(a["alt_km"])})
                   for a in assets]
        windows = [a["window"] for a in assets]
        qubo = build_qubo(rewards, windows)

        def named(res: dict) -> dict:
            plan = dict(zip(ids, res["assignment"]))
            return {**{k: v for k, v in res.items() if k != "assignment"}, "assignment": plan}

        qaoa, ort, exact = solve_qaoa(qubo, layers=layers), solve_ortools(rewards, windows), solve_exact(qubo)
        return {"qaoa": named(qaoa), "ortools": named(ort), "exact": named(exact),
                "qaoa_matches_ortools": bool(qaoa["feasible"] and
                                             abs(qaoa["objective"] - ort["objective"]) < 1e-6),
                "rewards": dict(zip(ids, rewards)), "note": NOTE}
