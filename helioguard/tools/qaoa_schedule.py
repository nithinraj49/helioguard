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

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

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


# --- QUBO ------------------------------------------------------------------------------
@dataclass
class Qubo:
    n: int
    Q: np.ndarray                   # upper-triangular, diagonal = linear terms
    const: float
    rewards: list[dict[str, float]]
    windows: list[int]


def build_qubo(rewards: list[dict[str, float]], windows: list[int]) -> Qubo:
    n_assets, n_act = len(rewards), len(ACTION_NAMES)
    allr = [r[a] for r in rewards for a in ACTION_NAMES]
    penalty = (max(allr) - min(allr)) + 1.0
    n = n_assets * n_act
    Q = np.zeros((n, n))
    const = 0.0
    for i in range(n_assets):
        idx = [i * n_act + a for a in range(n_act)]
        for a, v in enumerate(idx):
            Q[v, v] += -rewards[i][ACTION_NAMES[a]] - penalty      # reward and -P from (s-1)^2
        for a in range(n_act):
            for b in range(a + 1, n_act):
                Q[idx[a], idx[b]] += 2 * penalty
        const += penalty
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if windows[i] == windows[j]:
                Q[i * n_act + RAISE, j * n_act + RAISE] += penalty
    return Qubo(n, Q, const, rewards, list(windows))


def qubo_energy(qubo: Qubo, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(qubo.const + x @ qubo.Q @ x)


def _all_bitstrings(n: int) -> np.ndarray:
    return ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(float)


def _all_energies(qubo: Qubo) -> np.ndarray:
    bits = _all_bitstrings(qubo.n)
    return qubo.const + np.einsum("ki,ij,kj->k", bits, qubo.Q, bits)


def decode(qubo: Qubo, x: np.ndarray) -> dict[str, Any]:
    """Map a bit vector to one action per asset; feasible only if every constraint holds."""
    n_act = len(ACTION_NAMES)
    plan: list[str | None] = []
    for i in range(len(qubo.rewards)):
        chosen = [a for a in range(n_act) if x[i * n_act + a] > 0.5]
        plan.append(ACTION_NAMES[chosen[0]] if len(chosen) == 1 else None)
    ok = all(p is not None for p in plan)
    if ok:
        for w in set(qubo.windows):
            raisers = [i for i, p in enumerate(plan) if p == ACTION_NAMES[RAISE] and qubo.windows[i] == w]
            ok = ok and len(raisers) <= 1
    objective = sum(qubo.rewards[i][p] for i, p in enumerate(plan) if p is not None)
    return {"assignment": plan, "feasible": bool(ok), "objective": float(objective)}


def solve_exact(qubo: Qubo) -> dict[str, Any]:
    t0 = time.perf_counter()
    energies = _all_energies(qubo)
    k = int(np.argmin(energies))
    x = _all_bitstrings(qubo.n)[k]
    return {**decode(qubo, x), "energy": float(energies[k]), "x": x.astype(int).tolist(),
            "solve_time_s": time.perf_counter() - t0}


def solve_ortools(rewards: list[dict[str, float]], windows: list[int]) -> dict[str, Any]:
    from ortools.sat.python import cp_model
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    x = {(i, a): model.NewBoolVar(f"x_{i}_{a}") for i in range(len(rewards)) for a in ACTION_NAMES}
    for i in range(len(rewards)):
        model.AddExactlyOne(x[i, a] for a in ACTION_NAMES)
    for w in set(windows):
        model.AddAtMostOne(x[i, ACTION_NAMES[RAISE]] for i in range(len(rewards)) if windows[i] == w)
    model.Maximize(sum(round(rewards[i][a] * 1000) * x[i, a] for (i, a) in x))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("OR-Tools found no feasible schedule")
    plan = [next(a for a in ACTION_NAMES if solver.Value(x[i, a])) for i in range(len(rewards))]
    return {"assignment": plan, "objective": float(sum(rewards[i][p] for i, p in enumerate(plan))),
            "status": solver.StatusName(status), "feasible": True,
            "solve_time_s": time.perf_counter() - t0}


# --- QAOA ------------------------------------------------------------------------------
def _ising(qubo: Qubo) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
    """x = (1 - Z)/2  ->  E = const' + sum h_i Z_i + sum J_ij Z_i Z_j."""
    n, Q = qubo.n, qubo.Q
    h = -np.diag(Q) / 2
    J: dict[tuple[int, int], float] = {}
    const = qubo.const + np.trace(Q) / 2
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                J[(i, j)] = Q[i, j] / 4
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                const += Q[i, j] / 4
    return h, J, float(const)


def solve_qaoa(qubo: Qubo, layers: int = 2, seed: int = 0, restarts: int = 3,
               maxiter: int = 120) -> dict[str, Any]:
    import pennylane as qml
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    n = qubo.n
    h, J, _ = _ising(qubo)
    scale = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=0.0), 1e-9)
    terms = [(h[i] / scale, qml.PauliZ(i)) for i in range(n) if h[i] != 0]
    terms += [(v / scale, qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j), v in J.items()]
    cost_h = qml.dot([c for c, _ in terms], [o for _, o in terms])
    mixer_h = qml.qaoa.x_mixer(range(n))
    try:
        dev = qml.device("lightning.qubit", wires=n)
    except Exception:                                    # lightning not installed
        dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def probabilities(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(layers):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            qml.qaoa.mixer_layer(params[1][layer], mixer_h)
        return qml.probs(wires=range(n))

    energies = _all_energies(qubo)
    calls = 0

    def expected(flat: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        p = probabilities(np.array([flat[:layers], flat[layers:]]))
        return float(np.dot(p, energies))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        res = minimize(expected, rng.uniform(0, np.pi, 2 * layers), method="COBYLA",
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    probs = np.asarray(probabilities(np.array([best.x[:layers], best.x[layers:]])))

    bits = _all_bitstrings(n)
    e_min = float(energies.min())
    top = np.argsort(-probs)[:64]
    feasible_top = [k for k in top if decode(qubo, bits[k])["feasible"]]
    pick = min(feasible_top, key=lambda k: energies[k]) if feasible_top else int(top[0])
    dec = decode(qubo, bits[pick])
    uniform = float(energies.mean())
    gap = uniform - e_min
    return {**dec, "x": bits[pick].astype(int).tolist(), "energy": float(energies[pick]),
            "expected_energy": float(best.fun),
            "optimum_probability": float(probs[energies <= e_min + 1e-9].sum()),
            "approx_ratio": float((uniform - best.fun) / gap) if gap > 0 else 1.0,
            "n_qubits": n, "layers": layers, "simulator": dev.name, "evaluations": calls,
            "solve_time_s": time.perf_counter() - t0}


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
