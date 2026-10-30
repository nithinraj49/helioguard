"""QUBO helpers for assignment problems: one action per item, at most one "exclusive" action per group.

Domain-neutral and reusable. Example: fleet protection (items = satellites, actions =
raise / safe mode / hold, groups = ground windows, exclusive action = raise).

Model (x[i,a] = 1 if item i takes action a; rewards R[i][action], higher is better):
    E(x) = - sum R[i,a] x[i,a]
           + P * sum_i (sum_a x[i,a] - 1)^2                              one action per item
           + P * sum_{i<j, same group} x[i,excl] x[j,excl]               one exclusive action per group
with P = max(max R - min R, max R, -min R) + 1, large enough that breaking a constraint never pays (the range
alone is not enough when rewards are large and similar: a second action then still lowers the energy).
Bit i*len(actions)+a is x[i,a]. The group term is the capacity-1 case of "load over capacity" without slack variables.

Exact enumeration (all_bitstrings, all_energies, solve_exact) is for small problems only: 2**n states.
Solvers outside this module (QAOA, annealers) work on the Qubo or on its Ising form (qubo_to_ising).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["Qubo", "build_qubo", "qubo_energy", "all_bitstrings", "all_energies", "decode", "solve_exact",
           "solve_ortools", "qubo_to_ising"]


@dataclass
class Qubo:
    """Upper-triangular matrix Q (diagonal = linear terms) and a constant: E(x) = const + x^T Q x."""
    n: int
    Q: np.ndarray
    const: float
    rewards: list[dict[str, float]]
    groups: list[Any]
    actions: list[str]
    exclusive: int                          # index in `actions` of the once-per-group action

    @property
    def windows(self) -> list[Any]:         # alias of groups, for callers that call them windows
        return self.groups


def build_qubo(rewards: Sequence[dict[str, float]], groups: Sequence[Any], actions: Sequence[str],
               exclusive_action: str) -> Qubo:
    """Build the QUBO described in the module docstring. `rewards[i]` maps every action to a number."""
    actions = list(actions)
    if len(rewards) != len(groups):
        raise ValueError("rewards and groups must have one entry per item")
    n_items, n_act = len(rewards), len(actions)
    excl = actions.index(exclusive_action)
    allr = [r[a] for r in rewards for a in actions]
    penalty = max(max(allr) - min(allr), max(allr), -min(allr)) + 1.0
    n = n_items * n_act
    Q = np.zeros((n, n))
    const = 0.0
    for i in range(n_items):
        idx = [i * n_act + a for a in range(n_act)]
        for a, v in enumerate(idx):
            Q[v, v] += -rewards[i][actions[a]] - penalty        # reward and -P from (s-1)^2
        for a in range(n_act):
            for b in range(a + 1, n_act):
                Q[idx[a], idx[b]] += 2 * penalty
        const += penalty
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if groups[i] == groups[j]:
                Q[i * n_act + excl, j * n_act + excl] += penalty
    return Qubo(n, Q, const, [dict(r) for r in rewards], list(groups), actions, excl)


def qubo_energy(qubo: Qubo, x: np.ndarray) -> float:
    """E(x) for one bit vector."""
    x = np.asarray(x, dtype=float)
    return float(qubo.const + x @ qubo.Q @ x)


def all_bitstrings(n: int) -> np.ndarray:
    """All 2**n bit vectors as a (2**n, n) float array, most significant bit first."""
    return ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(float)


def all_energies(qubo: Qubo) -> np.ndarray:
    """Energy of every bit vector, in the order of all_bitstrings(qubo.n)."""
    bits = all_bitstrings(qubo.n)
    return qubo.const + np.einsum("ki,ij,kj->k", bits, qubo.Q, bits)


def decode(qubo: Qubo, x: np.ndarray) -> dict[str, Any]:
    """Map a bit vector to one action per item. `feasible` is True only if every item has exactly one
    action and no group has two exclusive actions. Returns assignment, feasible, objective."""
    n_act = len(qubo.actions)
    plan: list[str | None] = []
    for i in range(len(qubo.rewards)):
        chosen = [a for a in range(n_act) if x[i * n_act + a] > 0.5]
        plan.append(qubo.actions[chosen[0]] if len(chosen) == 1 else None)
    ok = all(p is not None for p in plan)
    if ok:
        for g in set(qubo.groups):
            takers = [i for i, p in enumerate(plan) if p == qubo.actions[qubo.exclusive] and qubo.groups[i] == g]
            ok = ok and len(takers) <= 1
    objective = sum(qubo.rewards[i][p] for i, p in enumerate(plan) if p is not None)
    return {"assignment": plan, "feasible": bool(ok), "objective": float(objective)}


def solve_exact(qubo: Qubo) -> dict[str, Any]:
    """Brute-force minimum-energy bit vector (small problems). Adds energy, x and solve_time_s to decode()."""
    t0 = time.perf_counter()
    energies = all_energies(qubo)
    k = int(np.argmin(energies))
    x = all_bitstrings(qubo.n)[k]
    return {**decode(qubo, x), "energy": float(energies[k]), "x": x.astype(int).tolist(),
            "solve_time_s": time.perf_counter() - t0}


def solve_ortools(rewards: Sequence[dict[str, float]], groups: Sequence[Any], actions: Sequence[str],
                  exclusive_action: str, time_limit_s: float = 10.0) -> dict[str, Any]:
    """The same problem as an OR-Tools CP-SAT model (the classical baseline). Needs `ortools`."""
    from ortools.sat.python import cp_model
    actions = list(actions)
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    x = {(i, a): model.NewBoolVar(f"x_{i}_{a}") for i in range(len(rewards)) for a in actions}
    for i in range(len(rewards)):
        model.AddExactlyOne(x[i, a] for a in actions)
    for g in set(groups):
        model.AddAtMostOne(x[i, exclusive_action] for i in range(len(rewards)) if groups[i] == g)
    model.Maximize(sum(round(rewards[i][a] * 1000) * x[i, a] for (i, a) in x))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("OR-Tools found no feasible schedule")
    plan = [next(a for a in actions if solver.Value(x[i, a])) for i in range(len(rewards))]
    return {"assignment": plan, "objective": float(sum(rewards[i][p] for i, p in enumerate(plan))),
            "status": solver.StatusName(status), "feasible": True,
            "solve_time_s": time.perf_counter() - t0}


def qubo_to_ising(qubo: Qubo) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
    """Substitute x = (1 - Z)/2: E = const' + sum_i h[i] Z_i + sum_{i<j} J[i,j] Z_i Z_j.
    Returns (h, J, const'). This is the form a QAOA cost Hamiltonian needs."""
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
