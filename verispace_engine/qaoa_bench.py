"""Benchmark harness for QAOA on assignment QUBOs (Phase 7). Does not touch the solvers in qaoa.py / qubo.py.

`solve_qaoa` fixes its optimiser (COBYLA) and its start (uniform random), so varying those needs a
second implementation of the same circuit. FastQaoa is a numpy statevector simulator of exactly the
engine's circuit (Hadamards, then per layer exp(-i gamma H_c) and exp(-i beta sum X), with H_c the engine's
normalised Ising Hamiltonian). tests/test_qaoa_bench.py pins it to the PennyLane circuit and to solve_qaoa.

Everything here runs on a classical simulator and implies no quantum advantage. The "top-64 readout" metric
reads the full probability vector, which hardware cannot do; the "64 shots" metric is the hardware-style one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .qubo import Qubo, qubo_to_ising

__all__ = ["FastQaoa", "Landscape", "OPTIMISERS", "INITS", "initial_params", "run_variant", "metrics_from_probs",
           "feasible_mask", "uniform_baseline", "shots_hit_probability", "bootstrap_ci", "verdict", "log_slope",
           "pennylane_probabilities"]

OPTIMISERS = ("COBYLA", "Nelder-Mead", "L-BFGS-B")
INITS = ("random", "ramp")
RAMP_SCALE = 0.75          # gamma_k = RAMP_SCALE (k+0.5)/p, beta_k = RAMP_SCALE (1 - (k+0.5)/p)
RAMP_JITTER = 0.1          # std of the Gaussian jitter added on restarts after the first
TOL = 1e-9                 # "optimal" means energy <= minimum + TOL (the engine's definition)


def _bit(n: int, w: int) -> np.ndarray:
    """Bit of wire w (wire 0 = most significant, as in all_bitstrings) for every basis state, as uint8."""
    return ((np.arange(2 ** n, dtype=np.int64) >> (n - 1 - w)) & 1).astype(np.uint8)


def _energies(qubo: Qubo) -> np.ndarray:
    """E(x) for every bitstring, sparse in Q (same values as qubo.all_energies, less memory at 18+ qubits)."""
    n, Q = qubo.n, qubo.Q
    bits = [_bit(n, w) for w in range(n)]
    e = np.full(2 ** n, float(qubo.const))
    for i in range(n):
        if Q[i, i] != 0:
            e += Q[i, i] * bits[i]
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                e += Q[i, j] * (bits[i] & bits[j])
    return e


def feasible_mask(qubo: Qubo) -> np.ndarray:
    """True for bitstrings that decode() calls feasible: one action per item, <= 1 exclusive action per group."""
    n, n_act = qubo.n, len(qubo.actions)
    bits = [_bit(n, w).astype(np.int8) for w in range(n)]
    ok = np.ones(2 ** n, dtype=bool)
    for i in range(len(qubo.rewards)):
        ok &= sum(bits[i * n_act + a] for a in range(n_act)) == 1
    for g in set(qubo.groups):
        members = [i for i in range(len(qubo.groups)) if qubo.groups[i] == g]
        ok &= sum(bits[i * n_act + qubo.exclusive] for i in members) <= 1
    return ok


@dataclass
class Landscape:
    """Everything about a QUBO that metrics need: energies, optimum, feasibility, uniform mean."""
    n: int
    energies: np.ndarray
    feasible: np.ndarray
    e_min: float
    optimal: np.ndarray
    uniform_mean: float

    @classmethod
    def of(cls, qubo: Qubo) -> "Landscape":
        e = _energies(qubo)
        e_min = float(e.min())
        return cls(qubo.n, e, feasible_mask(qubo), e_min, e <= e_min + TOL, float(e.mean()))


class FastQaoa:
    """Statevector simulation of depth-`layers` QAOA on `qubo`; parameters are [gammas..., betas...]."""

    def __init__(self, qubo: Qubo, layers: int, landscape: Landscape | None = None) -> None:
        self.n, self.layers = qubo.n, layers
        self.landscape = landscape or Landscape.of(qubo)
        h, J, const = qubo_to_ising(qubo)
        scale = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=0.0), 1e-9)   # as in solve_qaoa
        self._phase_energy = (self.landscape.energies - const) / scale

    def probabilities(self, params: np.ndarray) -> np.ndarray:
        n, L = self.n, self.layers
        gammas, betas = params[:L], params[L:2 * L]
        psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
        for g, b in zip(gammas, betas):
            psi *= np.exp(-1j * g * self._phase_energy)
            c, s = np.cos(b), -1j * np.sin(b)
            for w in range(n):
                v = psi.reshape(2 ** w, 2, -1)
                a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
                v[:, 0, :] = c * a0 + s * a1
                v[:, 1, :] = c * a1 + s * a0
        return psi.real ** 2 + psi.imag ** 2


def pennylane_probabilities(qubo: Qubo, layers: int, params: np.ndarray) -> np.ndarray:
    """The engine's PennyLane circuit (copied from solve_qaoa) at `params`; used to pin FastQaoa."""
    import pennylane as qml
    n = qubo.n
    h, J, _ = qubo_to_ising(qubo)
    scale = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=0.0), 1e-9)
    terms = [(h[i] / scale, qml.PauliZ(i)) for i in range(n) if h[i] != 0]
    terms += [(v / scale, qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j), v in J.items()]
    cost_h = qml.Hamiltonian([c for c, _ in terms], [o for _, o in terms])
    mixer_h = qml.qaoa.x_mixer(range(n))
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(p):
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(layers):
            qml.qaoa.cost_layer(p[0][layer], cost_h)
            qml.qaoa.mixer_layer(p[1][layer], mixer_h)
        return qml.probs(wires=range(n))

    return np.asarray(circuit(np.array([params[:layers], params[layers:]])))


# --- metrics ----------------------------------------------------------------------------------
def shots_hit_probability(p_optimum: float, shots: int = 64) -> float:
    """Chance that at least one of `shots` independent samples is an optimum."""
    return float(1.0 - (1.0 - p_optimum) ** shots)


def uniform_baseline(n_optimal: int, n_qubits: int, shots: int = 64, budget: int | None = None) -> dict[str, float]:
    """Classical reference: sampling bitstrings uniformly at random."""
    p = n_optimal / 2 ** n_qubits
    out = {"p_optimum": p, "best_of_shots": shots_hit_probability(p, shots), "approx_ratio": 0.0}
    if budget is not None:
        out["best_of_budget"] = shots_hit_probability(p, budget)
    return out


def metrics_from_probs(qubo: Qubo, probs: np.ndarray, landscape: Landscape | None = None) -> dict[str, Any]:
    """Single-sample P(optimum), approximation ratio, feasible mass, and the engine's top-64 readout
    (best feasible bitstring among the 64 most probable ones; falls back to the most probable)."""
    land = landscape or Landscape.of(qubo)
    probs = np.asarray(probs, dtype=float)
    p_opt = float(probs[land.optimal].sum())
    expected = float(probs @ land.energies)
    gap = land.uniform_mean - land.e_min
    k = min(64, len(probs))
    top = np.argpartition(-probs, k - 1)[:k]
    top = top[np.argsort(-probs[top], kind="stable")]
    cand = top[probs[top] > 1e-12]
    if len(cand) == 0:
        cand = top[:1]
    feas = cand[land.feasible[cand]]
    pick = int(feas[np.argmin(land.energies[feas])]) if len(feas) else int(cand[0])
    return {"optimum_probability": p_opt, "expected_energy": expected,
            "approx_ratio": float((land.uniform_mean - expected) / gap) if gap > 0 else 1.0,
            "feasible_probability": float(probs[land.feasible].sum()),
            "top64_energy": float(land.energies[pick]),
            "top64_hit": bool(land.energies[pick] <= land.e_min + TOL),
            "top64_feasible": bool(land.feasible[pick]),
            "best_of_64_shots": shots_hit_probability(p_opt, 64)}


# --- optimiser variants ------------------------------------------------------------------------
def initial_params(init: str, layers: int, rng: np.random.Generator, restart: int = 0) -> np.ndarray:
    if init == "random":                       # the engine's start: uniform on (0, pi) for every angle
        return rng.uniform(0, np.pi, 2 * layers)
    if init == "ramp":
        frac = (np.arange(layers) + 0.5) / layers
        x = np.concatenate([RAMP_SCALE * frac, RAMP_SCALE * (1 - frac)])
        return x if restart == 0 else x + rng.normal(0, RAMP_JITTER, 2 * layers)
    raise ValueError(f"unknown init {init!r}; choose from {INITS}")


def _options(optimiser: str, maxiter: int) -> dict[str, Any]:
    if optimiser == "COBYLA":
        return {"maxiter": maxiter}
    if optimiser == "Nelder-Mead":
        return {"maxfev": maxiter, "maxiter": maxiter}
    if optimiser == "L-BFGS-B":
        return {"maxfun": maxiter, "maxiter": maxiter}
    raise ValueError(f"unknown optimiser {optimiser!r}; choose from {OPTIMISERS}")


def run_variant(qubo: Qubo, layers: int, optimiser: str, init: str, seed: int, restarts: int = 3,
                maxiter: int = 120, landscape: Landscape | None = None) -> dict[str, Any]:
    """QAOA with the given optimiser and start on the fast simulator, same budget rules as solve_qaoa:
    `restarts` starts, at most `maxiter` objective evaluations each, every evaluation counted."""
    from scipy.optimize import minimize
    options = _options(optimiser, maxiter)
    if init not in INITS:
        raise ValueError(f"unknown init {init!r}; choose from {INITS}")
    t0 = time.perf_counter()
    sim = FastQaoa(qubo, layers, landscape)
    energies = sim.landscape.energies
    calls = 0

    def expected(flat: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return float(sim.probabilities(flat) @ energies)

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        res = minimize(expected, initial_params(init, layers, rng, r), method=optimiser, options=options)
        if best is None or res.fun < best.fun:
            best = res
    m = metrics_from_probs(qubo, sim.probabilities(best.x), sim.landscape)
    return {**m, "expected_energy": float(best.fun), "n_qubits": qubo.n, "layers": layers, "optimiser": optimiser,
            "init": init, "seed": seed, "evaluations": calls, "solve_time_s": time.perf_counter() - t0}


# --- statistics ---------------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over `values` (one value per instance)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def verdict(lo: float, hi: float) -> str:
    """Three-way reading of a CI on a paired difference (G4, G5)."""
    return "helps" if lo > 0 else "hurts" if hi < 0 else "no detectable change"


def log_slope(sizes, per_size_values, n_boot: int = 10_000, seed: int = 0, floor: float = 1e-12):
    """Slope of log10(mean value) against qubit count, with an instance-resampling bootstrap CI.
    Returns (slope, lo, hi). Sizes use independent instances, so each is resampled independently."""
    x = np.asarray(sizes, dtype=float)
    vals = [np.asarray(v, dtype=float) for v in per_size_values]

    def slope_of(means):
        return float(np.polyfit(x, np.log10(np.maximum(means, floor)), 1)[0])

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = slope_of([v[rng.integers(0, len(v), len(v))].mean() for v in vals])
    return slope_of([v.mean() for v in vals]), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
