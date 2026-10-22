"""QAOA solver for the assignment QUBOs in verispace_engine.qubo, run on a PennyLane simulator.

PennyLane is an optional dependency (the "quantum" extra) and is imported inside solve_qaoa only, so
the rest of the engine imports without it. Always report QAOA next to a classical baseline
(solve_ortools, solve_exact); this runs on a classical simulator and implies no quantum advantage.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .qubo import Qubo, all_bitstrings, all_energies, decode, qubo_to_ising

__all__ = ["solve_qaoa"]


def solve_qaoa(qubo: Qubo, layers: int = 2, seed: int = 0, restarts: int = 3,
               maxiter: int = 120) -> dict[str, Any]:
    """Depth-`layers` QAOA on `qubo` (at most about 12 qubits), COBYLA over `restarts` random starts.
    Returns the decode() fields plus x, energy, expected_energy, optimum_probability, approx_ratio,
    n_qubits, layers, simulator, evaluations and solve_time_s. Raises ImportError without PennyLane."""
    try:
        import pennylane as qml
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError("solve_qaoa needs PennyLane and SciPy: pip install pennylane pennylane-lightning scipy "
                          "(the optional 'quantum' extra)") from exc

    t0 = time.perf_counter()
    n = qubo.n
    h, J, _ = qubo_to_ising(qubo)
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

    energies = all_energies(qubo)
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

    bits = all_bitstrings(n)
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
