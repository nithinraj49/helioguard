"""Phase 7 QAOA benchmark harness: simulator fidelity, metric definitions, instances, statistics.

Spacecraft values in the instances are ILLUSTRATIVE. No quantum advantage is claimed or tested for.
"""
import random
from pathlib import Path

import numpy as np
import pytest

from helioguard.qaoa_instances import FAMILIES, make_instance, make_qubo
from verispace_engine import qaoa_bench as qb
from verispace_engine.qubo import all_bitstrings, all_energies, build_qubo, decode, solve_exact, solve_ortools

pennylane = pytest.importorskip("pennylane")


def small_qubo(seed=0, items=3, acts=2):
    rng = random.Random(seed)
    actions = ["raise", "safe", "hold"][:acts]
    rewards = [{a: rng.uniform(-0.2, 1.6) for a in actions} for _ in range(items)]
    return build_qubo(rewards, [i % 2 for i in range(items)], actions, "raise")


# --- G0(ii): the fast simulator is the engine's circuit -----------------------------------
@pytest.mark.parametrize("items,acts,layers", [(3, 2, 1), (4, 2, 3), (2, 3, 2)])
def test_fast_simulator_matches_the_pennylane_circuit(items, acts, layers):
    qubo = small_qubo(1, items, acts)
    params = np.random.default_rng(5).uniform(0, np.pi, 2 * layers)
    ref = qb.pennylane_probabilities(qubo, layers, params)
    got = qb.FastQaoa(qubo, layers).probabilities(params)
    assert got.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(got, ref, atol=1e-9)


def test_zero_angles_give_the_uniform_distribution():
    sim = qb.FastQaoa(small_qubo(), 2)
    assert np.allclose(sim.probabilities(np.zeros(4)), 1 / 2 ** sim.n)


# --- G0(iii): COBYLA + random start agrees with the engine -------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harness_agrees_with_solve_qaoa_to_optimiser_noise(seed):
    """The circuit is identical (test above, 1e-9), so the objective is the same function. COBYLA is
    sensitive to 1e-16 float noise, so paths are not bit-identical and at p >= 2 can end in a different
    local optimum (seen in a pre-registration probe); at p = 1 they agree to ~1e-7."""
    from verispace_engine.qaoa import solve_qaoa
    qubo = small_qubo(2, 3, 2)
    eng = solve_qaoa(qubo, layers=1, seed=seed, restarts=3, maxiter=120)
    mine = qb.run_variant(qubo, layers=1, optimiser="COBYLA", init="random", seed=seed)
    assert mine["expected_energy"] == pytest.approx(eng["expected_energy"], abs=1e-5)
    assert mine["optimum_probability"] == pytest.approx(eng["optimum_probability"], abs=1e-3)
    assert mine["evaluations"] <= 3 * 120 and eng["evaluations"] <= 3 * 120


# --- metric definitions --------------------------------------------------------------------
def test_feasibility_mask_agrees_with_engine_decode():
    qubo = small_qubo(3, 4, 2)
    bits = all_bitstrings(qubo.n)
    mask = qb.feasible_mask(qubo)
    for k in random.Random(0).sample(range(len(bits)), 80):
        assert bool(mask[k]) == decode(qubo, bits[k])["feasible"]


def test_metrics_on_a_distribution_concentrated_on_a_wrong_state():
    """Loophole: a confident but non-optimal state must score zero probability and no hit."""
    qubo = small_qubo(4, 3, 2)
    energies = all_energies(qubo)
    worst = int(np.argmax(energies))
    probs = np.zeros(2 ** qubo.n)
    probs[worst] = 1.0
    m = qb.metrics_from_probs(qubo, probs)
    assert m["optimum_probability"] == 0.0
    assert m["top64_hit"] is False
    assert m["approx_ratio"] < 0                           # worse than uniform


def test_metrics_on_the_exact_optimum_distribution():
    qubo = small_qubo(5, 3, 2)
    energies = all_energies(qubo)
    probs = (energies <= energies.min() + 1e-9).astype(float)
    probs /= probs.sum()
    m = qb.metrics_from_probs(qubo, probs)
    assert m["optimum_probability"] == pytest.approx(1.0)
    assert m["top64_hit"] is True
    assert m["approx_ratio"] == pytest.approx(1.0)


def test_top64_pick_prefers_feasible_over_lower_energy_infeasible():
    qubo = small_qubo(6, 3, 2)
    energies, mask = all_energies(qubo), qb.feasible_mask(qubo)
    infeasible = np.where(~mask)[0]
    k = infeasible[np.argmin(energies[infeasible])]
    feasible_k = int(np.where(mask)[0][np.argmax(energies[mask])])
    probs = np.zeros_like(energies)
    probs[k], probs[feasible_k] = 0.6, 0.4
    m = qb.metrics_from_probs(qubo, probs)
    assert m["top64_energy"] == pytest.approx(energies[feasible_k])


def test_uniform_baseline_formulas():
    b = qb.uniform_baseline(n_optimal=2, n_qubits=6, shots=64, budget=100)
    assert b["p_optimum"] == pytest.approx(2 / 64)
    assert b["best_of_shots"] == pytest.approx(1 - (1 - 2 / 64) ** 64)
    assert b["best_of_budget"] == pytest.approx(1 - (1 - 2 / 64) ** 100)
    assert qb.shots_hit_probability(0.0, 64) == 0.0
    assert qb.shots_hit_probability(1.0, 64) == 1.0


# --- statistics ----------------------------------------------------------------------------
def test_bootstrap_is_deterministic_and_brackets_the_mean():
    vals = np.array([0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.7, 0.6, 0.2, 0.8])
    a, b = qb.bootstrap_ci(vals), qb.bootstrap_ci(vals)
    assert a == b
    assert a[0] <= vals.mean() <= a[1]
    assert qb.bootstrap_ci(np.full(10, 0.25)) == (0.25, 0.25)


def test_paired_bootstrap_detects_a_real_shift_and_not_noise():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 1, 10)
    lo, hi = qb.bootstrap_ci(base + 0.3 - base)                # constant +0.3
    assert lo > 0
    noise = rng.normal(0, 0.2, 10)
    lo, hi = qb.bootstrap_ci(noise - noise.mean())             # mean exactly zero
    assert lo < 0 < hi
    assert qb.verdict(0.1, 0.3) == "helps" and qb.verdict(-0.3, -0.1) == "hurts"
    assert qb.verdict(-0.1, 0.2) == "no detectable change"


def test_slope_ci_recovers_an_exponential_decay():
    sizes = [6, 8, 10, 12]
    per_size = [np.full(10, 10 ** (-0.1 * n)) for n in sizes]
    slope, lo, hi = qb.log_slope(sizes, per_size)
    assert slope == pytest.approx(-0.1, abs=1e-9) and lo - 1e-9 <= slope <= hi + 1e-9


# --- optimiser variants -----------------------------------------------------------------------
@pytest.mark.parametrize("opt", ["COBYLA", "Nelder-Mead", "L-BFGS-B"])
@pytest.mark.parametrize("init", ["random", "ramp"])
def test_every_variant_runs_within_the_evaluation_budget(opt, init):
    qubo = small_qubo(7, 3, 2)
    r = qb.run_variant(qubo, layers=2, optimiser=opt, init=init, seed=1, restarts=2, maxiter=40)
    assert 0 <= r["optimum_probability"] <= 1
    assert r["evaluations"] <= 2 * 40 + 2 * 8              # slack for optimiser overshoot of maxfev
    assert r["feasible_probability"] <= 1.0
    assert r["top64_energy"] >= all_energies(qubo).min() - 1e-9


def test_unknown_optimiser_or_init_is_rejected():
    qubo = small_qubo()
    with pytest.raises(ValueError):
        qb.run_variant(qubo, 1, "ADAM", "random", 0)
    with pytest.raises(ValueError):
        qb.run_variant(qubo, 1, "COBYLA", "warm", 0)


def test_ramp_init_is_deterministic_and_jitters_later_restarts():
    a = qb.initial_params("ramp", 3, np.random.default_rng(0), restart=0)
    b = qb.initial_params("ramp", 3, np.random.default_rng(9), restart=0)
    c = qb.initial_params("ramp", 3, np.random.default_rng(0), restart=1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a[0] < a[1] < a[2] and a[3] > a[4] > a[5]       # gamma ramps up, beta ramps down


# --- instances ---------------------------------------------------------------------------------
@pytest.mark.parametrize("family,n", [("B", 6), ("B", 8), ("B", 16), ("A", 6), ("A", 9), ("A", 15)])
def test_instances_have_the_requested_qubit_count_and_are_deterministic(family, n):
    q1, q2 = make_qubo(family, n, 0), make_qubo(family, n, 0)
    assert q1.n == n and np.array_equal(q1.Q, q2.Q)
    assert not np.array_equal(q1.Q, make_qubo(family, n, 1).Q)


def test_requested_sizes_are_not_all_native_to_the_three_action_fleet():
    assert FAMILIES["A"]["sizes"] == [6, 9, 12, 15]
    assert FAMILIES["B"]["sizes"] == [6, 8, 10, 12, 14, 16]
    with pytest.raises(ValueError):
        make_instance("A", 8, 0)                           # 8 is not a multiple of 3


def test_exact_and_ortools_agree_on_every_small_instance():
    """G0(i). Exact is the reference; OR-Tools may trail it only by its own reward rounding (Amendment 1)."""
    for family, n in [("B", 6), ("B", 8), ("B", 10), ("A", 6), ("A", 9)]:
        for k in range(10):
            rewards, windows, actions = make_instance(family, n, k)
            qubo = make_qubo(family, n, k)
            ort = solve_ortools(rewards, windows, actions, actions[0])
            items = len(rewards)
            gap = solve_exact(qubo)["objective"] - ort["objective"]
            assert -1e-9 <= gap <= 0.5e-3 * items          # OR-Tools rounds rewards to 1e-3 (worst case 0.5e-3 each)


# --- gates doc ---------------------------------------------------------------------------------
def test_gates_doc_is_registered_and_claims_no_advantage():
    text = (Path(__file__).resolve().parents[1] / "docs" / "phase7_gates.md").read_text(encoding="utf-8")
    for gate in ("G0", "G1", "G2", "G3", "G4", "G5", "G6"):
        assert f"**{gate}" in text
    assert "Not claimed" in text and "Registered: 2026-10-30" in text


def test_sparse_energies_equal_the_engines_dense_energies():
    for qubo in (small_qubo(8, 4, 2), small_qubo(9, 2, 3), make_qubo("B", 10, 3)):
        assert np.allclose(qb.Landscape.of(qubo).energies, all_energies(qubo), atol=1e-9)
