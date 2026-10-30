"""verispace_engine.qubo: the public QUBO helpers, plus no-behaviour-change checks for HelioGuard."""
import hashlib
import itertools

import numpy as np
import pytest

from helioguard.tools import qaoa_schedule as old
from verispace_engine import qubo as engine

# Values captured from the pre-refactor helioguard.tools.qaoa_schedule on this fixed input.
REWARDS = [{"raise_orbit": 1.2, "safe_mode_edge_on": 0.9, "hold": 0.3},
           {"raise_orbit": 1.1, "safe_mode_edge_on": 0.95, "hold": 0.2},
           {"raise_orbit": -1.0, "safe_mode_edge_on": 0.5, "hold": 0.4}]
WINDOWS = [0, 0, 1]
GOLDEN_Q = "97cfbdcb685ab7bdd3d810fec4c8ab95980061e3bffb26e57140c8f828afc46b"
GOLDEN_H = "7678db8fb34abb0f4891508726c47aff3399d9ec817a8b1b3d59b8fe95d51d99"
GOLDEN_ISING_CONST = 8.125
GOLDEN_EXACT = (["raise_orbit", "safe_mode_edge_on", "safe_mode_edge_on"], -2.6499999999999986)


def digest(a):
    return hashlib.sha256(np.round(a, 9).tobytes()).hexdigest()


# --- generic, non-satellite use ---------------------------------------------------------------
def test_engine_qubo_works_for_any_actions_and_groups():
    actions = ["day", "night", "off"]
    rewards = [{"day": 3.0, "night": 1.0, "off": 0.0}, {"day": 2.5, "night": 1.2, "off": 0.0}]
    q = engine.build_qubo(rewards, [7, 7], actions, exclusive_action="day")
    best = engine.solve_exact(q)
    assert best["feasible"] and best["assignment"].count("day") == 1       # only one "day" per group 7
    assert best["objective"] == pytest.approx(3.0 + 1.2)


def test_public_api_is_documented():
    for name in ("Qubo", "build_qubo", "qubo_energy", "all_bitstrings", "all_energies", "decode",
                 "solve_exact", "solve_ortools", "qubo_to_ising"):
        assert getattr(engine, name).__doc__, name
    assert engine.__doc__


def test_energy_matches_brute_force_and_constraints_cost_energy():
    q = engine.build_qubo(REWARDS, WINDOWS, list(REWARDS[0]), "raise_orbit")
    bits = engine.all_bitstrings(q.n)
    assert bits.shape == (2 ** q.n, q.n)
    e = engine.all_energies(q)
    for k in (0, 5, 100, 400):
        assert e[k] == pytest.approx(engine.qubo_energy(q, bits[k]))
    feasible = [engine.decode(q, b)["feasible"] for b in bits]
    assert min(e[i] for i, ok in enumerate(feasible) if not ok) > e.min()    # no infeasible state wins


def test_ising_form_reproduces_the_qubo_energy():
    q = engine.build_qubo(REWARDS, WINDOWS, list(REWARDS[0]), "raise_orbit")
    h, J, const = engine.qubo_to_ising(q)
    for x in itertools.islice(engine.all_bitstrings(q.n), 0, 512, 37):
        z = 1 - 2 * x                                                        # x = (1 - Z) / 2
        e = const + h @ z + sum(v * z[i] * z[j] for (i, j), v in J.items())
        assert e == pytest.approx(engine.qubo_energy(q, x))


def test_decode_flags_bad_bit_vectors():
    q = engine.build_qubo(REWARDS, WINDOWS, list(REWARDS[0]), "raise_orbit")
    none_chosen, two_raises = np.zeros(q.n), np.zeros(q.n)
    two_raises[[0, 3, 7]] = 1                       # assets 0 and 1 both raise in window 0
    assert not engine.decode(q, none_chosen)["feasible"]
    assert not engine.decode(q, two_raises)["feasible"]


# --- no behaviour change in HelioGuard --------------------------------------------------------
def test_old_private_names_are_aliases_of_the_engine_functions():
    assert old._ising is engine.qubo_to_ising
    assert old._all_bitstrings is engine.all_bitstrings
    assert old._all_energies is engine.all_energies
    assert old.Qubo is engine.Qubo and old.qubo_energy is engine.qubo_energy
    assert old.decode is engine.decode and old.solve_exact is engine.solve_exact


def test_helioguard_qubo_is_numerically_identical_to_before_the_move():
    q = old.build_qubo(REWARDS, WINDOWS)
    assert digest(q.Q) == GOLDEN_Q and q.const == pytest.approx(9.6)
    h, J, const = old._ising(q)
    assert digest(h) == GOLDEN_H and const == pytest.approx(GOLDEN_ISING_CONST) and J[(0, 1)] == pytest.approx(1.6)
    exact = old.solve_exact(q)
    assert exact["assignment"] == GOLDEN_EXACT[0] and exact["energy"] == pytest.approx(GOLDEN_EXACT[1])
    assert old.solve_ortools(REWARDS, WINDOWS)["assignment"] == GOLDEN_EXACT[0]


# --- QAOA in the engine (optional "quantum" extra) --------------------------------------------
GOLDEN_QAOA = {"assignment": ["raise_orbit", "safe_mode_edge_on", "safe_mode_edge_on"], "feasible": True,
               "objective": 2.65, "x": [1, 0, 0, 0, 1, 0, 0, 1, 0], "energy": -2.6499999999999986,
               "expected_energy": 2.6861832403916432, "optimum_probability": 0.008551808618467643,
               "approx_ratio": 0.5047625762977594, "n_qubits": 9, "layers": 2, "simulator": "lightning.qubit",
               "evaluations": 269}


def test_qaoa_result_is_identical_before_and_after_the_move():
    from verispace_engine.qaoa import solve_qaoa
    q = old.build_qubo(REWARDS, WINDOWS)
    for fn in (solve_qaoa, old.solve_qaoa):
        out = fn(q, layers=2)
        out.pop("solve_time_s")
        assert out == pytest.approx(GOLDEN_QAOA)


def test_helioguard_keeps_the_old_name_as_an_alias():
    from verispace_engine.qaoa import solve_qaoa
    assert old.solve_qaoa is solve_qaoa


def test_engine_imports_without_pennylane_and_errors_clearly_when_used(tmp_path):
    import subprocess
    import sys
    code = ("import sys\n"
            "sys.modules['pennylane'] = None\n"                  # makes `import pennylane` raise ImportError
            "import verispace_engine, verispace_engine.qubo, verispace_engine.qaoa as q\n"
            "import verispace_engine.harness\n"
            "assert 'pennylane' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
            "try:\n"
            "    q.solve_qaoa(None)\n"
            "except ImportError as e:\n"
            "    assert 'pennylane' in str(e) and 'quantum' in str(e), e\n"
            "else:\n"
            "    raise SystemExit('no error')\n")
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         cwd=str(__import__("pathlib").Path(__file__).parent.parent))
    assert run.returncode == 0, run.stderr


def test_pyproject_has_an_optional_quantum_extra():
    import tomllib
    from pathlib import Path
    extras = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert any(d.startswith("pennylane") for d in extras["quantum"])
    assert "quantum" not in str(tomllib.loads(Path("pyproject.toml").read_text())["project"].get("dependencies", ""))


# --- decode_fn and the one-qubit case (added after the Navigator changes to qaoa.py) -----------
def test_decode_fn_replaces_the_default_decoder():
    from verispace_engine.qaoa import solve_qaoa
    q = old.build_qubo(REWARDS, WINDOWS)
    calls = []

    def custom(qubo, x):
        calls.append(1)
        return {"feasible": True, "custom": int(sum(x))}

    out = solve_qaoa(q, layers=1, decode_fn=custom)
    assert calls and out["custom"] == sum(out["x"]) and "assignment" not in out      # its fields, not decode()'s
    default = solve_qaoa(q, layers=1)
    assert "assignment" in default and "custom" not in default


def test_decode_fn_feasibility_steers_the_pick():
    from verispace_engine.qaoa import solve_qaoa
    q = old.build_qubo(REWARDS, WINDOWS)
    only_asset0_raises = lambda qubo, x: {"feasible": bool(x[0] > 0.5), "x0": float(x[0])}
    assert solve_qaoa(q, layers=1, decode_fn=only_asset0_raises)["x"][0] == 1


def test_qaoa_on_a_one_qubit_qubo():
    from verispace_engine.qaoa import solve_qaoa
    q = engine.Qubo(n=1, Q=np.array([[-1.0]]), const=0.0, rewards=[], groups=[], actions=["a"], exclusive=0)
    out = solve_qaoa(q, layers=1, decode_fn=lambda qubo, x: {"feasible": True})
    assert out["x"] == [1] and out["energy"] == pytest.approx(-1.0) and out["n_qubits"] == 1
    assert out["optimum_probability"] > 0.5


# --- penalty strength (Phase 7 regression) -------------------------------------------------------
@pytest.mark.parametrize("rewards", [
    [{"raise_orbit": 1.19, "safe": 1.11}, {"raise_orbit": 1.18, "safe": 1.10}, {"raise_orbit": 1.22, "safe": 1.14}],
    [{"raise_orbit": 5.0, "safe": 4.9, "hold": 4.8}] * 2,
    [{"raise_orbit": -3.0, "safe": -3.1}] * 2,
])
def test_penalty_beats_any_reward_when_rewards_are_large_and_similar(rewards):
    """Loophole: with a small reward range but large rewards, the old penalty (range + 1) was below one
    reward, so taking two actions for one asset lowered the energy: the QUBO minimum was infeasible."""
    actions = list(rewards[0])
    qubo = engine.build_qubo(rewards, [0] * len(rewards), actions, "raise_orbit")
    exact = engine.solve_exact(qubo)
    assert exact["feasible"]
    ort = engine.solve_ortools(rewards, [0] * len(rewards), actions, "raise_orbit")
    assert exact["objective"] == pytest.approx(ort["objective"], abs=1e-3)


# --- penalty_rule: "legacy" reproduces the original matrix, "sufficient" is the default ----------------
ACTIONS = ["raise_orbit", "safe_mode_edge_on", "hold"]


def test_legacy_rule_reproduces_the_original_matrix_exactly():
    q = engine.build_qubo(REWARDS, WINDOWS, ACTIONS, "raise_orbit", penalty_rule="legacy")
    assert digest(q.Q) == GOLDEN_Q and q.const == pytest.approx(9.6)       # P = (1.2 - -1.0) + 1 = 3.2


def test_the_helioguard_wrapper_defaults_to_legacy_and_the_engine_to_sufficient():
    assert np.array_equal(old.build_qubo(REWARDS, WINDOWS).Q, engine.build_qubo(REWARDS, WINDOWS, ACTIONS, "raise_orbit", "legacy").Q)
    sm = [{"raise_orbit": 1.19, "safe": 1.11}] * 2
    legacy = engine.build_qubo(sm, [0, 0], ["raise_orbit", "safe"], "raise_orbit", "legacy")
    default = engine.build_qubo(sm, [0, 0], ["raise_orbit", "safe"], "raise_orbit")
    assert default.const == pytest.approx(2 * (1.19 + 1)) and legacy.const == pytest.approx(2 * (0.08 + 1))   # P per asset
    assert default.const > legacy.const                                       # sufficient is the stronger penalty


def test_rules_agree_when_the_rewards_straddle_zero_widely():
    a = engine.build_qubo(REWARDS, WINDOWS, ACTIONS, "raise_orbit", "legacy")
    b = engine.build_qubo(REWARDS, WINDOWS, ACTIONS, "raise_orbit", "sufficient")
    assert np.array_equal(a.Q, b.Q) and a.const == b.const


def test_legacy_rule_still_has_the_weak_penalty_and_sufficient_fixes_it():
    """The loophole the rule exists to document: legacy is wrong for large similar rewards, kept only for reproduction."""
    rewards = [{"raise_orbit": 1.19, "safe": 1.11}, {"raise_orbit": 1.18, "safe": 1.10}, {"raise_orbit": 1.22, "safe": 1.14}]
    acts = ["raise_orbit", "safe"]
    weak = engine.build_qubo(rewards, [0] * 3, acts, "raise_orbit", "legacy")
    good = engine.build_qubo(rewards, [0] * 3, acts, "raise_orbit", "sufficient")
    assert not engine.solve_exact(weak)["feasible"]
    assert engine.solve_exact(good)["feasible"]


def test_unknown_penalty_rule_is_rejected():
    with pytest.raises(ValueError):
        engine.build_qubo(REWARDS, WINDOWS, ACTIONS, "raise_orbit", penalty_rule="strong")
