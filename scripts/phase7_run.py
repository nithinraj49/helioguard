"""Phase 7 QAOA scaling benchmark: run the grid registered in docs/phase7_gates.md and cache raw results.

    python scripts/phase7_run.py run [--workers 8]     # engine arm + harness arm + extension, resumable
    python scripts/phase7_run.py timing                # serial timing table of record (run it alone)

One cache file per (arm, family, size, instance) under results/phase7/cache/ (git-ignored). Re-running skips
finished units. Spacecraft and storm values are ILLUSTRATIVE; QAOA runs on a classical simulator and no
quantum advantage is claimed.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helioguard.qaoa_instances import EXTENSION_SIZES, FAMILIES, make_instance, make_qubo  # noqa: E402
from verispace_engine import qaoa_bench as qb  # noqa: E402

OUT = ROOT / "results" / "phase7"
CACHE = OUT / "cache"
N_INSTANCES, SEEDS, DEPTHS = 10, (0, 1, 2), (1, 2, 3, 4)
EXT_INSTANCES = 5
VARIANTS = [(o, i) for o in qb.OPTIMISERS for i in qb.INITS]


def units() -> list[tuple[str, str, int, int]]:
    """(arm, family, qubits, instance); biggest first so the long units start early."""
    out = []
    for fam, spec in FAMILIES.items():
        for n in spec["sizes"]:
            for k in range(N_INSTANCES):
                out += [("engine", fam, n, k), ("harness", fam, n, k)]
    for n in EXTENSION_SIZES:
        out += [("extension", "B", n, k) for k in range(EXT_INSTANCES)]
    return sorted(out, key=lambda u: (-u[2], u[0], u[1], u[3]))


def cache_path(u: tuple[str, str, int, int]) -> Path:
    return CACHE / f"{u[0]}_{u[1]}{u[2]:02d}_{u[3]:02d}.json"


def baseline(family: str, n: int, k: int, land: qb.Landscape) -> dict:
    """Classical references for one instance: exact (checked against the landscape) and OR-Tools."""
    from verispace_engine.qubo import solve_exact, solve_ortools
    rewards, windows, actions = make_instance(family, n, k)
    qubo = make_qubo(family, n, k)
    ort = solve_ortools(rewards, windows, actions, actions[0])
    out = {"n_optimal": int(land.optimal.sum()), "e_min": land.e_min, "uniform_mean": land.uniform_mean,
           "feasible_fraction": float(land.feasible.mean()), "ortools_objective": ort["objective"],
           "ortools_time_s": ort["solve_time_s"], "ortools_status": ort["status"]}
    if n <= 16:                                              # the engine's exact solver is cheap up to here
        ex = solve_exact(qubo)
        out |= {"exact_objective": ex["objective"], "exact_energy": ex["energy"], "exact_time_s": ex["solve_time_s"],
                "exact_feasible": ex["feasible"], "exact_matches_landscape": abs(ex["energy"] - land.e_min) < 1e-9}
    else:                                                    # decode the argmin of the landscape instead
        import numpy as np
        from verispace_engine.qubo import decode
        bits = ((np.int64(int(land.energies.argmin())) >> np.arange(n - 1, -1, -1)) & 1).astype(float)
        dec = decode(qubo, bits)
        out |= {"exact_objective": dec["objective"], "exact_energy": land.e_min, "exact_feasible": dec["feasible"],
                "exact_matches_landscape": True}
    gap = out["exact_objective"] - ort["objective"]                 # OR-Tools rounds rewards to 1e-3 (Amendment 1)
    out["ortools_gap"] = gap
    out["ortools_optimal"] = -1e-9 <= gap <= 0.5e-3 * len(rewards)
    return out


def run_unit(u: tuple[str, str, int, int]) -> str:
    arm, fam, n, k = u
    t0 = time.perf_counter()
    qubo = make_qubo(fam, n, k)
    land = qb.Landscape.of(qubo)
    runs: list[dict] = []
    if arm == "engine":
        from verispace_engine.qaoa import solve_qaoa
        for p in DEPTHS:
            for seed in SEEDS:
                r = solve_qaoa(qubo, layers=p, seed=seed)
                runs.append({"layers": p, "optimiser": "COBYLA", "init": "random", "seed": seed,
                             "optimum_probability": r["optimum_probability"], "approx_ratio": r["approx_ratio"],
                             "expected_energy": r["expected_energy"], "top64_energy": r["energy"],
                             "top64_hit": bool(r["energy"] <= land.e_min + qb.TOL),
                             "top64_feasible": bool(r["feasible"]), "evaluations": r["evaluations"],
                             "solve_time_s": r["solve_time_s"],
                             "best_of_64_shots": qb.shots_hit_probability(r["optimum_probability"], 64)})
    else:
        grid = [(p, o, i, s) for p in ((2,) if arm == "extension" else DEPTHS)
                for (o, i) in (VARIANTS[:1] if arm == "extension" else VARIANTS) for s in SEEDS]
        for p, o, i, s in grid:
            runs.append(qb.run_variant(qubo, p, o, i, s, landscape=land))
    rec = {"arm": arm, "family": fam, "n_qubits": n, "instance": k, "baseline": baseline(fam, n, k, land),
           "runs": runs, "unit_wall_s": time.perf_counter() - t0}
    tmp = cache_path(u).with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(cache_path(u))
    return f"{arm} {fam}{n} #{k}: {len(runs)} runs in {rec['unit_wall_s']:.0f}s"


def warm_up() -> None:
    """Import and run PennyLane once so first-call setup is not billed to the first benchmark unit."""
    from verispace_engine.qaoa import solve_qaoa
    solve_qaoa(make_qubo("B", 6, 0), layers=1, restarts=1, maxiter=3)


def cmd_run(workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [u for u in units() if not cache_path(u).exists()]
    print(f"{len(todo)} of {len(units())} units to run on {workers} workers", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(workers, initializer=warm_up) as pool:
        futs = [pool.submit(run_unit, u) for u in todo]
        for f in as_completed(futs):
            done += 1
            print(f"[{done}/{len(todo)} {time.time() - t0:5.0f}s] {f.result()}", flush=True)


def cmd_timing() -> None:
    """Serial timings (nothing else running): engine, numpy harness, exact, OR-Tools; plus engine cost per circuit
    evaluation at 16, 18 and 20 qubits to size the simulator limit."""
    import numpy as np
    from verispace_engine.qaoa import solve_qaoa
    from verispace_engine.qubo import solve_exact, solve_ortools
    warm_up()
    rows = []
    for n in FAMILIES["B"]["sizes"]:
        qubo, (rw, wi, ac) = make_qubo("B", n, 0), make_instance("B", n, 0)
        eng = solve_qaoa(qubo, layers=2, seed=0)
        har = qb.run_variant(qubo, 2, "COBYLA", "random", 0)
        rows.append({"n_qubits": n, "engine_p2_s": eng["solve_time_s"], "engine_evals": eng["evaluations"],
                     "harness_p2_s": har["solve_time_s"], "harness_evals": har["evaluations"],
                     "exact_s": solve_exact(qubo)["solve_time_s"],
                     "ortools_s": solve_ortools(rw, wi, ac, ac[0])["solve_time_s"]})
        print(rows[-1], flush=True)
    probes = []
    for n in (16, 18, 20):
        qubo = make_qubo("B", n, 0)
        t0 = time.perf_counter()
        r = solve_qaoa(qubo, layers=2, seed=0, restarts=1, maxiter=3)
        engine_total = time.perf_counter() - t0
        sim = qb.FastQaoa(qubo, 2)
        x = np.array([0.3, 0.4, 0.5, 0.6])
        t1 = time.perf_counter()
        for _ in range(3):
            sim.probabilities(x)
        probes.append({"n_qubits": n, "engine_s_per_eval_incl_setup": engine_total / max(r["evaluations"], 1),
                       "engine_evals": r["evaluations"], "engine_total_s": engine_total,
                       "harness_s_per_eval_p2": (time.perf_counter() - t1) / 3})
        print(probes[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing_serial.json").write_text(json.dumps({"serial_runs_p2": rows, "per_eval_probes": probes}, indent=1),
                                            encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "timing"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cmd_run(a.workers) if a.cmd == "run" else cmd_timing()
