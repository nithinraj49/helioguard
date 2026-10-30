"""Phase 7: aggregate the cached runs into the tables and gate verdicts registered in docs/phase7_gates.md.

    python scripts/phase7_report.py            # writes results/phase7/summary.json and results/phase7/results.md

Instance-clustered percentile bootstrap (10,000 resamples, seed 0) over instances; seeds are averaged within an
instance first. Spacecraft values are ILLUSTRATIVE. QAOA ran on a classical simulator; no quantum advantage is claimed.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verispace_engine import qaoa_bench as qb  # noqa: E402

OUT = ROOT / "results" / "phase7"
CACHE = OUT / "cache"
FAMILY_NAME = {"B": "B (2 actions, requested sizes)", "A": "A (3 actions, native fleet)"}


def load():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CACHE.glob("*.json"))]
    base = {}                                            # (family, n, inst) -> baseline (identical across arms)
    runs = defaultdict(lambda: defaultdict(list))        # (arm, fam, n, p, opt, init) -> metric -> [per-instance mean over seeds]
    inst_ids = defaultdict(list)
    for rec in recs:
        key = (rec["family"], rec["n_qubits"], rec["instance"])
        base[key] = rec["baseline"]
        by_cfg = defaultdict(list)
        for r in rec["runs"]:
            by_cfg[(r["layers"], r["optimiser"], r["init"])].append(r)
        for (p, o, i), rs in by_cfg.items():
            cell = runs[(rec["arm"], rec["family"], rec["n_qubits"], p, o, i)]
            for m in ("optimum_probability", "approx_ratio", "top64_hit", "best_of_64_shots", "evaluations",
                      "solve_time_s", "feasible_probability", "top64_feasible"):
                if m in rs[0]:
                    cell[m].append(float(np.mean([float(r[m]) for r in rs])))
            inst_ids[(rec["arm"], rec["family"], rec["n_qubits"], p, o, i)].append(rec["instance"])
    return base, runs, inst_ids


def ci(v):
    lo, hi = qb.bootstrap_ci(v)
    return float(np.mean(v)), lo, hi


def fmt(v, d=3):
    m, lo, hi = ci(v)
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def sort_by_instance(runs, ids, key, metric):
    order = np.argsort(ids[key])
    return np.asarray(runs[key][metric])[order]


def main() -> None:
    base, runs, ids = load()
    if not base:
        sys.exit("no cached runs under results/phase7/cache/")
    summary: dict = {"g0": {}, "cells": {}, "baselines": {}, "g3": {}, "g4": {}, "g5": {}, "arms_vs": {}}
    md: list[str] = ["# Phase 7 results: QAOA scaling benchmark",
                     "",
                     "ILLUSTRATIVE spacecraft values; classical statevector simulator; **no quantum advantage is claimed** (G6).",
                     "Mean over 10 instances (5 for 18 and 20 qubits), each averaged over 3 optimiser seeds; brackets are 95% "
                     "percentile-bootstrap CIs over instances. Gates: `docs/phase7_gates.md`.", ""]

    # --- G0 and per-size classical baselines -----------------------------------------------------
    keys = sorted(base)
    g0 = {"instances": len(keys),
          "exact_feasible_all": all(b["exact_feasible"] for b in base.values()),
          "exact_matches_landscape_all": all(b["exact_matches_landscape"] for b in base.values()),
          "ortools_within_rounding_all": all(b["ortools_optimal"] for b in base.values()),
          "ortools_max_gap": max(b["ortools_gap"] for b in base.values()),
          "ortools_instances_with_gap_over_1e-9": sum(b["ortools_gap"] > 1e-9 for b in base.values())}
    summary["g0"] = g0
    md += ["## G0 sanity", "", f"- instances: {g0['instances']}; exact optimum feasible on all: **{g0['exact_feasible_all']}**; "
           f"exact equals landscape minimum: {g0['exact_matches_landscape_all']}",
           f"- OR-Tools within its reward-rounding bound on all: **{g0['ortools_within_rounding_all']}**; largest gap to exact "
           f"{g0['ortools_max_gap']:.2e}; instances where it trails exact at all: {g0['ortools_instances_with_gap_over_1e-9']}",
           "- simulator fidelity (1e-9 vs PennyLane) and the engine/harness agreement are pinned by `tests/test_qaoa_bench.py`.", ""]

    def uni(fam, n, budget_map=None):
        insts = [k for k in keys if k[0] == fam and k[1] == n]
        p_uni = np.array([base[k]["n_optimal"] / 2 ** n for k in insts])
        return insts, p_uni

    # --- T1: engine arm at p = 2 vs classical ----------------------------------------------------
    for p_main in (2,):
        md += [f"## Engine arm (`solve_qaoa`, COBYLA, random start), p = {p_main}, with classical baselines", ""]
        for fam in ("B", "A"):
            md += [f"### Family {FAMILY_NAME[fam]}", "",
                   "| qubits | P(opt) QAOA | P(opt) uniform | enrichment | approx. ratio | top-64 hit | 64 shots hit | uniform 64 hit | "
                   "uniform hit, same evals | G1 | G2a | G2b | evals | OR-Tools optimal | OR-Tools ms | exact ms |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            sizes = sorted({k[1] for k in keys if k[0] == fam and ("engine", fam, k[1], p_main, "COBYLA", "random") in runs})
            for n in sizes:
                key = ("engine", fam, n, p_main, "COBYLA", "random")
                insts, p_uni = uni(fam, n)
                popt = sort_by_instance(runs, ids, key, "optimum_probability")
                hit = sort_by_instance(runs, ids, key, "top64_hit")
                shots = sort_by_instance(runs, ids, key, "best_of_64_shots")
                ar = sort_by_instance(runs, ids, key, "approx_ratio")
                ev = sort_by_instance(runs, ids, key, "evaluations")
                u64 = np.array([qb.shots_hit_probability(x, 64) for x in p_uni])
                ubud = np.array([qb.shots_hit_probability(x, int(e)) for x, e in zip(p_uni, ev)])
                m_p, lo_p, hi_p = ci(popt)
                m_h, lo_h, hi_h = ci(hit)
                g1 = lo_p > p_uni.mean()
                g2a = m_h >= 0.95 and lo_h >= 0.90
                g2b = m_p >= 0.5
                ort_ok = np.mean([base[k]["ortools_optimal"] for k in insts])
                ort_ms = 1e3 * np.mean([base[k]["ortools_time_s"] for k in insts])
                ex_ms = 1e3 * np.mean([base[k].get("exact_time_s", np.nan) for k in insts])
                summary["cells"][f"engine|{fam}|{n}|p{p_main}"] = {
                    "p_opt": ci(popt), "p_uniform": float(p_uni.mean()), "approx_ratio": ci(ar), "top64_hit": ci(hit),
                    "shots64_hit": ci(shots), "uniform_shots64_hit": float(u64.mean()), "uniform_budget_hit": float(ubud.mean()),
                    "G1": bool(g1), "G2a": bool(g2a), "G2b": bool(g2b), "evals": float(ev.mean()),
                    "ortools_optimal_rate": float(ort_ok), "ortools_ms": float(ort_ms), "exact_ms": float(ex_ms)}
                md.append(f"| {n} | {fmt(popt)} | {p_uni.mean():.2e} | {m_p / p_uni.mean():.0f}x | {fmt(ar)} | {fmt(hit, 2)} | "
                          f"{fmt(shots, 2)} | {u64.mean():.3f} | {ubud.mean():.3f} | {'pass' if g1 else 'FAIL'} | "
                          f"{'pass' if g2a else 'FAIL'} | {'pass' if g2b else 'FAIL'} | {ev.mean():.0f} | {ort_ok:.0%} | "
                          f"{ort_ms:.1f} | {ex_ms:.1f} |")
            md.append("")
        md += ["Wall times in the parallel run are contended; see the serial timing table below.", ""]

    # --- T2: depth (G1 per cell, G4 paired) ----------------------------------------------------------
    md += ["## Depth p = 1 to 4 (engine arm): single-sample P(optimum)", ""]
    for fam in ("B", "A"):
        sizes = sorted({k[1] for k in keys if k[0] == fam and ("engine", fam, k[1], 1, "COBYLA", "random") in runs})
        md += [f"### Family {FAMILY_NAME[fam]}", "", "| qubits | uniform | p=1 | p=2 | p=3 | p=4 | G1 (p=1..4) | G4: 1 to 2 | 2 to 3 | 3 to 4 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for n in sizes:
            _, p_uni = uni(fam, n)
            cols, g1s, g4 = [], [], []
            vals = {p: sort_by_instance(runs, ids, ("engine", fam, n, p, "COBYLA", "random"), "optimum_probability") for p in (1, 2, 3, 4)}
            for p in (1, 2, 3, 4):
                cols.append(fmt(vals[p]))
                g1s.append("pass" if ci(vals[p])[1] > p_uni.mean() else "FAIL")
            for p in (1, 2, 3):
                d = vals[p + 1] - vals[p]
                lo, hi = qb.bootstrap_ci(d)
                v = qb.verdict(lo, hi)
                g4.append(f"{v} ({d.mean():+.3f})")
                summary["g4"][f"{fam}|{n}|p{p}->p{p + 1}"] = {"diff": float(d.mean()), "ci": [lo, hi], "verdict": v}
            md.append(f"| {n} | {p_uni.mean():.2e} | " + " | ".join(cols) + f" | {'/'.join(g1s)} | " + " | ".join(g4) + " |")
        md.append("")

    # --- T4: scaling slopes (G3) ---------------------------------------------------------------------
    md += ["## G3 scaling: slope of log10 P(optimum) per added qubit (negative = decays)", "",
           "| family | arm | p | slope per qubit | 95% CI | verdict | sizes |", "|---|---|---|---|---|---|---|"]
    for arm, opt, init in (("engine", "COBYLA", "random"), ("harness", "COBYLA", "random")):
        for fam in ("B", "A"):
            for p in (1, 2, 3, 4):
                sizes = sorted({k[2] for k in runs if k[0] == arm and k[1] == fam and k[3] == p and k[4] == opt and k[5] == init})
                if len(sizes) < 3:
                    continue
                per = [sort_by_instance(runs, ids, (arm, fam, n, p, opt, init), "optimum_probability") for n in sizes]
                s, lo, hi = qb.log_slope(sizes, per)
                v = "decays with size" if hi < 0 else ("grows with size" if lo > 0 else "no detectable trend")
                summary["g3"][f"{arm}|{fam}|p{p}"] = {"slope": s, "ci": [lo, hi], "verdict": v, "sizes": sizes}
                md.append(f"| {fam} | {arm} | {p} | {s:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {v} | {sizes[0]}-{sizes[-1]} |")
    md.append("")

    # --- T3/T5: optimiser and init variants (harness arm) -------------------------------------------
    md += ["## G5 optimiser and initialisation (harness arm, same budget): P(optimum), paired difference vs COBYLA/random", ""]
    for fam in ("B", "A"):
        for p in (2, 4):
            sizes = sorted({k[2] for k in runs if k[0] == "harness" and k[1] == fam and k[3] == p})
            if not sizes:
                continue
            variants = [(o, i) for o in qb.OPTIMISERS for i in qb.INITS]
            md += [f"### Family {fam}, p = {p}", "", "| qubits | " + " | ".join(f"{o}/{i}" for o, i in variants) + " |",
                   "|---|" + "---|" * len(variants)]
            for n in sizes:
                ref = sort_by_instance(runs, ids, ("harness", fam, n, p, "COBYLA", "random"), "optimum_probability")
                cells = []
                for o, i in variants:
                    v = sort_by_instance(runs, ids, ("harness", fam, n, p, o, i), "optimum_probability")
                    if (o, i) == ("COBYLA", "random"):
                        cells.append(f"{v.mean():.3f} (ref)")
                        continue
                    d = v - ref
                    lo, hi = qb.bootstrap_ci(d)
                    verd = qb.verdict(lo, hi)
                    summary["g5"][f"{fam}|{n}|p{p}|{o}/{i}"] = {"mean": float(v.mean()), "diff": float(d.mean()), "ci": [lo, hi], "verdict": verd}
                    cells.append(f"{v.mean():.3f} ({d.mean():+.3f}, {verd})")
                md.append(f"| {n} | " + " | ".join(cells) + " |")
            md.append("")
    counts = defaultdict(lambda: defaultdict(int))
    for k, v in summary["g5"].items():
        counts[k.split("|")[-1]][v["verdict"]] += 1
    md += ["G5 verdict counts over all (family, size, p in {2, 4}) cells:", "",
           "| variant | helps | hurts | no detectable change |", "|---|---|---|---|"]
    for var, c in sorted(counts.items()):
        md.append(f"| {var} | {c['helps']} | {c['hurts']} | {c['no detectable change']} |")
    md.append("")

    # --- engine vs harness, same COBYLA/random config -----------------------------------------------
    md += ["## Engine (PennyLane) vs harness (numpy), both COBYLA/random: paired difference in P(optimum)", "",
           "| family | qubits | p | engine | harness | difference [95% CI] |", "|---|---|---|---|---|---|"]
    for fam in ("B", "A"):
        for n in sorted({k[2] for k in runs if k[0] == "engine" and k[1] == fam}):
            for p in (1, 2, 3, 4):
                ke, kh = ("engine", fam, n, p, "COBYLA", "random"), ("harness", fam, n, p, "COBYLA", "random")
                if ke in runs and kh in runs:
                    a, b = sort_by_instance(runs, ids, ke, "optimum_probability"), sort_by_instance(runs, ids, kh, "optimum_probability")
                    lo, hi = qb.bootstrap_ci(b - a)
                    summary["arms_vs"][f"{fam}|{n}|p{p}"] = {"diff": float((b - a).mean()), "ci": [lo, hi]}
                    md.append(f"| {fam} | {n} | {p} | {a.mean():.3f} | {b.mean():.3f} | {(b - a).mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    md.append("")

    # --- extension and timing -------------------------------------------------------------------------
    ext = sorted({k[2] for k in runs if k[0] == "extension"})
    if ext:
        md += ["## Extension: 18 and 20 qubits (harness arm, p = 2, COBYLA/random, 5 instances)", "",
               "| qubits | P(opt) | uniform | approx. ratio | top-64 hit | evals | mean run s (contended) |", "|---|---|---|---|---|---|---|"]
        for n in ext:
            key = ("extension", "B", n, 2, "COBYLA", "random")
            _, p_uni = uni("B", n)
            md.append(f"| {n} | {fmt(sort_by_instance(runs, ids, key, 'optimum_probability'))} | {p_uni.mean():.2e} | "
                      f"{fmt(sort_by_instance(runs, ids, key, 'approx_ratio'))} | {fmt(sort_by_instance(runs, ids, key, 'top64_hit'), 2)} | "
                      f"{np.mean(runs[key]['evaluations']):.0f} | {np.mean(runs[key]['solve_time_s']):.0f} |")
        md.append("")
    tpath = OUT / "timing_serial.json"
    if tpath.exists():
        t = json.loads(tpath.read_text(encoding="utf-8"))
        md += ["## Serial timing (nothing else running; one instance per size, p = 2, seed 0)", "",
               "| qubits | engine s (evals) | numpy harness s | exact ms | OR-Tools ms |", "|---|---|---|---|---|"]
        for r in t["serial_runs_p2"]:
            md.append(f"| {r['n_qubits']} | {r['engine_p2_s']:.2f} ({r['engine_evals']}) | {r['harness_p2_s']:.2f} | "
                      f"{1e3 * r['exact_s']:.1f} | {1e3 * r['ortools_s']:.1f} |")
        md += ["", "| qubits | engine s per circuit evaluation (incl. setup) | numpy s per evaluation, p = 2 |", "|---|---|---|"]
        for r in t["per_eval_probes"]:
            md.append(f"| {r['n_qubits']} | {r['engine_s_per_eval_incl_setup']:.3f} | {r['harness_s_per_eval_p2']:.3f} |")
        md.append("")
        summary["timing"] = t
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
