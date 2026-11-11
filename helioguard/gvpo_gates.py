"""Pre-registered gates for the GVPO experiment (docs/phase6_gvpo_gates.md, section 7). Constants mirror that text.

    python -m helioguard.gvpo_gates --sft results/phase6/local/sft_1p5b_greedy.jsonl \\
        --precheck results/phase6/gvpo_precheck/summary.json \\
        --arm A=a_s0.jsonl,a_s1.jsonl,a_s2.jsonl --arm B=... --arm C=...

Greedy episode files (one per seed) come from `python -m helioguard.sft_eval --greedy-only ...`. Gates apply to arm A;
the other arms are compared to A and to the SFT result, descriptively. Nothing here changes any Phase 6 gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from verispace_engine.harness.rollout import EpisodeResult
from verispace_engine.harness.stats import bootstrap_ci

from .sft_eval import load_episodes, paired_vs_baseline, per_task
from .sft_traces import OPTIMAL

G0_MIN_INFORMATIVE = 0.40            # share of precheck groups with non-identical final rewards
G1_MIN_VALID = 0.90                  # valid-submission rate, every seed
G3_NEEDED_ABOVE = 0.743              # no-tool plan: lower CI bound of mean final reward must exceed it
G5_SFT_PERCENT_OPTIMAL = 0.24        # SFT greedy percent optimal: the arm's point estimate must exceed it
REFERENCES = {"sft_mean_final_reward": 0.825, "expert_mean_final_reward": 0.948, "no_tool_plan": 0.743}


def _ci(values: Sequence[float]) -> dict[str, float]:
    mean, lo, hi = bootstrap_ci(list(values))
    return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def evaluate_gates(informative_fraction: float, arm_runs: dict[int, list[EpisodeResult]],
                   sft: Sequence[EpisodeResult]) -> dict:
    """Gates G0 to G5 for one arm. `arm_runs` maps training seed -> greedy episodes on the 50 eval tasks."""
    seeds = sorted(arm_runs)
    pooled = [r for s in seeds for r in arm_runs[s]]
    reward = _ci(list(per_task(pooled, lambda r: r.final_reward).values()))
    optimal = _ci(list(per_task(pooled, lambda r: r.valid_submission and r.final_reward >= OPTIMAL).values()))
    paired = paired_vs_baseline(pooled, sft)
    valid = [round(sum(r.valid_submission for r in arm_runs[s]) / len(arm_runs[s]), 4) for s in seeds]
    per_seed = [paired_vs_baseline(arm_runs[s], sft)["mean_difference"] for s in seeds]
    report = {
        "G0": {"pass": bool(informative_fraction >= G0_MIN_INFORMATIVE), "informative_fraction": informative_fraction,
               "needed": G0_MIN_INFORMATIVE},
        "G1": {"pass": bool(all(v >= G1_MIN_VALID for v in valid)), "valid_rate_per_seed": valid, "needed": G1_MIN_VALID},
        "G2": {"pass": bool(paired["ci_low"] > 0), "paired_vs_sft": paired},
        "G3": {"pass": bool(reward["ci_low"] > G3_NEEDED_ABOVE), "mean_final_reward": reward, "needed_above": G3_NEEDED_ABOVE},
        "G4": {"pass": bool(all(d > 0 for d in per_seed)), "per_seed": per_seed, "seeds": seeds},
        "G5": {"pass": bool(optimal["mean"] > G5_SFT_PERCENT_OPTIMAL), "percent_optimal": optimal["mean"],
               "ci": [optimal["ci_low"], optimal["ci_high"]], "needed_above": G5_SFT_PERCENT_OPTIMAL},
        "references": REFERENCES,
    }
    report["all_pass"] = bool(all(report[g]["pass"] for g in ("G0", "G1", "G2", "G3", "G4", "G5")))
    return report


def compare_arms(a_runs: dict[int, list[EpisodeResult]], b_runs: dict[int, list[EpisodeResult]]) -> dict:
    """Secondary, not a gate: paired per-task difference in final reward, arm a minus arm b (seed-averaged)."""
    pool = lambda runs: [r for s in sorted(runs) for r in runs[s]]
    return paired_vs_baseline(pool(a_runs), pool(b_runs))


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sft", required=True, help="SFT greedy episode file (the paired comparator)")
    p.add_argument("--precheck", required=True, help="precheck summary.json (gate G0)")
    p.add_argument("--arm", action="append", required=True, help="NAME=seed0.jsonl,seed1.jsonl,... (first arm is A)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    sft = load_episodes(a.sft)
    frac = json.loads(Path(a.precheck).read_text(encoding="utf-8"))["fraction_informative_final_reward"]
    arms = {}
    for spec in a.arm:
        name, files = spec.split("=", 1)
        arms[name] = {i: load_episodes(f) for i, f in enumerate(files.split(","))}
    first = next(iter(arms))
    out = {"primary_arm": first, "gates": evaluate_gates(frac, arms[first], sft),
           "secondary": {f"{first}_minus_{n}": compare_arms(arms[first], runs) for n, runs in arms.items() if n != first}}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
