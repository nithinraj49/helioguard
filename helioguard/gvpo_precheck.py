"""Precheck for GVPO: how often does a group of 8 sampled attempts have identical rewards? (docs/phase6_gvpo_gates.md, section 4)

GVPO and GRPO both need groups whose rewards differ. This samples `attempts` episodes at temperature 0.7 for each of
100 fixed train tasks with the SFT model, appends every finished group to a JSONL file at once (resumable: finished
groups are skipped), and summarises. Gate G0: at least 40% of groups must have non-identical final rewards.
Run on Colab:  python scripts/gvpo_precheck.py --adapter <SFT adapter folder> --out <dir> [--hub-repo user/name]
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Sequence

from verispace_engine.gvpo import Episode

from .gvpo_gates import G0_MIN_INFORMATIVE

TOL = 1e-9


def select_tasks(task_ids: Sequence[str], n: int = 100, seed: int = 0) -> list[str]:
    """A fixed sample of train tasks (refuses anything that is not a train task id)."""
    bad = [t for t in task_ids if not t.startswith("train")]
    if bad:
        raise ValueError(f"not train tasks: {bad[:3]}")
    return sorted(random.Random(seed).sample(list(task_ids), n))


def load_groups(path: str | Path) -> list[dict]:
    p = Path(path)
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:         # a half-written last line
                continue
    return out


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _std(values: Sequence[float]) -> float:
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def summarize_groups(groups: Sequence[dict]) -> dict:
    n = len(groups)
    if n == 0:
        raise ValueError("no groups")
    ident_final = [_spread(g["final_reward"]) < TOL for g in groups]
    ident_total = [_spread(g["total_return"]) < TOL for g in groups]
    only_step = [f and not t for f, t in zip(ident_final, ident_total)]
    samples = sum(len(g["final_reward"]) for g in groups)
    informative = 1 - sum(ident_final) / n
    # does step+final centre a group differently from outcome-only? (largest per-member difference of centred rewards)
    centre_diff = []
    for g in groups:
        cf = [x - sum(g["final_reward"]) / len(g["final_reward"]) for x in g["final_reward"]]
        ct = [x - sum(g["total_return"]) / len(g["total_return"]) for x in g["total_return"]]
        centre_diff.append(max(abs(a - b) for a, b in zip(cf, ct)))
    return {
        "groups": n, "samples": samples, "attempts_per_group": samples // n,
        "identical_final_reward": sum(ident_final), "fraction_identical_final_reward": sum(ident_final) / n,
        "fraction_informative_final_reward": informative,
        "identical_total_return": sum(ident_total), "fraction_identical_total_return": sum(ident_total) / n,
        "groups_where_only_step_rewards_differ": sum(only_step),
        "mean_within_group_std_final_reward": round(sum(_std(g["final_reward"]) for g in groups) / n, 4),
        "mean_within_group_std_total_return": round(sum(_std(g["total_return"]) for g in groups) / n, 4),
        "mean_final_reward": round(sum(sum(g["final_reward"]) for g in groups) / samples, 4),
        "valid_submission_rate": sum(sum(g["valid"]) for g in groups) / samples,
        "groups_where_centred_rewards_differ_by_over_0.05": sum(c > 0.05 for c in centre_diff),
        "gate_G0_threshold": G0_MIN_INFORMATIVE, "gate_G0_pass": bool(informative >= G0_MIN_INFORMATIVE),
    }


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_precheck(sample: Callable[[str, int], Sequence[Episode]], task_ids: Sequence[str], path: str | Path,
                 attempts: int = 8, hub=None, upload_every: int = 10) -> dict:
    """Sample every group not yet in `path`, append each as it finishes, upload every `upload_every` groups, summarise."""
    path = Path(path)
    done = {g["task_id"] for g in load_groups(path)}
    fresh = 0
    for tid in task_ids:
        if tid in done:
            continue
        eps = sample(tid, attempts)
        _append(path, {"task_id": tid, "final_reward": [e.final_reward for e in eps],
                       "total_return": [e.total_return for e in eps], "valid": [bool(e.valid) for e in eps]})
        fresh += 1
        print(f"[precheck] {tid}: final rewards {[round(e.final_reward, 2) for e in eps]} "
              f"({len(done) + fresh}/{len(task_ids)} groups saved)", flush=True)
        if hub is not None and fresh % upload_every == 0:
            hub.upload({f"precheck/{path.name}": path}, f"precheck {len(done) + fresh}/{len(task_ids)}")
    summary = summarize_groups(load_groups(path))
    (path.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if hub is not None:
        hub.upload({f"precheck/{path.name}": path, "precheck/summary.json": path.parent / "summary.json"}, "precheck summary")
    return summary
