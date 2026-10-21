"""Generate the Phase 4 task sets: tasks/train.jsonl (about 1000) and tasks/eval.jsonl (about 50).

Storms are real (Kp record since 1996); spacecraft, altitudes and the Kp -> density mapping are
ILLUSTRATIVE. Every task stores its exact optimum (OR-Tools). Tasks the rule-based policy solves
near-optimally are filtered out. A storm is in only one split; Feb 2022 and May 2024 are eval storms.

    python scripts/generate_tasks.py [n_train] [n_eval]
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

from helioguard.data.noaa import fetch_space_weather, load_space_weather
from helioguard.replays import CACHE_DIR
from helioguard.tasks import DISRUPTIONS, build_dataset, save_tasks


def summarize(tasks):
    n = len(tasks)
    return {
        "tasks": n,
        "storms": len({t["storm"]["id"] for t in tasks}),
        "g_scale": dict(sorted(Counter(t["storm"]["g_scale"] for t in tasks).items())),
        "with_launch": sum(bool(t["launch"]) for t in tasks),
        "needs_qaoa": sum(t["needs_qaoa"] for t in tasks),
        "assets_per_task": round(sum(len(t["fleet"]) for t in tasks) / n, 2),
        "disruptions": {d: sum(any(x["type"] == d for x in t["disruptions"]) for t in tasks) for d in DISRUPTIONS},
        "mean_rule_based_final": round(sum(t["baselines"]["rule_final"] for t in tasks) / n, 3),
        "mean_random_final": round(sum(t["baselines"]["random_final"] for t in tasks) / n, 3),
        "optimum_status": dict(Counter(t["optimal"]["status"] for t in tasks)),
    }


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    rows = load_space_weather(fetch_space_weather(CACHE_DIR))
    train, evals = build_dataset(rows, n_train, n_eval, seed=0, per_storm_train=5, per_storm_eval=2,
                                 eval_storm_dates=["2022-02-03", "2024-05-10"],
                                 workers=max(1, (os.cpu_count() or 2) - 2))
    shared = {t["storm"]["id"] for t in train} & {t["storm"]["id"] for t in evals}
    assert not shared, f"storms in both splits: {shared}"
    out = Path("tasks")
    out.mkdir(exist_ok=True)
    save_tasks(out / "train.jsonl", train)
    save_tasks(out / "eval.jsonl", evals)
    summary = {"train": summarize(train), "eval": summarize(evals), "storms_shared_between_splits": 0,
               "eval_includes": sorted(d for d in ("2022-02-03", "2024-05-10")
                                       if any(t["storm"]["onset"][:10] in (d, d[:-2] + str(int(d[-2:]) + 1).zfill(2))
                                              for t in evals))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
