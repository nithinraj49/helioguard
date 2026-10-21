"""Score the scripted policies on the Phase 4 task sets through the environment.

Writes tasks/baselines.json. Final reward is the end-of-episode reward from the exact-solver gap;
return adds the small step rewards and tool costs.

    python scripts/evaluate_baselines.py
"""
import json
import random
import statistics
from pathlib import Path

from helioguard.env import HelioGuardEnv
from helioguard.policies import (AlwaysRaisePolicy, EndlessToolCallsPolicy, NeverCallToolsPolicy,
                                 NeverSubmitPolicy, OraclePolicy, RandomPolicy, RuleBasedPolicy,
                                 VaguePredictionsPolicy, run_episode)
from helioguard.tasks import load_tasks

POLICIES = {
    "oracle (reads the stored optimum)": OraclePolicy,
    "rule-based": RuleBasedPolicy,
    "random": None,
    "no tools (rule-based plan)": NeverCallToolsPolicy,
    "always raise": AlwaysRaisePolicy,
    "vague claims": VaguePredictionsPolicy,
    "endless tool calls": EndlessToolCallsPolicy,
    "never submits": NeverSubmitPolicy,
}


def evaluate(tasks, name, factory):
    env = HelioGuardEnv(tasks)
    rng = random.Random(0)
    returns, finals, solved = [], [], 0
    for t in tasks:
        policy = RandomPolicy(rng) if factory is None else factory()
        r = run_episode(env, policy, t["id"])
        returns.append(r["return"])
        f = r["final"]["final_reward"] if r["final"] else None
        if f is not None:
            finals.append(f)
            solved += f >= 0.9
    return {"policy": name, "episodes": len(tasks), "mean_return": round(statistics.fmean(returns), 3),
            "mean_final_reward": round(statistics.fmean(finals), 3) if finals else None,
            "share_near_optimal": round(solved / len(tasks), 3), "submitted": len(finals)}


if __name__ == "__main__":
    out = {}
    for split in ("train", "eval"):
        tasks = load_tasks(Path("tasks") / f"{split}.jsonl")
        out[split] = [evaluate(tasks, n, f) for n, f in POLICIES.items()]
        print(f"\n{split}: {len(tasks)} tasks")
        for row in out[split]:
            print(f"  {row['policy']:36s} return {row['mean_return']:+.3f}  final {row['mean_final_reward']}  "
                  f"near-optimal {row['share_near_optimal']:.0%}")
    Path("tasks/baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
