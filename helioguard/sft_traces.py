"""SFT traces from the tool-informed expert, serialised exactly as the harness prompts a model.

Messages are built with the same functions the rollout runner uses (system_prompt, briefing_message,
observation_message with the HelioGuard compactor), and every assistant turn is the compact JSON of
the action, which the strict parser accepts. The expert sees the same compacted observations the model
will see. Only TRAIN tasks are ever used; eval tasks and eval storms are refused.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from verispace_engine.env_base import StepOutcome
from verispace_engine.harness.prompts import briefing_message, observation_message, system_prompt

from .env import HelioGuardEnv
from .harness_env import ALLOWED_TYPES, EXTRA_ACTIONS, compact_observation
from .policies import ToolInformedPolicy
from .tasks import load_tasks

THRESHOLD = 0.90               # keep traces whose final reward is at least this (the "near-optimal" line)
OPTIMAL = 0.999                # exact CP-SAT optimum scores 1.0


def action_text(action: dict) -> str:
    """The assistant turn: one compact JSON object, nothing else."""
    return json.dumps(action, separators=(",", ":"))


def record_trace(env: HelioGuardEnv, task_id: str, policy=None) -> dict[str, Any]:
    """Play the expert through the environment and keep what the model would see and say."""
    policy = policy or ToolInformedPolicy()
    briefing = env.reset(task_id=task_id)
    task = env.task
    messages = [{"role": "system", "content": system_prompt(env.tool_specs(), EXTRA_ACTIONS)},
                {"role": "user", "content": briefing_message(briefing)}]
    gen = policy.actions(task)
    action = next(gen)
    step_rewards: list[float] = []
    final = None
    while True:
        messages.append({"role": "assistant", "content": action_text(action)})
        out = env.step(action)
        step_rewards.append(out.reward)
        if out.done:
            final = out.info.get("final")
            break
        messages.append({"role": "user",
                         "content": observation_message(out.observation, compactor=compact_observation)})
        seen = StepOutcome(compact_observation(out.observation), out.reward, out.done, out.info)
        try:
            action = gen.send(seen)
        except StopIteration:
            break
    valid = bool(final) and final.get("error") is None
    return {"task_id": task_id, "storm_id": task["storm"]["id"], "split": task["split"],
            "disruptions": sorted({d["type"] for d in task["disruptions"]}),
            "n_assets": len(task["fleet"]), "has_launch": bool(task["launch"]),
            "messages": messages, "step_rewards": step_rewards, "valid": valid,
            "final_reward": final["final_reward"] if valid else None,
            "tool_calls": env.executed_tool_calls, "optimal": bool(valid and final["final_reward"] >= OPTIMAL)}


def _job(args: tuple[Sequence[dict], str]) -> dict[str, Any]:
    tasks, task_id = args
    return record_trace(HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20), task_id)


def train_tasks(path: str | Path = "tasks/train.jsonl", eval_path: str | Path = "tasks/eval.jsonl") -> list[dict]:
    """The train tasks, after checking that none is an eval task and no storm is in both splits."""
    tasks = load_tasks(Path(path))
    eval_tasks = load_tasks(Path(eval_path))
    bad = [t["id"] for t in tasks if t["split"] != "train"]
    if bad:
        raise ValueError(f"not train tasks: {bad[:3]}")
    if {t["id"] for t in tasks} & {t["id"] for t in eval_tasks}:
        raise ValueError("task ids shared between train and eval")
    if {t["storm"]["id"] for t in tasks} & {t["storm"]["id"] for t in eval_tasks}:
        raise ValueError("storms shared between train and eval")
    return tasks


def generate_traces(tasks: Sequence[dict], threshold: float = THRESHOLD, workers: int = 1
                    ) -> tuple[list[dict], list[dict]]:
    """Returns (kept, all). Refuses any task that is not from the train split."""
    if any(t["split"] != "train" for t in tasks):
        raise ValueError("SFT traces may only come from train tasks")
    jobs = [(list(tasks), t["id"]) for t in tasks]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            everything = list(pool.map(_job, jobs, chunksize=8))
    else:
        env = HelioGuardEnv(list(tasks), max_tool_calls=10, max_turns=20)
        everything = [record_trace(env, t["id"]) for t in tasks]
    kept = [t for t in everything if t["valid"] and t["final_reward"] >= threshold]
    return kept, everything


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def data_card(kept: Sequence[dict], everything: Sequence[dict], threshold: float, seed: int,
              token_counts: Sequence[int] | None = None) -> dict[str, Any]:
    """Counts, disruption mix, reward distribution. Generation is deterministic (no randomness);
    `seed` is the shuffle and training seed recorded for reproducibility."""
    rewards = [t["final_reward"] for t in kept]
    all_rewards = [t["final_reward"] for t in everything if t["valid"]]

    def bins(values):
        edges = [(-1.0, 0.0), (0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.999), (0.999, 1.01)]
        return {f"[{lo}, {min(hi, 1.0)})" if hi < 1.0 else f"[{lo}, 1.0]": sum(lo <= v < hi for v in values)
                for lo, hi in edges}

    card = {
        "source": "tool-informed expert (helioguard.policies.ToolInformedPolicy) on train tasks only",
        "threshold_final_reward": threshold, "optimal_means_final_reward_at_least": OPTIMAL, "seed": seed,
        "generation_is_deterministic": True,
        "tasks_in": len(everything), "valid_submissions": sum(t["valid"] for t in everything),
        "traces_kept": len(kept), "kept_share": round(len(kept) / max(1, len(everything)), 3),
        "kept_optimal": sum(t["optimal"] for t in kept),
        "kept_non_optimal": sum(not t["optimal"] for t in kept),
        "kept_final_reward": {"mean": round(statistics.fmean(rewards), 4) if rewards else None,
                              "min": round(min(rewards), 4) if rewards else None,
                              "histogram": bins(rewards)},
        "all_final_reward_histogram": bins(all_rewards),
        "dropped_below_threshold": sum(t["valid"] and t["final_reward"] < threshold for t in everything),
        "disruption_mix_kept": dict(Counter(d for t in kept for d in t["disruptions"])),
        "disruption_mix_all": dict(Counter(d for t in everything for d in t["disruptions"])),
        "tasks_with_no_disruption_kept": sum(not t["disruptions"] for t in kept),
        "with_launch_kept": sum(t["has_launch"] for t in kept),
        "assets_per_task_kept": round(statistics.fmean(t["n_assets"] for t in kept), 2) if kept else None,
        "tool_calls_per_trace_kept": round(statistics.fmean(t["tool_calls"] for t in kept), 2) if kept else None,
        "storms_kept": len({t["storm_id"] for t in kept}),
        "limitations": ["traces below the threshold are dropped, which selects tasks the expert solves "
                        "(easier tasks are over-represented)",
                        "the expert never makes claims and never recovers from errors, so the model sees no "
                        "error or recovery examples"],
    }
    if token_counts:
        card["tokens_per_trace"] = {"mean": round(statistics.fmean(token_counts)), "max": max(token_counts),
                                    "min": min(token_counts)}
    return card
