"""Rollout runner: N attempts per task against any EnvBase-style environment, with JSONL logging.

The environment enforces the turn and tool-call caps. The runner adds one rule: a reply the strict
parser rejects ends the episode as a failure (reward = cfg.malformed_reward). Nothing is repaired.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .backends import ModelBackend
from .parser import ActionParseError, parse_action
from .prompts import briefing_message, observation_message, system_prompt


@dataclass
class RolloutConfig:
    attempts: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.7
    seed: int = 0
    allowed_types: tuple[str, ...] = ("tool_call", "submit_decision")
    extra_actions: tuple[str, ...] = ()           # domain action lines shown in the system prompt
    malformed_reward: float = -0.5
    failure_final_reward: float = -0.5            # final reward credited to an episode with no valid submission


@dataclass
class EpisodeResult:
    task_id: str
    attempt: int
    total_return: float
    final_reward: float                           # failure episodes carry cfg.failure_final_reward
    valid_submission: bool
    tool_calls: int
    turns: int
    failure: str | None                           # None, malformed:<kind>, never_submits, rejected_submission
    step_rewards: list[float] = field(default_factory=list)
    seconds: float = 0.0


class JsonlLog:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")


def rollout_task(env, task_id: str, backend: ModelBackend, cfg: RolloutConfig, attempt: int = 0,
                 log: JsonlLog | None = None) -> EpisodeResult:
    log = log or JsonlLog(None)
    started = time.time()
    briefing = env.reset(task_id=task_id)
    messages = [{"role": "system", "content": system_prompt(env.tool_specs(), cfg.extra_actions)},
                {"role": "user", "content": briefing_message(briefing)}]
    tool_names = set(env.tool_names())
    base = {"model": backend.name, "task_id": task_id, "attempt": attempt}
    total, turns, steps = 0.0, 0, []
    failure: str | None = None
    final: dict | None = None
    for turn in range(1, getattr(env, "max_turns", 20) + 2):      # the env times the episode out first
        turns = turn
        raw = backend.generate(messages, cfg.max_new_tokens, cfg.temperature, cfg.seed + attempt)
        try:
            action = parse_action(raw, cfg.allowed_types, tool_names)
        except ActionParseError as exc:
            total += cfg.malformed_reward
            steps.append(cfg.malformed_reward)
            failure = f"malformed:{exc.kind}"
            log.write({**base, "event": "turn", "turn": turn, "raw_output": raw, "parse_error": str(exc),
                       "failure": failure, "reward": cfg.malformed_reward, "done": True})
            break
        out = env.step(action)
        total += out.reward
        steps.append(out.reward)
        log.write({**base, "event": "turn", "turn": turn, "raw_output": raw, "action": action,
                   "observation": out.observation, "reward": out.reward, "done": out.done,
                   "tool_calls_so_far": getattr(env, "executed_tool_calls", None)})
        if out.done:
            final = out.info.get("final")
            if out.info.get("timeout"):
                failure = "never_submits"
            break
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": observation_message(out.observation)}]
    valid = bool(final) and final.get("error") is None
    if final and final.get("error"):
        failure = "rejected_submission"
    if failure is None and not valid:
        failure = "never_submits"
    result = EpisodeResult(task_id, attempt, total, final["final_reward"] if valid else cfg.failure_final_reward,
                           valid, getattr(env, "executed_tool_calls", 0), turns, failure, steps,
                           time.time() - started)
    log.write({**base, "event": "episode", **asdict(result), "final": final})
    return result


def run_rollouts(env, task_ids: Sequence[str], backend: ModelBackend, cfg: RolloutConfig,
                 log_path: str | Path | None = None, progress: bool = True) -> list[EpisodeResult]:
    log = JsonlLog(log_path)
    results = []
    for tid in task_ids:
        for attempt in range(cfg.attempts):
            r = rollout_task(env, tid, backend, cfg, attempt, log)
            results.append(r)
            if progress:
                print(f"[{len(results)}/{len(task_ids) * cfg.attempts}] {tid} a{attempt}: "
                      f"final {r.final_reward:+.3f} tools {r.tool_calls} {r.failure or 'ok'} ({r.seconds:.0f}s)",
                      flush=True)
    return results


def summarize(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    n = len(results)
    valid = [r for r in results if r.valid_submission]
    failures = Counter(r.failure for r in results if r.failure)
    return {"episodes": n,
            "mean_final_reward": round(statistics.fmean(r.final_reward for r in results), 3) if n else None,
            "mean_final_reward_valid_only": round(statistics.fmean(r.final_reward for r in valid), 3) if valid else None,
            "mean_return": round(statistics.fmean(r.total_return for r in results), 3) if n else None,
            "valid_submission_rate": round(len(valid) / n, 3) if n else None,
            "tool_calls_per_episode": round(statistics.fmean(r.tool_calls for r in results), 2) if n else None,
            "turns_per_episode": round(statistics.fmean(r.turns for r in results), 2) if n else None,
            "failure_types": dict(failures)}
