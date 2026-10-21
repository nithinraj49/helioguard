"""Scripted policies for baselines and loophole tests, and an episode runner.

These are not learned agents. The oracle reads the stored optimum (an upper bound); the rule-based
policy is the baseline used to filter tasks; the rest are cheap tricks that must lose reward.
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Iterator

from .env import HelioGuardEnv
from .tasks import ACTION_NAMES, RAISE, oracle_plan, random_plan, rule_based_plan


def _call(name: str, **arguments: Any) -> dict:
    return {"type": "tool_call", "name": name, "arguments": arguments}


def _submit(plan: dict) -> dict:
    return {"type": "submit_decision", **plan}


class Policy:
    """Subclasses yield actions for a task; the runner sends them one by one."""

    def actions(self, task: dict) -> Iterator[dict]:
        raise NotImplementedError


class _Informed(Policy):
    """Checks raise feasibility for every asset (cheap, useful), then submits a plan."""

    def plan(self, task: dict) -> dict:
        raise NotImplementedError

    def actions(self, task: dict) -> Iterator[dict]:
        for a in task["fleet"]:
            yield _call("check_feasibility", asset_id=a["id"], action=RAISE)
        yield _submit(self.plan(task))


class OraclePolicy(_Informed):
    def plan(self, task: dict) -> dict:
        return oracle_plan(task)


class RuleBasedPolicy(_Informed):
    def plan(self, task: dict) -> dict:
        return rule_based_plan(task)


class NeverCallToolsPolicy(Policy):
    """Same decision as the rule-based policy, without looking at any tool."""

    def actions(self, task: dict) -> Iterator[dict]:
        yield _submit(rule_based_plan(task))


class RandomPolicy(Policy):
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def actions(self, task: dict) -> Iterator[dict]:
        yield _submit(random_plan(task, self.rng))


class AlwaysRaisePolicy(Policy):
    def actions(self, task: dict) -> Iterator[dict]:
        plan: dict[str, Any] = {"assets": {a["id"]: RAISE for a in task["fleet"]}}
        if task["launch"]:
            plan["launch"] = {"action": "deploy", "alt_km": task["launch"]["options_km"][0]}
        yield _submit(plan)


class EndlessToolCallsPolicy(Policy):
    """Keeps calling tools with fresh arguments and never decides."""

    def actions(self, task: dict) -> Iterator[dict]:
        ids = [a["id"] for a in task["fleet"]]
        for h in itertools.count(1):
            yield _call("simulate_decay", asset_id=ids[h % len(ids)], action=ACTION_NAMES[h % 3], hours=float(h % 90 + 1))


class NeverSubmitPolicy(Policy):
    def actions(self, task: dict) -> Iterator[dict]:
        while True:
            yield {"type": "noop"}


class VaguePredictionsPolicy(Policy):
    """Makes vague decay claims (no loss at all, or an absurd loss), then submits the rule plan."""

    def actions(self, task: dict) -> Iterator[dict]:
        for a in task["fleet"]:
            for vague in (0.0, 1000.0):
                yield {"type": "claim_decay", "asset_id": a["id"], "action": "safe_mode_edge_on",
                       "hours": 24, "altitude_loss_km": vague}
        yield _submit(rule_based_plan(task))


def run_episode(env: HelioGuardEnv, policy: Policy, task_id: str) -> dict[str, Any]:
    """Play one episode. Returns the total reward and a breakdown."""
    env.reset(task_id=task_id)
    total = step_total = claim_total = 0.0
    out = None
    submitted = timeout = False
    final = None
    for action in policy.actions(env.task):
        out = env.step(action)
        total += out.reward
        if action.get("type") == "submit_decision":
            submitted = True
            final = out.info.get("final")
        else:
            step_total += out.reward
            if str(action.get("type", "")).startswith("claim"):
                claim_total += out.reward
        if out.done:
            timeout = bool(out.info.get("timeout"))
            break
    return {"return": total, "step_reward_total": step_total, "claim_reward_total": claim_total,
            "tool_calls": env.executed_tool_calls, "submitted": submitted, "timeout": timeout,
            "final": final, "done": bool(out and out.done)}
