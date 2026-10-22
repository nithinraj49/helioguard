"""Scripted policies for baselines and loophole tests, and an episode runner.

These are not learned agents. The oracle reads the stored optimum (an upper bound); the rule-based
policy is the baseline used to filter tasks; the rest are cheap tricks that must lose reward.
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Iterator

from .env import HelioGuardEnv
from .tasks import (ACTION_NAMES, ALT_SCALE_KM, EDGE_ON, HOLD, HORIZON_H, RAISE, oracle_plan, random_plan,
                    rule_based_plan)


def _call(name: str, **arguments: Any) -> dict:
    return {"type": "tool_call", "name": name, "arguments": arguments}


def _submit(plan: dict) -> dict:
    return {"type": "submit_decision", **plan}


class Policy:
    """Subclasses yield actions for a task; the runner sends each one to the environment and sends
    the outcome back into the generator, so a policy can read tool results (most ignore them)."""

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


ALLOWED_BRIEFING_KEYS = ("fleet", "windows", "launch", "forecast", "storm", "disruptions")


def _action_value(sim: dict) -> float:
    """Value of one simulated action, on the same scale as the task rewards: survivors score by
    final altitude, a satellite that re-enters scores by how long it lasted."""
    if sim.get("reentered"):
        return 0.5 * sim["reentry_hour"] / HORIZON_H
    return 1.0 + (sim["final_alt_km"] - 150.0) / ALT_SCALE_KM


class ToolInformedPolicy(Policy):
    """Decides only from the briefing and tool results (never the stored answers).

    Simulates raise and safe mode for every asset over the planning horizon (then hold, then
    check_feasibility, as long as the 10-call cap allows), asks min_safe_altitude for the launch,
    and submits the best plan that respects the ground windows. It sees the forecast storm only:
    the stronger or longer storms are not visible to any tool, which is the gap to the optimum."""

    def __init__(self, extra_calls: bool = True) -> None:
        self.extra_calls = extra_calls

    def actions(self, task: dict) -> Iterator[dict]:
        brief = {k: task[k] for k in ALLOWED_BRIEFING_KEYS}
        fleet, outage = brief["fleet"], set(brief["windows"]["outage"])
        value: dict[tuple[str, str], float] = {}
        budget = 10
        order = [(a["id"], act) for act in (RAISE, EDGE_ON) for a in fleet]
        if self.extra_calls:
            order += [(a["id"], HOLD) for a in fleet]
        reserve = 1 if brief["launch"] else 0                # keep one call for min_safe_altitude
        for asset_id, action in order:
            if budget <= (reserve if action == HOLD else 0):
                continue
            out = yield _call("simulate_decay", asset_id=asset_id, action=action, hours=float(HORIZON_H))
            budget -= 1
            if out.observation.get("type") == "tool_result":
                value[(asset_id, action)] = _action_value(out.observation["result"])
        min_alt = None
        if brief["launch"] and budget > 0:
            out = yield _call("min_safe_altitude")
            budget -= 1
            if out.observation.get("type") == "tool_result":
                min_alt = out.observation["result"]["min_alt_km"]
        if self.extra_calls:
            for a in fleet:
                if budget <= 0:
                    break
                yield _call("check_feasibility", asset_id=a["id"], action=RAISE)
                budget -= 1
        yield _submit(self._plan(brief, value, outage, min_alt))

    @staticmethod
    def _plan(brief: dict, value: dict, outage: set, min_alt: float | None) -> dict:
        fleet = brief["fleet"]
        best: dict[str, str] = {}
        for a in fleet:
            options = [act for act in ACTION_NAMES if (a["id"], act) in value] or [EDGE_ON]
            if a["window"] in outage:
                options = [o for o in options if o != RAISE] or [EDGE_ON]
            best[a["id"]] = max(options, key=lambda act: value.get((a["id"], act), -1.0))
        by_window: dict[Any, list[dict]] = {}
        for a in fleet:
            if best[a["id"]] == RAISE:
                by_window.setdefault(a["window"], []).append(a)
        for window, claimants in by_window.items():       # one raise per ground window: keep the biggest gain
            def gain(a):
                return value[(a["id"], RAISE)] - value.get((a["id"], EDGE_ON), 0.0)
            for loser in sorted(claimants, key=gain, reverse=True)[1:]:
                best[loser["id"]] = EDGE_ON
        plan: dict[str, Any] = {"assets": best}
        if brief["launch"]:
            options = brief["launch"]["options_km"]
            pick = next((o for o in options if min_alt is not None and o >= min_alt), None)
            plan["launch"] = {"action": "deploy", "alt_km": pick} if pick else {"action": "hold"}
        return plan


def run_episode(env: HelioGuardEnv, policy: Policy, task_id: str) -> dict[str, Any]:
    """Play one episode. Returns the total reward and a breakdown."""
    env.reset(task_id=task_id)
    total = step_total = claim_total = 0.0
    out = None
    submitted = timeout = False
    final = None
    gen = policy.actions(env.task)
    action = next(gen, None)
    while action is not None:
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
        try:
            action = gen.send(out)
        except StopIteration:
            action = None
    return {"return": total, "step_reward_total": step_total, "claim_reward_total": claim_total,
            "tool_calls": env.executed_tool_calls, "submitted": submitted, "timeout": timeout,
            "final": final, "done": bool(out and out.done)}
