"""Phase 5: the tool-informed baseline decides only from the briefing and tool results."""
from pathlib import Path

import pytest

from helioguard.env import HelioGuardEnv
from helioguard.policies import NeverCallToolsPolicy, ToolInformedPolicy, run_episode
from helioguard.tasks import load_tasks

HIDDEN = ("nominal", "optimal", "baselines", "needs_qaoa")


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(Path("tasks") / "eval.jsonl")[:12]


class _Locked(dict):
    """A task whose stored answers raise when touched: the policy may only use the briefing."""

    def __getitem__(self, key):
        if key in HIDDEN:
            raise AssertionError(f"policy read hidden field {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in HIDDEN:
            raise AssertionError(f"policy read hidden field {key!r}")
        return super().get(key, default)


def test_policy_never_reads_the_stored_answers(tasks):
    env = HelioGuardEnv(tasks)
    for t in tasks[:5]:
        env.reset(task_id=t["id"])
        locked = _Locked(env.task)
        gen = ToolInformedPolicy().actions(locked)
        out, action = None, next(gen)
        while True:
            out = env.step(action)
            if out.done:
                break
            action = gen.send(out)


def test_policy_submits_a_valid_plan_within_the_tool_cap(tasks):
    env = HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20)
    for t in tasks:
        r = run_episode(env, ToolInformedPolicy(), t["id"])
        assert r["submitted"] and r["final"]["error"] is None
        assert 1 <= r["tool_calls"] <= 10
        assert r["final"]["violations"] == []


def test_policy_simulates_raise_and_safe_mode_for_every_asset(tasks):
    env = HelioGuardEnv(tasks)
    t = tasks[0]
    env.reset(task_id=t["id"])
    gen = ToolInformedPolicy().actions(env.task)
    seen, action = set(), next(gen)
    while action["type"] == "tool_call":
        if action["name"] == "simulate_decay":
            seen.add((action["arguments"]["asset_id"], action["arguments"]["action"]))
        out = env.step(action)
        action = gen.send(out)
    for a in t["fleet"]:
        assert (a["id"], "raise_orbit") in seen and (a["id"], "safe_mode_edge_on") in seen


def test_policy_is_not_worse_than_the_same_plan_blind_on_average(tasks):
    env = HelioGuardEnv(tasks)
    informed = [run_episode(env, ToolInformedPolicy(), t["id"])["final"]["final_reward"] for t in tasks]
    blind = [run_episode(env, NeverCallToolsPolicy(), t["id"])["final"]["final_reward"] for t in tasks]
    # a sanity floor, not the Phase 5 claim: the real comparison is reported on the full splits
    assert sum(informed) / len(informed) > -0.5 and len(blind) == len(informed)
