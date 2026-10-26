"""The compact tool-observation view must not lose decision-relevant information (Amendment 2).

For every simulate_decay result the view carries the true minimum of the FULL hourly raw series
(min_alt_km, min_alt_hour), not the minimum of the 12-hourly samples. Raise trajectories dip and then
recover, so a downsampled series can hide the lowest point (18 of 459 eval raise trajectories did, by up
to 1.52 km, before this fix).
"""
from pathlib import Path

import pytest

from helioguard.env import HelioGuardEnv
from helioguard.harness_env import compact_observation
from helioguard.tasks import load_tasks

ROOT = Path(__file__).parent.parent
ACTIONS = ("raise_orbit", "safe_mode_edge_on", "hold")


def _r3(v):
    return round(v, 3) if isinstance(v, float) else v


def check_task(env, task, hours):
    """Assert the compact view of every simulate_decay for this task keeps the facts. Returns the count."""
    env.reset(task_id=task["id"])
    n = 0
    for a in task["fleet"]:
        for act in ACTIONS:
            out = env.step({"type": "tool_call", "name": "simulate_decay",
                            "arguments": {"asset_id": a["id"], "action": act, "hours": hours}})
            raw = out.observation
            if raw.get("type") != "tool_result":
                continue
            r, c = raw["result"], compact_observation(raw)["result"]
            alts, times = r["altitudes_km"], r["times_h"]
            i = min(range(len(alts)), key=lambda k: alts[k])          # first occurrence of the true minimum
            assert c["min_alt_km"] == _r3(alts[i]), (task["id"], act)
            assert c["min_alt_hour"] == _r3(times[i]), (task["id"], act)
            assert c["min_alt_km"] <= min(c["altitude_km_at_hours"].values()) + 1e-9
            assert c["min_alt_km"] <= c["final_alt_km"] + 1e-9
            for k in ("final_alt_km", "reentry_hour", "reentered", "altitude_loss_km"):   # already-confirmed fields
                assert c[k] == _r3(r[k]), (task["id"], act, k)
            assert "altitudes_km" not in c and "times_h" not in c
            n += 1
    return n


def test_every_simulate_decay_on_all_eval_tasks_keeps_the_true_minimum():
    tasks = load_tasks(ROOT / "tasks" / "eval.jsonl")
    env = HelioGuardEnv(tasks)
    assert sum(check_task(env, t, 48.0) for t in tasks) > 400
    for hours in (24.0, 96.0):                                          # other horizons the agent may ask for
        assert sum(check_task(env, t, hours) for t in tasks[:12]) > 50


@pytest.mark.slow
def test_every_simulate_decay_on_all_train_tasks_keeps_the_true_minimum():
    tasks = load_tasks(ROOT / "tasks" / "train.jsonl")
    env = HelioGuardEnv(tasks)
    assert sum(check_task(env, t, 48.0) for t in tasks) > 8000


def test_a_dip_below_the_12_hourly_samples_is_now_visible():
    """eval_storm_20220203_0 / b2 / raise: true minimum 209.529 km at hour 10, samples show 210.0 at hour 0."""
    tasks = load_tasks(ROOT / "tasks" / "eval.jsonl")
    env = HelioGuardEnv(tasks)
    env.reset(task_id="eval_storm_20220203_0")
    out = env.step({"type": "tool_call", "name": "simulate_decay",
                    "arguments": {"asset_id": "b2", "action": "raise_orbit", "hours": 48.0}})
    c = compact_observation(out.observation)["result"]
    assert c["min_alt_km"] == pytest.approx(209.529, abs=1e-3) and c["min_alt_hour"] == 10.0
    assert min(c["altitude_km_at_hours"].values()) == 210.0            # the samples alone would have hidden it


def test_other_tool_facts_are_unchanged_by_the_view():
    tasks = load_tasks(ROOT / "tasks" / "eval.jsonl")
    env = HelioGuardEnv(tasks)
    for t in tasks:
        env.reset(task_id=t["id"])
        for a in t["fleet"]:
            raw = env.step({"type": "tool_call", "name": "check_feasibility",
                            "arguments": {"asset_id": a["id"], "action": "raise_orbit"}}).observation
            c = compact_observation(raw)["result"]
            for k in ("can_gain_altitude", "thrust_slows_decay", "can_gain_altitude_through_storm",
                      "ground_window_available"):
                assert c[k] == raw["result"][k]
        raw = env.step({"type": "tool_call", "name": "min_safe_altitude", "arguments": {}}).observation
        assert compact_observation(raw)["result"]["min_alt_km"] == _r3(raw["result"]["min_alt_km"])


def test_the_environment_itself_still_returns_the_raw_series():
    """The view is a presentation layer: the environment observation is untouched."""
    tasks = load_tasks(ROOT / "tasks" / "eval.jsonl")[:1]
    env = HelioGuardEnv(tasks)
    env.reset(task_id=tasks[0]["id"])
    raw = env.step({"type": "tool_call", "name": "simulate_decay",
                    "arguments": {"asset_id": tasks[0]["fleet"][0]["id"], "action": "hold", "hours": 48.0}}).observation
    assert "altitudes_km" in raw["result"] and "min_alt_km" not in raw["result"]
