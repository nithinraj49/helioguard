"""Phase 4 loophole tests: each cheap trick must lose reward against the proper policies."""
import random
import statistics
from pathlib import Path

import pytest

from helioguard.data.noaa import load_space_weather
from helioguard.env import HelioGuardEnv
from helioguard.policies import (AlwaysRaisePolicy, EndlessToolCallsPolicy, NeverCallToolsPolicy,
                                 NeverSubmitPolicy, OraclePolicy, RandomPolicy, RuleBasedPolicy,
                                 VaguePredictionsPolicy, run_episode)
from helioguard.storms import storm_catalog
from helioguard.tasks import sample_task

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tasks():
    rows = load_space_weather(FIX / "sw_feb2022.csv") + load_space_weather(FIX / "sw_may2024.csv")
    storms = storm_catalog(rows, min_kp=5.0)
    return [sample_task(storms[i % len(storms)], random.Random(100 + i), f"l{i}", "train") for i in range(16)]


def mean_return(tasks, policy_factory):
    env = HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20)
    return statistics.fmean(run_episode(env, policy_factory(), t["id"])["return"] for t in tasks)


def test_oracle_is_the_ceiling(tasks):
    assert mean_return(tasks, OraclePolicy) > 1.0                       # final 1.0 plus step bonuses
    assert mean_return(tasks, OraclePolicy) > mean_return(tasks, RuleBasedPolicy)


def test_always_raise_loses_reward(tasks):
    assert mean_return(tasks, AlwaysRaisePolicy) < mean_return(tasks, RuleBasedPolicy) - 0.1
    assert mean_return(tasks, AlwaysRaisePolicy) < mean_return(tasks, OraclePolicy) - 0.3


def test_never_calling_tools_loses_reward_against_the_same_decision_with_tools(tasks):
    blind = mean_return(tasks, NeverCallToolsPolicy)
    informed = mean_return(tasks, RuleBasedPolicy)
    assert blind < informed


def test_never_calling_tools_earns_no_step_reward(tasks):
    env = HelioGuardEnv(tasks)
    r = run_episode(env, NeverCallToolsPolicy(), tasks[0]["id"])
    assert r["tool_calls"] == 0 and r["step_reward_total"] == pytest.approx(0.0)


def test_endless_tool_calls_end_with_the_timeout_penalty(tasks):
    env = HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20)
    r = run_episode(env, EndlessToolCallsPolicy(), tasks[0]["id"])
    assert r["timeout"] is True and not r["submitted"]
    assert r["return"] < 0 and r["tool_calls"] == 10                    # capped, then rejected
    assert r["return"] < mean_return(tasks, RuleBasedPolicy)


def test_never_submitting_is_punished(tasks):
    env = HelioGuardEnv(tasks)
    r = run_episode(env, NeverSubmitPolicy(), tasks[0]["id"])
    assert r["timeout"] and not r["submitted"] and r["return"] <= -0.5


def test_vague_predictions_earn_nothing_and_do_not_help_the_decision(tasks):
    vague = mean_return(tasks, VaguePredictionsPolicy)
    assert vague < mean_return(tasks, RuleBasedPolicy)
    env = HelioGuardEnv(tasks)
    r = run_episode(env, VaguePredictionsPolicy(), tasks[0]["id"])
    assert r["claim_reward_total"] <= 0.0


def test_random_policy_is_worse_than_the_rule_based_policy(tasks):
    rng = random.Random(0)
    rand = statistics.fmean(run_episode(HelioGuardEnv(tasks), RandomPolicy(rng), t["id"])["return"] for t in tasks)
    assert rand < mean_return(tasks, RuleBasedPolicy)


def test_farming_step_rewards_cannot_beat_a_good_decision(tasks):
    env = HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20)
    t = tasks[0]
    env.reset(task_id=t["id"])
    farmed = 0.0
    for a in t["fleet"]:
        for act in ("raise_orbit", "safe_mode_edge_on", "hold"):
            for h in (6, 12):
                farmed += env.step({"type": "tool_call", "name": "simulate_decay",
                                    "arguments": {"asset_id": a["id"], "action": act, "hours": h}}).reward
    assert farmed < 0.6          # bounded; the final reward (up to 1.0) dominates
