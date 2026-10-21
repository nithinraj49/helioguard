"""Phase 4 multi-turn environment: prompt, tools, caps, step and final rewards, no LLM judge."""
import functools
import json
import random
from pathlib import Path

import pytest

from helioguard.data.noaa import load_space_weather
from helioguard.env import HelioGuardEnv
from helioguard.storms import storm_catalog
from helioguard.tasks import ACTION_NAMES, oracle_plan, sample_task, score_plan
from verispace_engine.env_base import EnvBase, StepOutcome

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tasks():
    rows = load_space_weather(FIX / "sw_feb2022.csv") + load_space_weather(FIX / "sw_may2024.csv")
    storms = storm_catalog(rows, min_kp=5.0)
    return [sample_task(storms[i % len(storms)], random.Random(i), f"e{i}", "train") for i in range(12)]


@pytest.fixture
def env(tasks):
    return HelioGuardEnv(tasks, seed=0)


def call(name, **arguments):
    return {"type": "tool_call", "name": name, "arguments": arguments}


def submit(task, plan=None):
    return {"type": "submit_decision", **(plan or oracle_plan(task))}


# --- episode and prompt ------------------------------------------------------------------------
def test_env_is_an_engine_env_and_reset_returns_a_scenario_prompt(env, tasks):
    assert isinstance(env, EnvBase)
    prompt = env.reset(task_id=tasks[0]["id"])
    assert isinstance(prompt, str)
    for a in tasks[0]["fleet"]:
        assert a["id"] in prompt and f"{a['alt_km']:.0f}" in prompt
    low = prompt.lower()
    assert "forecast" in low and "kp" in low and "ground" in low
    if tasks[0]["launch"]:
        assert "launch" in low
    for d in tasks[0]["disruptions"]:
        assert d["type"].replace("_", " ") in low or d["type"] in low
    assert "recommend" in low                                    # HelioGuard recommends; humans decide


def test_prompt_never_reveals_the_optimal_answer_or_reward_tables(env, tasks):
    prompt = env.reset(task_id=tasks[0]["id"])
    assert "optimal" not in prompt.lower() and "reward" not in prompt.lower().replace("rewards are", "")
    assert str(round(tasks[0]["optimal"]["objective"], 3)) not in prompt


def test_reset_without_an_id_cycles_through_the_tasks(env, tasks):
    seen = {env.reset() for _ in range(len(tasks))}
    assert len(seen) > 1


def test_step_before_reset_is_an_error(tasks):
    with pytest.raises(RuntimeError):
        HelioGuardEnv(tasks).step(call("check_feasibility", asset_id="x", action="hold"))


# --- tools ---------------------------------------------------------------------------------------
def test_the_five_tools_are_available(env, tasks):
    env.reset(task_id=tasks[0]["id"])
    assert sorted(env.tool_names()) == ["check_feasibility", "min_safe_altitude", "qaoa_schedule",
                                        "qnn_density_correction", "simulate_decay"]
    assert all(s["description"] for s in env.tool_specs())


def test_physics_tools_return_results_through_step(env, tasks):
    t = tasks[0]
    env.reset(task_id=t["id"])
    a = t["fleet"][0]["id"]
    out = env.step(call("check_feasibility", asset_id=a, action="raise_orbit"))
    assert isinstance(out, StepOutcome) and not out.done
    res = out.observation["result"]
    assert {"can_gain_altitude", "can_gain_altitude_through_storm", "thrust_slows_decay",
            "drag_mN", "thrust_mN"} <= set(res)
    assert "feasible_now" not in res and "feasible_through_storm" not in res   # the old, misleading words
    out = env.step(call("simulate_decay", asset_id=a, action="safe_mode_edge_on", hours=24))
    assert len(out.observation["result"]["altitudes_km"]) >= 2
    out = env.step(call("min_safe_altitude", asset_id=a))
    assert "min_alt_km" in out.observation["result"]


def test_thruster_failure_is_visible_in_the_tool_results(tasks, env):
    t = next((t for t in tasks if any(d["type"] == "thruster_failure" for d in t["disruptions"])), None)
    if t is None:
        pytest.skip("no thruster-failure task in this sample")
    env.reset(task_id=t["id"])
    d = next(d for d in t["disruptions"] if d["type"] == "thruster_failure")
    out = env.step(call("check_feasibility", asset_id=d["asset_id"], action="raise_orbit"))
    assert out.observation["result"]["thrust_mN"] < 70.0


def test_qaoa_tool_returns_the_ortools_solution_next_to_it(env, tasks):
    t = tasks[0]
    env.reset(task_id=t["id"])
    out = env.step(call("qaoa_schedule"))
    r = out.observation["result"]
    assert "ortools" in r and "qaoa" in r and "no quantum advantage" in r["note"].lower()


def test_qnn_tool_works_while_disabled_and_returns_plain_nrlmsis(env, tasks):
    env.reset(task_id=tasks[0]["id"])
    out = env.step(call("qnn_density_correction", altitude_km=230.0, f107=130.0, ap=60.0,
                        hours_since_onset=6.0, local_solar_time_h=12.0, lat_band_deg=30.0))
    r = out.observation["result"]
    assert r["enabled"] is False and r["log_correction"] == 0.0
    assert r["corrected_density"] == r["base_density"] > 0
    assert "held-out" in r["reason"]


def test_bad_tool_calls_fail_cleanly_and_cost_reward(env, tasks):
    env.reset(task_id=tasks[0]["id"])
    for bad in (call("warp_drive"), call("check_feasibility", asset_id="ghost", action="hold"),
                call("check_feasibility", asset_id=tasks[0]["fleet"][0]["id"], action="teleport"),
                call("simulate_decay", asset_id=tasks[0]["fleet"][0]["id"], action="hold")):
        out = env.step(bad)
        assert out.reward < 0 and not out.done and out.observation["type"] == "error"


# --- caps ------------------------------------------------------------------------------------------
def test_tool_calls_are_capped_per_episode(tasks):
    env = HelioGuardEnv(tasks, max_tool_calls=3, max_turns=30)
    t = tasks[0]
    env.reset(task_id=t["id"])
    a = t["fleet"][0]["id"]
    outs = [env.step(call("simulate_decay", asset_id=a, action="hold", hours=h)) for h in (6, 12, 18, 24, 30)]
    assert [o.observation["type"] for o in outs] == ["tool_result"] * 3 + ["error"] * 2
    assert "limit" in outs[3].observation["message"].lower()
    assert all(o.reward < 0 for o in outs[3:])


def test_turn_cap_ends_the_episode_with_a_timeout_penalty(tasks):
    env = HelioGuardEnv(tasks, max_tool_calls=3, max_turns=5)
    env.reset(task_id=tasks[0]["id"])
    last = None
    for _ in range(5):
        last = env.step({"type": "noop"})
    assert last.done and last.reward <= -0.5 and last.info["timeout"] is True
    with pytest.raises(RuntimeError):
        env.step({"type": "noop"})


# --- submit_decision and the final reward ---------------------------------------------------------
def test_submit_ends_the_episode_and_the_oracle_gets_the_full_final_reward(env, tasks):
    t = tasks[0]
    env.reset(task_id=t["id"])
    out = env.step(submit(t))
    assert out.done and out.info["final"]["final_reward"] == pytest.approx(1.0, abs=1e-6)
    assert out.reward == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(RuntimeError):
        env.step(submit(t))


def test_final_reward_comes_from_the_exact_solver_gap(env, tasks):
    t = tasks[0]
    ids = [a["id"] for a in t["fleet"]]
    env.reset(task_id=t["id"])
    worse = {"assets": {i: "hold" for i in ids}}
    if t["launch"]:
        worse["launch"] = {"action": "hold"}
    out = env.step({"type": "submit_decision", **worse})
    f = out.info["final"]
    assert f["optimal_objective"] == pytest.approx(t["optimal"]["objective"])
    assert f["final_reward"] == pytest.approx(score_plan(t, worse)["final_reward"])
    assert f["final_reward"] < 1.0


def test_malformed_submission_ends_the_episode_with_a_penalty(env, tasks):
    env.reset(task_id=tasks[0]["id"])
    out = env.step({"type": "submit_decision", "assets": {"ghost": "hold"}})
    assert out.done and out.reward <= -0.5 and out.info["final"]["error"]


def test_raising_against_thrust_slows_decay_false_is_penalised(tasks):
    for t in tasks:
        for a in t["fleet"]:
            e = HelioGuardEnv(tasks, seed=0)
            e.reset(task_id=t["id"])
            res = e.step(call("check_feasibility", asset_id=a["id"], action="raise_orbit")).observation["result"]
            if res["thrust_slows_decay"] is False and a["window"] not in t["windows"]["outage"]:
                plan = oracle_plan(t)
                plan["assets"][a["id"]] = "raise_orbit"               # raise although it does not slow decay
                out = e.step({"type": "submit_decision", **plan})
                assert any("ignored" in p for p in out.info["final"]["penalties"])
                return
    pytest.skip("no raise that fails to slow decay in this sample")


@functools.lru_cache(maxsize=1)
def extra_tasks():
    """A larger fixed sample (60 tasks), for tests that need a rarer situation."""
    rows = load_space_weather(FIX / "sw_feb2022.csv") + load_space_weather(FIX / "sw_may2024.csv")
    storms = storm_catalog(rows, min_kp=5.0)
    return tuple(sample_task(storms[i % len(storms)], random.Random(500 + i), f"x{i}", "train")
                 for i in range(60))


def test_drag_above_thrust_alone_is_not_penalised_when_raising_still_slows_decay(tasks):
    """The tool says it cannot gain altitude but thrust slows decay: raising must not be punished."""
    tasks = list(tasks) + list(extra_tasks())
    for t in tasks:
        for a in t["fleet"]:
            e = HelioGuardEnv(tasks, seed=0)
            e.reset(task_id=t["id"])
            res = e.step(call("check_feasibility", asset_id=a["id"], action="raise_orbit")).observation["result"]
            if (res["can_gain_altitude"] is False and res["thrust_slows_decay"] is True
                    and a["window"] not in t["windows"]["outage"]):
                plan = {"assets": {x["id"]: "safe_mode_edge_on" for x in t["fleet"]}}
                plan["assets"][a["id"]] = "raise_orbit"
                if t["launch"]:
                    plan["launch"] = {"action": "hold"}
                out = e.step({"type": "submit_decision", **plan})
                assert not any("ignored" in p for p in out.info["final"]["penalties"])
                assert not any("infeasible" in v for v in out.info["final"]["violations"])
                return
    pytest.skip("no such asset in this sample")


def find_task_where_optimum_raises_with_drag_above_thrust(tasks):
    for t in tasks:
        for a in t["fleet"]:
            if (t["optimal"]["plan"]["assets"][a["id"]] == "raise_orbit"
                    and not t["nominal"]["can_gain_altitude"][a["id"]]):
                return t
    return None


def test_optimal_trace_with_drag_above_thrust_gets_no_step_penalty_and_rewards_agree(tasks):
    t = find_task_where_optimum_raises_with_drag_above_thrust(tasks)
    if t is None:                                    # look in a larger, fixed sample
        tasks = list(extra_tasks())
        t = find_task_where_optimum_raises_with_drag_above_thrust(tasks)
    assert t is not None, "the stored optimum never raises with drag above thrust in this sample"
    env = HelioGuardEnv(tasks)
    env.reset(task_id=t["id"])
    step_rewards = []
    for a in t["fleet"]:
        out = env.step(call("check_feasibility", asset_id=a["id"], action="raise_orbit"))
        step_rewards.append(out.reward)
        if t["optimal"]["plan"]["assets"][a["id"]] == "raise_orbit" and not t["nominal"]["can_gain_altitude"][a["id"]]:
            assert out.observation["result"]["can_gain_altitude"] is False        # cannot gain altitude ...
            assert out.observation["result"]["thrust_slows_decay"] is True        # ... but raising still pays
    final = env.step(submit(t))
    assert all(r >= 0 for r in step_rewards)                                       # no step penalty
    assert final.info["final"]["penalties"] == [] and final.info["final"]["violations"] == []
    assert sum(step_rewards) > 0 and final.reward > 0                              # same sign
    assert final.reward == pytest.approx(1.0, abs=1e-6)


# --- step rewards (verifier, no LLM judge) ---------------------------------------------------------
def test_a_useful_new_tool_call_beats_its_cost_and_a_duplicate_does_not(env, tasks):
    t = tasks[0]
    env.reset(task_id=t["id"])
    a = t["fleet"][0]["id"]
    first = env.step(call("check_feasibility", asset_id=a, action="raise_orbit"))
    dup = env.step(call("check_feasibility", asset_id=a, action="raise_orbit"))
    assert first.reward > 0 > dup.reward


def test_expensive_tool_costs_more_and_is_only_rewarded_when_the_problem_is_hard(tasks):
    easy = next(t for t in tasks if not t["needs_qaoa"])
    hard = next((t for t in tasks if t["needs_qaoa"]), None)
    e = HelioGuardEnv(tasks)
    e.reset(task_id=easy["id"])
    assert e.step(call("qaoa_schedule")).reward < 0                      # not needed: pure cost
    if hard is not None:
        e.reset(task_id=hard["id"])
        r_hard = e.step(call("qaoa_schedule")).reward
        assert r_hard > -0.1 and r_hard > -0.11


def test_correct_claims_are_rewarded_wrong_or_repeated_ones_are_not(env, tasks):
    t = tasks[0]
    env.reset(task_id=t["id"])
    a = t["fleet"][0]["id"]
    truth = env.step(call("check_feasibility", asset_id=a, action="raise_orbit")).observation["result"]
    claim = {"type": "claim_can_gain_altitude", "asset_id": a, "when": "now",
             "can_gain_altitude": truth["can_gain_altitude"]}
    good = env.step(claim)
    again = env.step(claim)
    wrong = env.step({"type": "claim_thrust_slows_decay", "asset_id": a,
                      "thrust_slows_decay": not truth["thrust_slows_decay"]})
    other = t["fleet"][-1]["id"]                    # a different question gets its own reward
    other_truth = env.step(call("check_feasibility", asset_id=other, action="raise_orbit")).observation["result"]
    right = env.step({"type": "claim_thrust_slows_decay", "asset_id": other,
                      "thrust_slows_decay": other_truth["thrust_slows_decay"]})
    flipped = env.step({"type": "claim_thrust_slows_decay", "asset_id": other,
                        "thrust_slows_decay": not other_truth["thrust_slows_decay"]})
    assert good.reward > 0 and again.reward <= 0 and wrong.reward < 0 and right.reward > 0
    assert flipped.reward <= 0                       # re-asking the same question earns nothing


def test_vague_decay_predictions_are_not_rewarded(env, tasks):
    t = tasks[0]
    env.reset(task_id=t["id"])
    a = t["fleet"][0]["id"]
    sim = env.step(call("simulate_decay", asset_id=a, action="safe_mode_edge_on", hours=24)).observation["result"]
    true_loss = t["fleet"][0]["alt_km"] - sim["final_alt_km"]
    for vague in (0.0, 1000.0, true_loss * 2, true_loss * 0.5):
        e = HelioGuardEnv(tasks)
        e.reset(task_id=t["id"])
        out = e.step({"type": "claim_decay", "asset_id": a, "action": "safe_mode_edge_on", "hours": 24,
                      "altitude_loss_km": vague})
        assert out.reward < 0
    e = HelioGuardEnv(tasks)
    e.reset(task_id=t["id"])
    ok = e.step({"type": "claim_decay", "asset_id": a, "action": "safe_mode_edge_on", "hours": 24,
                 "altitude_loss_km": true_loss})
    assert ok.reward > 0


def test_unknown_or_malformed_actions_cost_reward_and_a_turn(env, tasks):
    env.reset(task_id=tasks[0]["id"])
    for bad in ({"type": "think_harder"}, {"nope": 1}, {"type": "tool_call"}):
        out = env.step(bad)
        assert out.reward < 0 and not out.done


def test_step_rewards_are_small_next_to_the_final_reward(env, tasks):
    t = tasks[0]
    env.reset(task_id=t["id"])
    total = 0.0
    for a in t["fleet"]:
        total += env.step(call("check_feasibility", asset_id=a["id"], action="raise_orbit")).reward
    assert 0 < total < 0.5
    assert env.step(submit(t)).reward == pytest.approx(1.0, abs=1e-6)
