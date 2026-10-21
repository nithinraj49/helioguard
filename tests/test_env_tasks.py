"""Phase 4 task generation: real-storm scenarios, disruptions, exact optimum, filtering, splits.

Spacecraft values and the Kp -> density-multiplier mapping are ILLUSTRATIVE. Storm dates and Kp
series are real (from the SW-All indices in the offline fixtures).
"""
import itertools
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

from helioguard.data.noaa import load_space_weather
from helioguard.storms import StormProfile, peak_multiplier, storm_catalog
from helioguard.tasks import (ACTION_NAMES, DISRUPTIONS, build_dataset, density_fn_for, load_tasks,
                              oracle_plan, random_plan, rule_based_plan, sample_task, save_tasks,
                              score_plan, solve_optimal)

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def rows():
    return load_space_weather(FIX / "sw_feb2022.csv") + load_space_weather(FIX / "sw_may2024.csv")


@pytest.fixture(scope="module")
def storms(rows):
    return storm_catalog(rows, min_kp=5.0)


# --- storms ---------------------------------------------------------------------------------
def test_catalog_finds_the_real_feb_2022_and_may_2024_storms(storms):
    by_day = {s.onset.date().isoformat(): s for s in storms}
    feb = next(s for s in storms if s.onset.year == 2022 and s.onset.month == 2)
    may = max((s for s in storms if s.onset.year == 2024 and s.onset.month == 5), key=lambda s: s.kp_max)
    assert feb.kp_max == pytest.approx(5.3) and feb.g_scale == "G1"
    assert may.kp_max == pytest.approx(9.0) and may.g_scale == "G5"
    assert len(by_day) == len(storms)                        # one entry per storm
    assert all(s.onset.tzinfo is not None and len(s.kp_series) == 48 for s in storms)


def test_every_storm_has_a_stable_unique_id(storms):
    ids = [s.id for s in storms]
    assert len(set(ids)) == len(ids) and all(i.startswith("storm_") for i in ids)


def test_peak_multiplier_is_monotone_and_marked_illustrative():
    assert peak_multiplier(5.0) < peak_multiplier(7.0) < peak_multiplier(9.0)
    assert peak_multiplier(5.3) == pytest.approx(1.455, abs=0.01)     # about the Phase 1 x1.5


def test_profile_is_quiet_before_the_storm_and_peaks_during_it(storms):
    s = next(s for s in storms if s.kp_max >= 9)
    prof = StormProfile(s, arrival_h=12.0, peak=2.5)
    assert prof(0.0) == pytest.approx(1.0)
    peak = max(prof(t) for t in range(0, 96))
    assert peak == pytest.approx(2.5, abs=1e-6)
    stretched = StormProfile(s, arrival_h=12.0, peak=2.5, stretch=2.0)
    hours_above = lambda p: sum(p(t / 2) > 1.2 for t in range(0, 200))
    assert hours_above(stretched) > hours_above(prof)                  # a longer storm lasts longer


# --- task sampling ------------------------------------------------------------------------------
def make(storms, seed=0, i=0):
    return sample_task(storms[i % len(storms)], random.Random(seed), f"t{seed}_{i}", "train")


@pytest.fixture(scope="module")
def pool(storms):
    """40 sampled tasks, generated once (each takes about half a second)."""
    return [make(storms, seed) for seed in range(40)]


def test_task_is_deterministic_for_a_seed(storms):
    assert make(storms, 3) == make(storms, 3)
    assert make(storms, 3) != make(storms, 4)


def test_task_has_everything_the_prompt_and_scoring_need(storms):
    t = make(storms, 1)
    for key in ("id", "split", "storm", "forecast", "fleet", "windows", "launch", "disruptions",
                "rewards", "optimal", "needs_qaoa", "baselines"):
        assert key in t
    assert 2 <= len(t["fleet"]) <= 4
    assert {"arrival_h", "peak_multiplier", "horizon_h", "stronger_factor", "stretch"} <= set(t["forecast"])
    for a in t["fleet"]:
        assert set(t["rewards"]["assets"][a["id"]]) == set(ACTION_NAMES)
        assert all(len(v) == 3 for v in t["rewards"]["assets"][a["id"]].values())   # 3 scenarios
    assert t["optimal"]["status"] == "OPTIMAL"


def test_disruption_types_are_the_documented_ones_and_change_the_task(pool):
    assert set(DISRUPTIONS) == {"forecast_error", "storm_stronger", "storm_longer", "thruster_failure",
                                "ground_outage", "launch_slip"}
    seen = set()
    for t in pool:
        for d in t["disruptions"]:
            seen.add(d["type"])
            if d["type"] == "thruster_failure":
                asset = next(a for a in t["fleet"] if a["id"] == d["asset_id"])
                assert 0.2 <= asset["thrust_fraction"] <= 0.7
            if d["type"] == "ground_outage":
                assert d["window"] in t["windows"]["outage"]
                assert len(t["windows"]["all"]) >= 2
            if d["type"] == "launch_slip":
                assert t["launch"] is not None and t["launch"]["slip_h"] > 0
            if d["type"] in ("forecast_error", "storm_stronger"):
                assert t["forecast"]["stronger_factor"] >= 1.4
            if d["type"] == "storm_longer":
                assert t["forecast"]["stretch"] >= 1.6
    assert seen == set(DISRUPTIONS)


def test_launch_options_cover_hold_and_a_range_of_deployment_altitudes(pool):
    for t in pool:
        if t["launch"]:
            opts = t["launch"]["options_km"]
            assert opts == sorted(opts) and opts[0] <= 210 and opts[-1] >= 280
            assert set(t["rewards"]["launch"]) == {"hold"} | {str(o) for o in opts}
            return
    pytest.fail("no task with a launch in the pool")


# --- the exact optimum --------------------------------------------------------------------------
def brute_force(task):
    ids = [a["id"] for a in task["fleet"]]
    launch_opts = [None] + (task["launch"]["options_km"] if task["launch"] else [])
    best = None
    for combo in itertools.product(ACTION_NAMES, repeat=len(ids)):
        for lo in (launch_opts if task["launch"] else [None]):
            plan = {"assets": dict(zip(ids, combo))}
            if task["launch"]:
                plan["launch"] = {"action": "hold"} if lo is None else {"action": "deploy", "alt_km": lo}
            s = score_plan(task, plan)
            if s["violations"]:
                continue
            if best is None or s["objective"] > best:
                best = s["objective"]
    return best


def test_ortools_optimum_equals_brute_force_over_every_plan(pool):
    checked = 0
    for t in pool:
        if len(t["fleet"]) > 3:
            continue
        assert t["optimal"]["objective"] == pytest.approx(brute_force(t), abs=2e-3)
        checked += 1
        if checked == 5:
            return
    pytest.fail("not enough small tasks")


def test_stored_optimum_is_reproducible_and_never_beaten(pool):
    rng = random.Random(0)
    for t in pool[:15]:
        again = solve_optimal(t)
        assert again["objective"] == pytest.approx(t["optimal"]["objective"], abs=1e-9)
        for _ in range(20):
            s = score_plan(t, random_plan(t, rng))
            assert s["objective"] <= t["optimal"]["objective"] + 1e-6


def test_oracle_plan_scores_one_and_has_no_violations(pool):
    for t in pool[:15]:
        s = score_plan(t, oracle_plan(t))
        assert s["final_reward"] == pytest.approx(1.0, abs=1e-6) and s["violations"] == []


# --- scoring --------------------------------------------------------------------------------------
def test_reward_falls_with_the_gap_and_rule_breaking_is_penalised(pool):
    t = pool[1]
    best = oracle_plan(t)
    ids = [a["id"] for a in t["fleet"]]
    worse = {**best, "assets": {i: "hold" for i in ids}}
    assert score_plan(t, worse)["final_reward"] <= score_plan(t, best)["final_reward"]
    # raising in a window that is out is a violation
    t2 = next(d for d in pool if d["windows"]["outage"])
    bad = {"assets": {a["id"]: ("raise_orbit" if a["window"] in t2["windows"]["outage"] else "safe_mode_edge_on")
                      for a in t2["fleet"]}}
    if t2["launch"]:
        bad["launch"] = {"action": "hold"}
    if any(a["window"] in t2["windows"]["outage"] for a in t2["fleet"]):
        s = score_plan(t2, bad)
        assert s["violations"] and s["penalty"] > 0


def test_two_raises_in_one_ground_window_is_a_violation(pool):
    for t in pool:
        by_w = {}
        for a in t["fleet"]:
            by_w.setdefault(a["window"], []).append(a["id"])
        pair = next((v[:2] for v in by_w.values() if len(v) >= 2), None)
        if pair:
            plan = {"assets": {a["id"]: "safe_mode_edge_on" for a in t["fleet"]}}
            for i in pair:
                plan["assets"][i] = "raise_orbit"
            if t["launch"]:
                plan["launch"] = {"action": "hold"}
            assert any("window" in v for v in score_plan(t, plan)["violations"])
            return
    pytest.fail("no shared window found")


def test_missing_unknown_or_malformed_plans_are_rejected(storms):
    t = make(storms, 1)
    ids = [a["id"] for a in t["fleet"]]
    with pytest.raises(ValueError):
        score_plan(t, {"assets": {ids[0]: "raise_orbit"}})                          # missing assets
    with pytest.raises(ValueError):
        score_plan(t, {"assets": {**{i: "hold" for i in ids}, "ghost": "hold"}})    # extra asset
    with pytest.raises(ValueError):
        score_plan(t, {"assets": {i: "teleport" for i in ids}})                     # unknown action


# --- filtering and splits -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset(rows):
    return build_dataset(rows, n_train=8, n_eval=4, seed=0, per_storm_train=2, per_storm_eval=2)


def test_dataset_filters_tasks_the_rule_based_policy_already_solves(dataset):
    train, evals = dataset
    for t in train + evals:
        assert t["baselines"]["rule_final"] < 0.9               # unsolved by the simple rule
        assert score_plan(t, rule_based_plan(t))["final_reward"] == pytest.approx(t["baselines"]["rule_final"])


def test_train_and_eval_never_share_a_storm(dataset):
    train, evals = dataset
    assert train and evals
    assert {t["storm"]["id"] for t in train}.isdisjoint({t["storm"]["id"] for t in evals})
    assert all(t["split"] == "train" for t in train) and all(t["split"] == "eval" for t in evals)
    assert len({t["id"] for t in train + evals}) == len(train + evals)


def test_real_replay_storms_are_in_the_eval_split(rows):
    train, evals = build_dataset(rows, n_train=4, n_eval=6, seed=0, per_storm_train=2, per_storm_eval=3,
                                 eval_storm_dates=["2022-02-03", "2024-05-10"])
    eval_dates = {t["storm"]["onset"][:10] for t in evals}
    assert eval_dates & {"2022-02-03", "2022-02-04", "2024-05-10", "2024-05-11"}
    assert not {t["storm"]["onset"][:10] for t in train} & eval_dates


def test_tasks_roundtrip_through_jsonl(tmp_path, pool):
    tasks = pool[:3]
    save_tasks(tmp_path / "t.jsonl", tasks)
    assert load_tasks(tmp_path / "t.jsonl") == tasks


def test_density_fn_for_scenarios_orders_by_severity(pool):
    t = pool[2]
    peak = lambda s: max(density_fn_for(t, s)(230.0, h) for h in range(0, 49))
    assert peak(1) >= peak(0)                                   # the stronger scenario is at least as dense
