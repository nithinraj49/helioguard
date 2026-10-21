"""Task generation for the multi-turn environment.

A task is a fleet decision under a real storm: the storm calendar and Kp series are real, the
Kp -> density mapping, spacecraft and altitudes are ILLUSTRATIVE. Each task stores a reward
table for three plausible storm realisations (forecast, stronger, longer) and the exact optimum
(OR-Tools CP-SAT) of the robust objective: the worst total reward over those three realisations.

Plan format (what submit_decision carries):
    {"assets": {asset_id: action}, "launch": {"action": "hold"} | {"action": "deploy", "alt_km": 230}}
`launch` is present exactly when the task has a planned launch.

Per-action reward, always from simulation: lost: 0.5 * (hours survived / horizon); survives:
1 + (final altitude - 150 km) / 250 km. Raising is simulated whether or not thrust beats drag:
with drag above thrust the satellite cannot gain altitude, but thrust can still slow its decay
(raising beats safe mode until drag is about 1.6x thrust). Final reward = 1 - gap/scale (1.0 at
the optimum, 0 when the gap is half a point per decision, floor -1) minus 0.3 per rule violation
(raise in an outage window, two raises in one ground window). No LLM judge.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from .atmosphere import density as reference_density
from .data.noaa import DailyIndices
from .physics import can_raise, simulate
from .scenarios import STARLINK_LIKE
from .storms import Storm, StormProfile, peak_multiplier, storm_catalog
from .tools.qaoa_schedule import ACTION_NAMES
from .verifier import ACTIONS

RAISE, EDGE_ON, HOLD = ACTION_NAMES
DISRUPTIONS = ("forecast_error", "storm_stronger", "storm_longer", "thruster_failure",
               "ground_outage", "launch_slip")
LAUNCH_OPTIONS_KM = [210, 220, 230, 240, 250, 260, 280, 300]
HORIZON_H = 48
HOLD_LAUNCH_REWARD = 0.5               # holding a launch: safe, but a delay
ALT_COST_PER_KM = 0.006                # payload cost of deploying higher, per km above 200
ALT_SCALE_KM = 250.0                   # survivors gain 1 point per this many km above 150 km
VIOLATION_PENALTY = 0.3
RULE_SOLVED = 0.9                      # tasks the rule-based policy scores above this are dropped
MIN_STAKES = 0.25
SCENARIOS = 3                          # forecast, stronger, longer


# --- physics for one task ---------------------------------------------------------------------
def scenario_params(task: dict, s: int) -> tuple[float, float]:
    """(peak multiplier, time stretch) of scenario s: 0 forecast, 1 stronger, 2 longer."""
    f = task["forecast"]
    m = f["peak_multiplier"]
    return [(m, 1.0), (1.0 + (m - 1.0) * f["stronger_factor"], 1.0), (m, f["stretch"])][s]


def density_fn_for(task: dict, s: int, offset_h: float = 0.0):
    """density_fn(alt_km, hours since the forecast) for scenario s, shifted by offset_h."""
    peak, stretch = scenario_params(task, s)
    profile = StormProfile(Storm.from_dict(task["storm"]), task["forecast"]["arrival_h"], peak, stretch)
    return lambda alt_km, t_h: reference_density(alt_km, profile(t_h + offset_h))


def spacecraft_for(thrust_fraction: float):
    return replace(STARLINK_LIKE, thrust_n=STARLINK_LIKE.thrust_n * thrust_fraction)


def launch_time_h(task: dict) -> float:
    return task["launch"]["planned_in_h"] + task["launch"]["slip_h"]


def raise_threshold_km(peak: float) -> float:
    """Altitude above which raising works at a constant density multiplier (full thrust)."""
    lo, hi = 150.0, 800.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if can_raise(STARLINK_LIKE, mid, "raise", peak) else (mid, hi)
    return hi


def _scen(alt_km: float, fn) -> dict:
    return {"alt_km": float(alt_km), "storm_multiplier": 1.0, "horizon_h": HORIZON_H, "density_fn": fn}


def action_rewards(sc, scen: dict) -> dict[str, float]:
    """Reward of each action for one asset in one storm realisation (see the module docstring)."""
    out: dict[str, float] = {}
    for action, (attitude, thrusting) in ACTIONS.items():
        o = simulate(sc, scen["alt_km"], attitude, 1.0, scen["horizon_h"], thrusting=thrusting,
                     density_fn=scen["density_fn"])
        out[action] = (0.5 * o["reentry_hour"] / scen["horizon_h"] if o["reentered"]
                       else 1.0 + (o["final_alt_km"] - 150.0) / ALT_SCALE_KM)
    return out


def _launch_reward(alt: int, fn) -> float:
    return max(action_rewards(STARLINK_LIKE, _scen(alt, fn)).values()) - ALT_COST_PER_KM * (alt - 200)


# --- sampling ---------------------------------------------------------------------------------
def sample_task(storm: Storm, rng: random.Random, task_id: str, split: str) -> dict[str, Any]:
    probs = {"forecast_error": 0.45, "storm_stronger": 0.3, "storm_longer": 0.4,
             "thruster_failure": 0.35, "ground_outage": 0.35, "launch_slip": 0.4}
    has_launch = rng.random() < 0.55
    flags = {name: rng.random() < probs[name] for name in DISRUPTIONS}
    flags["launch_slip"] = flags["launch_slip"] and has_launch
    n_assets = rng.choice([2, 3, 3, 4])
    n_windows = 2 if flags["ground_outage"] else rng.choice([1, 2, 2])

    peak = round(peak_multiplier(storm.kp_max) * rng.uniform(0.9, 1.1), 3)
    disruptions: list[dict[str, Any]] = []
    stronger, stretch = 1.2, 1.25
    if flags["forecast_error"]:
        stronger = max(stronger, rng.uniform(1.5, 2.0))
        disruptions.append({"type": "forecast_error"})
    if flags["storm_stronger"]:
        stronger = max(stronger, rng.uniform(1.9, 2.6))
        disruptions.append({"type": "storm_stronger"})
    if flags["storm_longer"]:
        stretch = rng.uniform(1.8, 2.6)
        disruptions.append({"type": "storm_longer"})

    # Asset altitudes sit where the decision is hard: raising works in quiet air but not at the
    # storm's peak (forecast or stronger), so the answer depends on timing and forecast error.
    zone_lo = raise_threshold_km(1.0) - 4.0
    zone_hi = raise_threshold_km(1.0 + (peak - 1.0) * stronger) + 8.0
    fleet = [{"id": f"b{i + 1}", "alt_km": float(5 * round(rng.uniform(zone_lo, zone_hi) / 5)),
              "window": rng.randrange(n_windows), "thrust_fraction": 1.0} for i in range(n_assets)]
    if flags["thruster_failure"]:
        a = rng.choice(fleet)
        a["thrust_fraction"] = round(rng.uniform(0.25, 0.65), 2)
        disruptions.append({"type": "thruster_failure", "asset_id": a["id"],
                            "thrust_fraction": a["thrust_fraction"]})
    outage: list[int] = []
    if flags["ground_outage"]:
        outage = [rng.randrange(n_windows)]
        disruptions.append({"type": "ground_outage", "window": outage[0]})
    launch = None
    if has_launch:
        slip = rng.randrange(6, 25, 2) if flags["launch_slip"] else 0
        launch = {"id": f"L-{rng.randrange(10, 99)}", "planned_in_h": rng.randrange(6, 31, 2),
                  "slip_h": slip, "options_km": list(LAUNCH_OPTIONS_KM)}
        if slip:
            disruptions.append({"type": "launch_slip", "slip_h": slip})
    task: dict[str, Any] = {
        "id": task_id, "split": split, "storm": storm.to_dict(),
        "forecast": {"arrival_h": float(rng.randrange(2, 15, 2)), "peak_multiplier": peak,
                     "horizon_h": HORIZON_H, "stronger_factor": round(stronger, 3),
                     "stretch": round(stretch, 3)},
        "fleet": fleet, "windows": {"all": list(range(n_windows)), "outage": outage},
        "launch": launch, "disruptions": disruptions,
    }
    _fill_physics(task)
    return task


def _fill_physics(task: dict) -> None:
    fns = [density_fn_for(task, s) for s in range(SCENARIOS)]
    assets: dict[str, dict[str, list[float]]] = {}
    can_gain: dict[str, bool] = {}
    slows_decay: dict[str, bool] = {}
    for a in task["fleet"]:
        sc = spacecraft_for(a["thrust_fraction"])
        per_scen = [action_rewards(sc, _scen(a["alt_km"], fns[s])) for s in range(SCENARIOS)]
        assets[a["id"]] = {act: [per_scen[s][act] for s in range(SCENARIOS)] for act in ACTION_NAMES}
        can_gain[a["id"]] = bool(can_raise(sc, a["alt_km"], "raise", 1.0, fns[0]))      # thrust > drag now
        r0 = per_scen[0]                                  # forecast storm: raise outlives the others
        slows_decay[a["id"]] = r0[RAISE] > max(r0[EDGE_ON], r0[HOLD])
    rewards: dict[str, Any] = {"assets": assets}
    nominal: dict[str, Any] = {"can_gain_altitude": can_gain, "thrust_slows_decay": slows_decay}
    if task["launch"]:
        t_l = launch_time_h(task)
        launch_rewards: dict[str, list[float]] = {"hold": [HOLD_LAUNCH_REWARD] * SCENARIOS}
        launch_feasible: dict[str, bool] = {}
        for opt in task["launch"]["options_km"]:
            shifted = [density_fn_for(task, s, offset_h=t_l) for s in range(SCENARIOS)]
            launch_rewards[str(opt)] = [_launch_reward(opt, shifted[s]) for s in range(SCENARIOS)]
            launch_feasible[str(opt)] = bool(can_raise(STARLINK_LIKE, opt, "raise", 1.0, shifted[0]))
        rewards["launch"] = launch_rewards
        nominal["launch_can_gain_altitude"] = launch_feasible
    task["rewards"], task["nominal"] = rewards, nominal
    task["needs_qaoa"] = _needs_qaoa(task)
    task["optimal"] = solve_optimal(task)
    task["baselines"] = _baselines(task)


def _needs_qaoa(task: dict) -> bool:
    """True when two assets contend for one ground window and both would best raise."""
    by_window: dict[int, int] = {}
    for a in task["fleet"]:
        r = task["rewards"]["assets"][a["id"]]
        wants_raise = (a["window"] not in task["windows"]["outage"] and
                       r[RAISE][0] >= max(r[EDGE_ON][0], r[HOLD][0]))
        if wants_raise:
            by_window[a["window"]] = by_window.get(a["window"], 0) + 1
    return any(n >= 2 for n in by_window.values())


# --- plans, scoring and the exact solver -------------------------------------------------------
def _check_plan(task: dict, plan: dict) -> tuple[dict[str, str], str | None]:
    ids = [a["id"] for a in task["fleet"]]
    assets = plan.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ids):
        raise ValueError(f"plan must give exactly one action for each of {ids}")
    for i, act in assets.items():
        if act not in ACTION_NAMES:
            raise ValueError(f"unknown action '{act}' for {i}")
    launch_key = None
    if task["launch"]:
        launch = plan.get("launch")
        if not isinstance(launch, dict) or launch.get("action") not in ("hold", "deploy"):
            raise ValueError("plan needs launch: hold or deploy with alt_km")
        if launch["action"] == "hold":
            launch_key = "hold"
        else:
            alt = launch.get("alt_km")
            if alt not in task["launch"]["options_km"]:
                raise ValueError(f"deploy altitude must be one of {task['launch']['options_km']}")
            launch_key = str(alt)
    elif "launch" in plan:
        raise ValueError("this task has no planned launch")
    return dict(assets), launch_key


def _violations(task: dict, assets: dict[str, str]) -> list[str]:
    out, raisers = [], {}
    for a in task["fleet"]:
        if assets[a["id"]] != RAISE:
            continue
        if a["window"] in task["windows"]["outage"]:
            out.append(f"raise {a['id']} in ground window {a['window']} that is out")
        raisers.setdefault(a["window"], []).append(a["id"])
    out += [f"two raises in ground window {w}: {ids}" for w, ids in raisers.items() if len(ids) > 1]
    return out


def _totals(task: dict, assets: dict[str, str], launch_key: str | None) -> list[float]:
    clash = {w for w, n in _raise_counts(task, assets).items() if n > 1}
    totals = []
    for s in range(SCENARIOS):
        t = 0.0
        for a in task["fleet"]:
            act = assets[a["id"]]
            bad = act == RAISE and (a["window"] in task["windows"]["outage"] or a["window"] in clash)
            t += -1.0 if bad else task["rewards"]["assets"][a["id"]][act][s]
        if launch_key is not None:
            t += task["rewards"]["launch"][launch_key][s]
        totals.append(t)
    return totals


def _raise_counts(task: dict, assets: dict[str, str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for a in task["fleet"]:
        if assets[a["id"]] == RAISE:
            counts[a["window"]] = counts.get(a["window"], 0) + 1
    return counts


def hold_objective(task: dict) -> float:
    assets = {a["id"]: HOLD for a in task["fleet"]}
    return min(_totals(task, assets, "hold" if task["launch"] else None))


def score_plan(task: dict, plan: dict) -> dict[str, Any]:
    """Robust objective, gap to the proven optimum and final reward. Raises ValueError on a
    malformed plan (the environment turns that into a penalty)."""
    assets, launch_key = _check_plan(task, plan)
    totals = _totals(task, assets, launch_key)
    objective = min(totals)
    opt = task["optimal"]
    gap = opt["objective"] - objective
    r_gap = max(-1.0, 1.0 - gap / opt["scale"])
    violations = _violations(task, assets)
    penalty = VIOLATION_PENALTY * len(violations)
    return {"objective": objective, "per_scenario": totals, "optimal_objective": opt["objective"],
            "gap": gap, "r_gap": r_gap, "violations": violations, "penalty": penalty,
            "final_reward": max(-2.0, r_gap - penalty)}


def _plan_dict(assets: dict[str, str], launch_key: str | None) -> dict[str, Any]:
    plan: dict[str, Any] = {"assets": dict(assets)}
    if launch_key is not None:
        plan["launch"] = {"action": "hold"} if launch_key == "hold" else {"action": "deploy",
                                                                           "alt_km": int(launch_key)}
    return plan


def solve_optimal(task: dict) -> dict[str, Any]:
    """Exact solution (OR-Tools CP-SAT) of: maximise the worst-case total reward over the three
    storm realisations, subject to the rules (one action per asset, no raise in an outage window,
    at most one raise per ground window). Raising with drag above thrust is allowed."""
    from ortools.sat.python import cp_model
    S = 1_000_000
    model = cp_model.CpModel()
    ids = [a["id"] for a in task["fleet"]]
    x = {(i, act): model.NewBoolVar(f"x_{i}_{act}") for i in ids for act in ACTION_NAMES}
    for i in ids:
        model.AddExactlyOne(x[i, act] for act in ACTION_NAMES)
    by_window: dict[int, list[str]] = {}
    for a in task["fleet"]:
        by_window.setdefault(a["window"], []).append(a["id"])
        if a["window"] in task["windows"]["outage"]:
            model.Add(x[a["id"], RAISE] == 0)
    for members in by_window.values():
        model.AddAtMostOne(x[i, RAISE] for i in members)
    launch_keys = list(task["rewards"]["launch"]) if task["launch"] else []
    y = {k: model.NewBoolVar(f"y_{k}") for k in launch_keys}
    if y:
        model.AddExactlyOne(y.values())
    z = model.NewIntVar(-10 ** 10, 10 ** 10, "z")
    for s in range(SCENARIOS):
        expr = sum(round(task["rewards"]["assets"][i][act][s] * S) * x[i, act] for i in ids for act in ACTION_NAMES)
        expr += sum(round(task["rewards"]["launch"][k][s] * S) * y[k] for k in launch_keys)
        model.Add(z <= expr)
    model.Maximize(z)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("no feasible plan")
    assets = {i: next(act for act in ACTION_NAMES if solver.Value(x[i, act])) for i in ids}
    key = next((k for k in launch_keys if solver.Value(y[k])), None)
    objective = min(_totals(task, assets, key))
    j_hold = hold_objective(task)
    return {"plan": _plan_dict(assets, key), "objective": objective, "status": solver.StatusName(status),
            "hold_objective": j_hold, "stakes": objective - j_hold,
            "scale": 0.5 * (len(ids) + (1 if task["launch"] else 0))}


def oracle_plan(task: dict) -> dict[str, Any]:
    return json.loads(json.dumps(task["optimal"]["plan"]))


def random_plan(task: dict, rng: random.Random) -> dict[str, Any]:
    assets = {a["id"]: rng.choice(ACTION_NAMES) for a in task["fleet"]}
    key = rng.choice(list(task["rewards"]["launch"])) if task["launch"] else None
    return _plan_dict(assets, key)


def rule_based_plan(task: dict) -> dict[str, Any]:
    """Simple baseline: raise whenever thrust beats drag right now and the ground window is
    free, otherwise safe mode; deploy at the lowest altitude where thrust beats drag at
    deployment, else hold the launch. It ignores forecast uncertainty, the arrival time of the
    storm, and the fact that thrust can slow decay even when it cannot gain altitude."""
    used, assets = set(), {}
    outage = set(task["windows"]["outage"])
    for a in task["fleet"]:
        if (task["nominal"]["can_gain_altitude"][a["id"]] and a["window"] not in outage
                and a["window"] not in used):
            assets[a["id"]] = RAISE
            used.add(a["window"])
        else:
            assets[a["id"]] = EDGE_ON
    key = None
    if task["launch"]:
        gains = task["nominal"]["launch_can_gain_altitude"]
        key = next((str(o) for o in task["launch"]["options_km"] if gains[str(o)]), "hold")
    return _plan_dict(assets, key)


def _baselines(task: dict) -> dict[str, float]:
    rng = random.Random(f"baseline:{task['id']}")
    rand = [score_plan(task, random_plan(task, rng))["final_reward"] for _ in range(20)]
    return {"rule_final": score_plan(task, rule_based_plan(task))["final_reward"],
            "random_final": statistics.fmean(rand)}


# --- dataset ------------------------------------------------------------------------------------
def _accepted(task: dict) -> bool:
    return task["baselines"]["rule_final"] < RULE_SOLVED and task["optimal"]["stakes"] >= MIN_STAKES


def _storm_tasks(job: tuple) -> list[dict]:
    """Accepted tasks for one storm. Own RNG per storm, so results do not depend on parallelism."""
    storm_dict, split, per_storm, attempts, seed = job
    storm = Storm.from_dict(storm_dict)
    rng = random.Random(f"{seed}:{split}:{storm.id}")
    out: list[dict] = []
    for k in range(attempts):
        if len(out) == per_storm:
            break
        t = sample_task(storm, rng, f"{split}_{storm.id}_{k}", split)
        if _accepted(t):
            out.append(t)
    return out


def build_dataset(rows: Sequence[DailyIndices], n_train: int, n_eval: int, seed: int = 0,
                  per_storm_train: int = 5, per_storm_eval: int = 2,
                  eval_storm_dates: Sequence[str] = (), attempts_factor: int = 6, workers: int = 1):
    """(train, eval) tasks. A storm is in exactly one split. Tasks the rule-based policy already
    solves (final reward >= 0.9) and tasks with little at stake are filtered out."""
    storms = storm_catalog(rows)
    rng = random.Random(seed)
    forced_days = {(date.fromisoformat(d) + timedelta(days=k)).isoformat()
                   for d in eval_storm_dates for k in (-1, 0, 1)}
    forced = [s for s in storms if s.onset.date().isoformat() in forced_days]
    rest = [s for s in storms if s not in forced]
    rng.shuffle(rest)
    n_eval_storms = max(len(forced), 2 * math.ceil(n_eval / per_storm_eval))
    eval_storms = forced + rest[:max(0, n_eval_storms - len(forced))]
    train_storms = rest[max(0, n_eval_storms - len(forced)):]

    def collect(pool: list[Storm], split: str, want: int, per_storm: int) -> list[dict]:
        jobs = [(s.to_dict(), split, per_storm, per_storm * attempts_factor, seed) for s in pool]
        out: list[dict] = []
        batch = workers * 4 if workers > 1 else 1
        executor = None
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(max_workers=workers)
        try:
            for i in range(0, len(jobs), batch):
                chunk = jobs[i:i + batch]
                results = executor.map(_storm_tasks, chunk) if executor else map(_storm_tasks, chunk)
                for tasks in results:
                    out += tasks
                if len(out) >= want:
                    break
        finally:
            if executor:
                executor.shutdown()
        return out[:want]

    return (collect(train_storms, "train", n_train, per_storm_train),
            collect(eval_storms, "eval", n_eval, per_storm_eval))


def save_tasks(path: Path, tasks: Sequence[dict]) -> None:
    Path(path).write_text("".join(json.dumps(t, separators=(",", ":")) + "\n" for t in tasks),
                          encoding="utf-8")


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
