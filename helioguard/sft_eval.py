"""Evaluation hook for the SFT adapter: Phase 5 harness on the held-out tasks, bootstrap CIs, gates.

    python -m helioguard.sft_eval --model Qwen/Qwen2.5-1.5B-Instruct --adapter results/phase6/sft_1p5b/adapter

Conditions (docs/phase6_gates.md): greedy (primary, 1 run: no seed variance) and temperature 0.7 with
seeds 0, 1, 2. 95 percent percentile bootstrap over tasks (10,000 resamples, seed 0); the seeds of a
task are averaged first. Gates are decided on greedy.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from verispace_engine.harness.rollout import EpisodeResult, RolloutConfig, rollout_task
from verispace_engine.harness.stats import bootstrap_ci

from .harness_env import ALLOWED_TYPES, EXTRA_ACTIONS, compact_observation, load
from .sft_traces import OPTIMAL

NO_TOOL_PLAN = 0.743          # gate B reference (Phase 5, 50 eval tasks)
RULE_BASED = 0.727            # gate C reference
GATE_A_MIN_VALID = 0.90


def per_task(results: Sequence[EpisodeResult], key) -> dict[str, float]:
    """Mean of key(result) over the attempts of each task."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in results:
        acc[r.task_id].append(float(key(r)))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def _ci(values: dict[str, float]) -> dict[str, float]:
    mean, lo, hi = bootstrap_ci(list(values.values()))
    return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def summarize_condition(results: Sequence[EpisodeResult], tasks: Sequence[dict]) -> dict[str, Any]:
    """Primary: valid-submission rate. Secondary: mean final reward, percent optimal, tool calls, per-disruption."""
    metrics = {
        "valid_submission_rate": lambda r: r.valid_submission,
        "mean_final_reward": lambda r: r.final_reward,
        "percent_optimal": lambda r: r.valid_submission and r.final_reward >= OPTIMAL,
        "tool_calls_per_episode": lambda r: r.tool_calls,
    }
    out: dict[str, Any] = {"episodes": len(results), "tasks": len({r.task_id for r in results}),
                           **{name: _ci(per_task(results, fn)) for name, fn in metrics.items()}}
    out["failure_types"] = {}
    for r in results:
        if r.failure:
            out["failure_types"][r.failure] = out["failure_types"].get(r.failure, 0) + 1
    by_task = {t["id"]: t for t in tasks}
    groups: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        kinds = sorted({d["type"] for d in by_task[r.task_id]["disruptions"]}) or ["none"]
        for k in kinds:
            groups[k].append(r)
    out["per_disruption"] = {
        k: {"tasks": len({r.task_id for r in rs}),
            "valid_submission_rate": round(sum(r.valid_submission for r in rs) / len(rs), 3),
            "mean_final_reward": round(sum(r.final_reward for r in rs) / len(rs), 3),
            "percent_optimal": round(sum(r.valid_submission and r.final_reward >= OPTIMAL for r in rs) / len(rs), 3)}
        for k, rs in sorted(groups.items())}
    return out


def paired_vs_baseline(results: Sequence[EpisodeResult], baseline: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Secondary analysis (not a gate): paired bootstrap of the per-task difference in final reward,
    model minus baseline (the no-tool plan under the compact view). Tasks are the unit of resampling."""
    m, b = per_task(results, lambda r: r.final_reward), per_task(baseline, lambda r: r.final_reward)
    shared = sorted(set(m) & set(b))
    diff = [m[t] - b[t] for t in shared]
    mean, lo, hi = bootstrap_ci(diff)
    return {"tasks": len(shared), "mean_difference": round(mean, 4), "ci_low": round(lo, 4),
            "ci_high": round(hi, 4), "ci_excludes_zero_above": bool(lo > 0)}


def _submitted_assets(log_path: str | Path) -> dict[str, dict[str, str]]:
    """task id -> {asset: action} of the last submit_decision in a harness JSONL log (attempt 0)."""
    plans: dict[str, dict[str, str]] = {}
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        act = rec.get("action") if rec.get("event") == "turn" else None
        if act and act.get("type") == "submit_decision" and rec.get("attempt", 0) == 0:
            plans[rec["task_id"]] = act.get("assets", {})
    return plans


def quiet_task_action_mix(log_path: str | Path, tasks: Sequence[dict]) -> dict[str, Any]:
    """Descriptive: on tasks with no disruption listed, how often each action is chosen per asset,
    next to the stored optimum's mix. Not a gate."""
    plans = _submitted_assets(log_path)
    quiet = [t for t in tasks if not t["disruptions"]]
    model_mix: dict[str, int] = defaultdict(int)
    optimum_mix: dict[str, int] = defaultdict(int)
    for t in quiet:
        for a in t["optimal"]["plan"]["assets"].values():
            optimum_mix[a] += 1
        for a in plans.get(t["id"], {}).values():
            model_mix[str(a)] += 1
    return {"quiet_tasks": len(quiet), "model_actions": dict(model_mix), "optimum_actions": dict(optimum_mix),
            "tasks_with_a_submission": sum(t["id"] in plans for t in quiet)}


def gate_report(greedy: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered gates (docs/phase6_gates.md) to the greedy summary."""
    valid = greedy["valid_submission_rate"]["mean"]
    reward = greedy["mean_final_reward"]
    a = valid >= GATE_A_MIN_VALID
    b = reward["ci_low"] > NO_TOOL_PLAN
    if reward["mean"] < RULE_BASED or reward["ci_low"] <= RULE_BASED <= reward["ci_high"]:
        below = reward["mean"] < RULE_BASED
        c = (f"Gate C: mean final reward {reward['mean']:+.3f} (95% CI {reward['ci_low']:+.3f} to "
             f"{reward['ci_high']:+.3f}) is {'below' if below else 'not distinguishable from'} "
             f"the rule-based plan ({RULE_BASED:+.3f}): the SFT model "
             f"{'fails to beat' if below else 'only matches'} the rule-based plan.")
    else:
        c = (f"Gate C: mean final reward {reward['mean']:+.3f} is above the rule-based plan ({RULE_BASED:+.3f}) "
             f"with the CI excluding it.")
    return {"gate_A_format": {"pass": bool(a), "valid_submission_rate": valid, "needed": GATE_A_MIN_VALID},
            "gate_B_value": {"pass": bool(b), "ci_low": reward["ci_low"], "needed_above": NO_TOOL_PLAN},
            "gate_C_honesty": c}


class FlushLog:
    """JSONL log that appends each record and flushes it to disk at once (survives a killed runtime)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


def load_episodes(path: str | Path) -> list[EpisodeResult]:
    """Finished episodes saved in a harness JSONL file (a half-written last line is ignored)."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "episode":
            out.append(EpisodeResult(**{k: v for k, v in rec.items() if k in EpisodeResult.__dataclass_fields__}))
    return out


def _drop_unfinished_turns(path: Path) -> None:
    """On restart, remove the turn records of an episode that never finished (it will be re-run in full)."""
    if not path.exists():
        return
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    done = {(r["task_id"], r.get("attempt", 0)) for r in recs if r.get("event") == "episode"}
    keep = [r for r in recs if (r.get("task_id"), r.get("attempt", 0)) in done]
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in keep), encoding="utf-8")


def condition_files(out_dir: str | Path, tag: str, seeds: Sequence[int], greedy_only: bool = False) -> dict[str, Path]:
    """Greedy first, then temperature 0.7 with one file per seed."""
    out_dir = Path(out_dir)
    files = {"greedy": out_dir / f"{tag}_greedy.jsonl"}
    if not greedy_only:
        files.update({f"t0.7_seed{s}": out_dir / f"{tag}_t0.7_seed{s}.jsonl" for s in seeds})
    return files


def aggregate(out_dir: str | Path, tag: str, tasks: Sequence[dict], seeds: Sequence[int] = (0, 1, 2),
              greedy_only: bool = False, baseline_dir: str | Path = "results/phase6/baselines_compact",
              model: str = "unknown") -> dict[str, Any]:
    """Summary built from the saved episode files alone (no model needed). Partial files give a partial summary:
    `complete` is False, the counts say how far each condition got, and no gates are issued."""
    files = condition_files(out_dir, tag, seeds, greedy_only)
    ids = [t["id"] for t in tasks]
    got = {name: load_episodes(p) for name, p in files.items()}
    progress = {name: {"episodes": len(r), "expected": len(ids)} for name, r in got.items()}
    complete = all(len({x.task_id for x in r}) == len(ids) for r in got.values())
    summary: dict[str, Any] = {"model": model, "tasks": len(ids), "complete": complete, "progress": progress}
    greedy = got["greedy"]
    if greedy:
        summary["greedy"] = summarize_condition(greedy, tasks)
    sampled = [r for name, rs in got.items() if name != "greedy" for r in rs]
    if sampled:
        summary["temperature_0.7_3_seeds"] = summarize_condition(sampled, tasks)
    if complete:
        summary["gates"] = gate_report(summary["greedy"])       # decided only on a complete greedy run
    else:
        summary["gates"] = None
        summary["note"] = "partial: gates are issued only when every condition has finished all its episodes"
    ref = Path(baseline_dir) / "baseline_no-tools_eval.jsonl"
    if ref.exists() and greedy:                                 # secondary analyses (Amendment 1)
        base_eps = load_episodes(ref)
        summary["secondary"] = {"greedy_paired_vs_no_tool_plan": paired_vs_baseline(greedy, base_eps),
                                **({"temperature_0.7_paired_vs_no_tool_plan": paired_vs_baseline(sampled, base_eps)}
                                   if sampled else {})}
        if summary["gates"]:
            paired_ok = summary["secondary"]["greedy_paired_vs_no_tool_plan"]["ci_excludes_zero_above"]
            if paired_ok != summary["gates"]["gate_B_value"]["pass"]:
                summary["secondary"]["note"] = ("paired and unpaired Gate B disagree: unpaired "
                                                f"{summary['gates']['gate_B_value']['pass']}, paired {paired_ok}; both reported")
    if greedy:
        summary["secondary"] = {**summary.get("secondary", {}),
                                "quiet_task_action_mix_greedy": quiet_task_action_mix(files["greedy"], tasks)}
    return summary


def _upload_eval(hub, out_dir: Path, tag: str, message: str) -> None:
    files = {f"eval/{p.name}": p for p in sorted(out_dir.glob(f"{tag}_*")) if p.is_file()}
    if files:
        hub.upload(files, message)


def evaluate(backend, n: int = 50, seeds: Sequence[int] = (0, 1, 2), out_dir: str | Path = "results/phase6",
             tag: str = "sft", max_new_tokens: int = 256, greedy_only: bool = False,
             baseline_dir: str | Path = "results/phase6/baselines_compact",
             hub=None, upload_every: int = 10) -> dict[str, Any]:
    """Resumable: every finished episode is appended (and fsynced) to its file at once; episodes already in the
    files are skipped on restart; with `hub`, files are uploaded and verified every `upload_every` new episodes
    and at the end of each condition. The summary is aggregated from the saved files alone."""
    env, ids = load("eval", n)
    tasks = [env._by_id[i] for i in ids]
    base = dict(max_new_tokens=max_new_tokens, allowed_types=tuple(ALLOWED_TYPES),
                extra_actions=tuple(EXTRA_ACTIONS), compactor=compact_observation)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = condition_files(out_dir, tag, seeds, greedy_only)
    configs = {"greedy": RolloutConfig(attempts=1, temperature=0.0, seed=0, **base),
               **{f"t0.7_seed{s}": RolloutConfig(attempts=1, temperature=0.7, seed=s, **base) for s in seeds}}
    new_since_upload = 0
    for name, path in files.items():                            # greedy first, then each seed in its own file
        _drop_unfinished_turns(path)
        done = {(r.task_id, r.attempt) for r in load_episodes(path)}
        log = FlushLog(path)
        for tid in ids:
            if (tid, 0) in done:
                continue
            r = rollout_task(env, tid, backend, configs[name], 0, log)
            new_since_upload += 1
            print(f"[{name}] {tid}: final {r.final_reward:+.3f} tools {r.tool_calls} {r.failure or 'ok'} "
                  f"({len(load_episodes(path))}/{len(ids)} saved)", flush=True)
            if hub is not None and new_since_upload >= upload_every:
                _upload_eval(hub, out_dir, tag, f"eval {tag}: {name} {len(load_episodes(path))}/{len(ids)}")
                new_since_upload = 0
        if hub is not None and new_since_upload:
            _upload_eval(hub, out_dir, tag, f"eval {tag}: {name} finished")
            new_since_upload = 0
    summary = aggregate(out_dir, tag, tasks, seeds, greedy_only, baseline_dir, model=backend.name)
    (out_dir / f"{tag}_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if hub is not None:
        _upload_eval(hub, out_dir, tag, f"eval {tag}: summary")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--base-adapter", default=None, help="adapter merged first (the SFT adapter when --adapter is a GVPO adapter)")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", default="results/phase6")
    p.add_argument("--tag", default="sft")
    p.add_argument("--device", default=None)
    p.add_argument("--greedy-only", action="store_true", help="skip the temperature 0.7 runs (integration checks)")
    p.add_argument("--hub-repo", default=None, help="private Hub repo for periodic eval uploads (token from HF_TOKEN)")
    p.add_argument("--upload-every", type=int, default=10, help="episodes between Hub uploads")
    p.add_argument("--aggregate-only", action="store_true", help="summarise the saved episode files; load no model")
    a = p.parse_args(argv)
    if a.aggregate_only:
        env, ids = load("eval", a.n)
        summary = aggregate(a.out, a.tag, [env._by_id[i] for i in ids], greedy_only=a.greedy_only, model=a.model)
        (Path(a.out) / f"{a.tag}_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        hub = None
        if a.hub_repo:
            from verispace_engine.hub import HubSync
            hub = HubSync(a.hub_repo)
        from verispace_engine.harness.backends import HFBackend
        summary = evaluate(HFBackend(a.model, device=a.device, adapter=a.adapter, base_adapter=a.base_adapter), a.n, out_dir=a.out, tag=a.tag,
                           greedy_only=a.greedy_only, hub=hub, upload_every=a.upload_every)
    print(json.dumps(summary["gates"], indent=2))
    return summary


if __name__ == "__main__":
    main()
