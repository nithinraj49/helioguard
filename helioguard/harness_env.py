"""HelioGuard's plug-in for the model harness: environment, task ids, prompt extras, baselines."""
from __future__ import annotations

from pathlib import Path

from verispace_engine.harness.rollout import EpisodeResult

from .env import HelioGuardEnv
from .policies import (AlwaysRaisePolicy, NeverCallToolsPolicy, OraclePolicy, RuleBasedPolicy,
                       ToolInformedPolicy, run_episode)
from .tasks import load_tasks

TASK_DIR = Path("tasks")
ALLOWED_TYPES = ("tool_call", "submit_decision", "claim_can_gain_altitude", "claim_thrust_slows_decay",
                 "claim_decay")
EXTRA_ACTIONS = (
    '(optional claim) {"type": "claim_decay", "asset_id": ..., "action": ..., "hours": ..., "altitude_loss_km": ...}',
    '(optional claim) {"type": "claim_thrust_slows_decay", "asset_id": ..., "thrust_slows_decay": true|false}',
    '(optional claim) {"type": "claim_can_gain_altitude", "asset_id": ..., "when": "now"|"through_storm", "can_gain_altitude": true|false}',
    'submit example: {"type": "submit_decision", "assets": {"<asset id>": "raise_orbit"|"safe_mode_edge_on"|"hold"}, '
    '"launch": {"action": "hold"} or {"action": "deploy", "alt_km": N}}  (launch only if the briefing has one)',
)
BASELINES = {"rule-based": RuleBasedPolicy, "no-tools": NeverCallToolsPolicy, "tool-informed": ToolInformedPolicy,
             "oracle": OraclePolicy, "always-raise": AlwaysRaisePolicy}


def load(split: str, n: int | None = None) -> tuple[HelioGuardEnv, list[str]]:
    tasks = load_tasks(TASK_DIR / f"{split}.jsonl")
    env = HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20)
    ids = [t["id"] for t in tasks][: n or None]
    return env, ids


def baseline_result(env: HelioGuardEnv, name: str, task_id: str) -> EpisodeResult:
    r = run_episode(env, BASELINES[name](), task_id)
    final = r["final"]
    ok = bool(final) and final.get("error") is None
    return EpisodeResult(task_id, 0, r["return"], final["final_reward"] if ok else -0.5, ok, r["tool_calls"],
                         env._turn, None if ok else "never_submits")


def _round(x, digits: int = 3):
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: _round(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, digits) for v in x]
    return x


def compact_observation(obs):
    """What the model is shown: floats rounded to 3 decimals, and the hourly altitude series of
    simulate_decay replaced by altitudes every 12 h (raw observations are ~1,000 tokens each, which
    made a 10-call episode ~12,000 tokens). Decisions need only final altitude and re-entry hour,
    which are kept exactly (to 3 decimals). The environment and its rewards are untouched."""
    if not isinstance(obs, dict) or obs.get("type") != "tool_result":
        return _round(obs)
    result = dict(obs.get("result") or {})
    times, alts = result.pop("times_h", None), result.pop("altitudes_km", None)
    if times and alts:
        result["altitude_km_at_hours"] = {str(int(round(t))): a for t, a in zip(times, alts)
                                          if abs(t - round(t)) < 1e-6 and int(round(t)) % 12 == 0}
    return _round({**obs, "result": result})
