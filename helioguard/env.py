"""HelioGuard multi-turn environment on verispace_engine.EnvBase.

An episode is one fleet decision under a storm forecast. reset() returns a scenario prompt; the
agent may call tools, make verifiable claims, and ends the episode with submit_decision.

Actions (dicts):
  {"type": "tool_call", "name": ..., "arguments": {...}}
  {"type": "claim_can_gain_altitude", "asset_id", "when": "now"|"through_storm", "can_gain_altitude": bool}
  {"type": "claim_thrust_slows_decay", "asset_id", "thrust_slows_decay": bool}
  {"type": "claim_decay", "asset_id", "action", "hours", "altitude_loss_km"}
  {"type": "submit_decision", "assets": {id: action}, "launch": {...}}     (ends the episode)

Rewards (no LLM judge; all from physics and the exact solver):
  tool call:    +0.05 if new and useful, -0.05 if invalid or a duplicate, minus a cost per call
                (0.02; 0.10 for qaoa_schedule). Over the cap: -0.10, not executed.
  claim:        +0.10 if true, -0.10 if false, -0.05 if repeated. Claims only get a true/false answer.
  malformed or unknown action: -0.05.
  submit:       the final reward from tasks.score_plan (1.0 at the exact optimum, falling with the
                gap, -0.3 per rule violation) minus 0.2 for each raise ordered although check_feasibility
                said thrust_slows_decay is false. Drag above thrust alone is not penalised: raising
                can still be the best choice. A malformed submission ends the episode at -0.5.
  no submission within max_turns: -0.5 and the episode ends.
HelioGuard recommends; humans decide.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from verispace_engine.env_base import EnvBase, StepOutcome
from verispace_engine.schema import ToolCall

from .env_tools import DEFAULT_TOOL_COST, TOOL_COSTS, FleetToolbox
from .env_verifier import EnvVerifier, EpisodeMemory, call_key
from .tasks import score_plan

USEFUL_CALL_BONUS = 0.05
BAD_CALL_PENALTY = -0.05
OVER_CAP_PENALTY = -0.10
CLAIM_REWARD = 0.10
REPEAT_CLAIM_PENALTY = -0.05
BAD_ACTION_PENALTY = -0.05
MALFORMED_SUBMIT = -0.5
TIMEOUT_PENALTY = -0.5
IGNORED_RESULT_PENALTY = 0.2


def render_prompt(task: dict, tool_specs: Sequence[dict]) -> str:
    f, storm = task["forecast"], task["storm"]
    worst = 1.0 + (f["peak_multiplier"] - 1.0) * f["stronger_factor"]
    lines = [
        "HELIOGUARD STORM BRIEFING (illustrative values; HelioGuard recommends, a human operator decides)",
        "",
        f"Forecast: geomagnetic storm, Kp {storm['kp_max']:.1f} ({storm['g_scale']}), arrival in "
        f"{f['arrival_h']:.0f} h, peak upper-atmosphere density about x{f['peak_multiplier']:.2f} of quiet. "
        f"Planning horizon {f['horizon_h']} h. (Replayed from the record of {storm['onset'][:10]}.)",
        f"Forecast uncertainty: the peak could reach x{worst:.2f} and the storm could last "
        f"{f['stretch']:.1f}x longer than forecast. A good plan holds up across that range.",
        "",
        "Fleet (each asset needs one action: raise_orbit, safe_mode_edge_on or hold):",
    ]
    for a in task["fleet"]:
        lines.append(f"  {a['id']}: {a['alt_km']:.0f} km, ground window {a['window']}, "
                     f"thrust {a['thrust_fraction'] * 100:.0f}% of nominal")
    out = task["windows"]["outage"]
    lines.append(f"Ground windows: {', '.join(map(str, task['windows']['all']))}. Raising needs a ground "
                 f"pass, and a window serves at most one raise."
                 + (f" Window(s) {', '.join(map(str, out))} are out." if out else ""))
    if task["launch"]:
        L = task["launch"]
        slip = f" (slipped by {L['slip_h']} h)" if L["slip_h"] else ""
        lines += ["", f"Planned launch {L['id']} in {L['planned_in_h']} h{slip}. Decide: hold, or deploy at "
                      f"one of {L['options_km']} km. Deploying higher costs payload; deploying where "
                      f"satellites cannot survive loses them; holding delays the launch."]
    if task["disruptions"]:
        lines += ["", "Disruptions in force:"]
        for d in task["disruptions"]:
            t = d["type"]
            if t == "thruster_failure":
                lines.append(f"  - thruster failure: {d['asset_id']} has {d['thrust_fraction'] * 100:.0f}% thrust")
            elif t == "ground_outage":
                lines.append(f"  - ground outage: window {d['window']} is unavailable")
            elif t == "launch_slip":
                lines.append(f"  - launch slip: the launch moved later by {d['slip_h']} h")
            elif t == "forecast_error":
                lines.append("  - forecast error: the storm strength forecast is poorly constrained")
            elif t == "storm_stronger":
                lines.append("  - storm stronger than forecast is a real possibility")
            elif t == "storm_longer":
                lines.append("  - storm longer than forecast is a real possibility")
    lines += ["", "Tools (call with {\"type\": \"tool_call\", \"name\": ..., \"arguments\": {...}}):"]
    for s in tool_specs:
        lines.append(f"  {s['name']}: {s['description']}")
    lines += ["", "You may also state checkable claims (claim_can_gain_altitude, claim_thrust_slows_decay, "
                  "claim_decay). Finish with "
                  "{\"type\": \"submit_decision\", \"assets\": {id: action, ...}" +
              (", \"launch\": {\"action\": \"hold\"} or {\"action\": \"deploy\", \"alt_km\": N}" if task["launch"] else "") +
              "}. Tool calls are limited and each costs a little."]
    return "\n".join(lines)


class HelioGuardEnv(EnvBase):
    def __init__(self, tasks: Sequence[dict], seed: int = 0, max_tool_calls: int = 10, max_turns: int = 20) -> None:
        if not tasks:
            raise ValueError("need at least one task")
        self.tasks = list(tasks)
        self._by_id = {t["id"]: t for t in self.tasks}
        self.max_tool_calls, self.max_turns = max_tool_calls, max_turns
        self.verifier = EnvVerifier()
        self._cursor = seed % len(self.tasks)
        self.task: dict | None = None
        self._box: FleetToolbox | None = None
        self._done = True
        self._turn = 0
        self.executed_tool_calls = 0
        self._memory = EpisodeMemory()

    # -- engine API -----------------------------------------------------------------------------
    def reset(self, task_id: str | None = None) -> str:
        if task_id is None:
            task = self.tasks[self._cursor % len(self.tasks)]
            self._cursor += 1
        elif task_id in self._by_id:
            task = self._by_id[task_id]
        else:
            raise ValueError(f"unknown task '{task_id}'")
        self.task, self._box = task, FleetToolbox(task)
        self._memory, self._turn, self.executed_tool_calls, self._done = EpisodeMemory(), 0, 0, False
        return render_prompt(task, self.tool_specs())

    def tool_names(self) -> list[str]:
        return self._need_box().registry.names()

    def tool_specs(self) -> list[dict]:
        return self._need_box().registry.specs()

    def step(self, action: Any) -> StepOutcome:
        box = self._need_box()
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        self._turn += 1
        kind = action.get("type") if isinstance(action, dict) else None
        if kind == "tool_call":
            out = self._tool_call(action, box)
        elif kind in ("claim_can_gain_altitude", "claim_thrust_slows_decay", "claim_decay"):
            out = self._claim(action, kind, box)
        elif kind == "submit_decision":
            return self._submit(action)
        else:
            out = self._error(f"unknown or malformed action {kind!r}", BAD_ACTION_PENALTY)
        if self._turn >= self.max_turns:
            self._done = True
            out = StepOutcome(out.observation, out.reward + TIMEOUT_PENALTY, True,
                              {**out.info, "timeout": True})
        return out

    # -- internals --------------------------------------------------------------------------------
    def _need_box(self) -> FleetToolbox:
        if self._box is None:
            raise RuntimeError("call reset() first")
        return self._box

    def _obs(self, kind: str, **extra: Any) -> dict[str, Any]:
        return {"type": kind, "turn": self._turn,
                "tool_calls_left": max(0, self.max_tool_calls - self.executed_tool_calls), **extra}

    def _error(self, message: str, reward: float) -> StepOutcome:
        return StepOutcome(self._obs("error", message=message), reward, False, {"timeout": False})

    def _tool_call(self, a: dict, box: FleetToolbox) -> StepOutcome:
        name, args = a.get("name"), a.get("arguments", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            return self._error("a tool call needs a name and an arguments object", BAD_ACTION_PENALTY)
        if self.executed_tool_calls >= self.max_tool_calls:
            return self._error(f"tool call limit of {self.max_tool_calls} reached", OVER_CAP_PENALTY)
        cost = TOOL_COSTS.get(name, DEFAULT_TOOL_COST)
        key = call_key(name, args)
        ok = bool(self.verifier.score_step({"type": "tool_call", "name": name, "arguments": args},
                                           (box, self._memory)).reward)
        self.executed_tool_calls += 1
        duplicate = key in self._memory.call_keys
        if duplicate:
            result = self._memory.results.get(key)
            if result is not None:
                return StepOutcome(self._obs("tool_result", name=name, result=result, note="duplicate call"),
                                   BAD_CALL_PENALTY - cost, False, {"timeout": False})
        res = box.registry.call(ToolCall(name=name, arguments=args))
        self._memory.call_keys.add(key)
        if not res.ok:
            return self._error(res.error or "tool call failed", BAD_CALL_PENALTY - cost)
        self._memory.results[key] = res.output
        if name == "check_feasibility":
            self._memory.feasibility_seen[(args["asset_id"], args["action"])] = res.output
        reward = (USEFUL_CALL_BONUS if ok else 0.0) - cost
        return StepOutcome(self._obs("tool_result", name=name, result=res.output), reward, False,
                           {"timeout": False, "useful": bool(ok)})

    def _claim(self, a: dict, kind: str, box: FleetToolbox) -> StepOutcome:
        key = json.dumps({k: v for k, v in a.items() if k not in ("can_gain_altitude", "thrust_slows_decay", "altitude_loss_km")},
                         sort_keys=True, default=str)
        verdict = self.verifier.score_step(a, (box, self._memory))
        if key in self._memory.claims:
            return StepOutcome(self._obs("claim_feedback", correct=bool(verdict.reward), note="repeated claim"),
                               REPEAT_CLAIM_PENALTY, False, {"timeout": False})
        if verdict.reason.startswith("malformed"):
            return self._error("malformed claim", BAD_ACTION_PENALTY)
        self._memory.claims.add(key)
        reward = CLAIM_REWARD if verdict.reward else -CLAIM_REWARD
        return StepOutcome(self._obs("claim_feedback", correct=bool(verdict.reward)), reward, False,
                           {"timeout": False, "claim_reason": verdict.reason})

    def _submit(self, a: dict) -> StepOutcome:
        self._done = True
        task = self.task
        plan = {k: v for k, v in a.items() if k != "type"}
        try:
            score = score_plan(task, plan)
        except ValueError as exc:
            return StepOutcome(self._obs("final", message=f"submission rejected: {exc}"), MALFORMED_SUBMIT, True,
                               {"timeout": False, "final": {"error": str(exc), "final_reward": MALFORMED_SUBMIT}})
        penalties = list(score["violations"])
        for asset_id, action in plan["assets"].items():
            seen = self._memory.feasibility_seen.get((asset_id, "raise_orbit"))
            if action == "raise_orbit" and seen is not None and not seen["thrust_slows_decay"]:
                penalties.append(f"ignored tool result: raised {asset_id} although check_feasibility "
                                 f"said thrust does not slow its decay")
        ignored = sum(p.startswith("ignored") for p in penalties)
        final_reward = score["final_reward"] - IGNORED_RESULT_PENALTY * ignored
        final = {**score, "penalties": penalties, "final_reward": final_reward, "error": None}
        return StepOutcome(self._obs("final", message="decision recorded"), final_reward, True,
                           {"timeout": False, "final": final})
