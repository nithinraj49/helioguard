"""Step verification for the multi-turn environment, on the engine VerifierBase.

Rules check a step against physics (never an LLM): was a tool call valid, new and relevant, and is
a claim about feasibility or decay true for the forecast storm. Context is (toolbox, memory).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from verispace_engine.verifier_base import VerifierBase

from .env_tools import FleetToolbox

DECAY_REL_TOL = 0.10                  # a decay prediction must be within 10 percent ...
DECAY_ABS_TOL_KM = 0.5                # ... or half a kilometre, whichever is larger


@dataclass
class EpisodeMemory:
    """What the agent has done and seen in this episode."""
    call_keys: set[str] = field(default_factory=set)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)       # call key -> tool output
    claims: set[str] = field(default_factory=set)
    feasibility_seen: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)


def call_key(name: str, arguments: dict) -> str:
    return name + json.dumps(arguments, sort_keys=True, default=str)


class EnvVerifier(VerifierBase):
    def __init__(self) -> None:
        super().__init__()
        self.add_rule("tool_call", self._tool_call)
        self.add_rule("claim_can_gain_altitude", self._claim_can_gain_altitude)
        self.add_rule("claim_thrust_slows_decay", self._claim_thrust_slows_decay)
        self.add_rule("claim_decay", self._claim_decay)

    @staticmethod
    def _tool_call(step: dict, ctx: tuple[FleetToolbox, EpisodeMemory]) -> tuple[bool, str]:
        """OK means: a new call to a tool that helps this task."""
        box, memory = ctx
        name, args = step["name"], step["arguments"]
        if name not in box.registry.names():
            return False, f"unknown tool '{name}'"
        if call_key(name, args) in memory.call_keys:
            return False, "duplicate call"
        if name in ("simulate_decay", "check_feasibility"):
            return args.get("asset_id") in box.assets, "asset must be in the fleet"
        if name == "qaoa_schedule":
            return bool(box.task["needs_qaoa"]), "only useful when assets compete for a ground window"
        if name == "min_safe_altitude":
            return True, "a threshold the decision can use"
        return False, "the density correction is disabled, so it adds nothing"

    @staticmethod
    def _claim_can_gain_altitude(step: dict, ctx: tuple[FleetToolbox, EpisodeMemory]) -> tuple[bool, str]:
        """Claim that thrust exceeds drag for a raise, now or at the forecast storm peak."""
        box, _ = ctx
        truth = box.feasibility(step["asset_id"], "raise_orbit")
        when = step["when"]
        if when not in ("now", "through_storm"):
            raise ValueError("when must be 'now' or 'through_storm'")
        key = "can_gain_altitude" if when == "now" else "can_gain_altitude_through_storm"
        if not isinstance(step["can_gain_altitude"], bool):
            raise ValueError("can_gain_altitude must be true or false")
        return step["can_gain_altitude"] == truth[key], f"{when}: physics says {truth[key]}"

    @staticmethod
    def _claim_thrust_slows_decay(step: dict, ctx: tuple[FleetToolbox, EpisodeMemory]) -> tuple[bool, str]:
        """Claim that raising outlives safe mode and hold in the forecast storm."""
        box, _ = ctx
        if not isinstance(step["thrust_slows_decay"], bool):
            raise ValueError("thrust_slows_decay must be true or false")
        truth = box.thrust_slows_decay(step["asset_id"])
        return step["thrust_slows_decay"] == truth, f"physics says {truth}"

    @staticmethod
    def _claim_decay(step: dict, ctx: tuple[FleetToolbox, EpisodeMemory]) -> tuple[bool, str]:
        box, _ = ctx
        true_loss = box.decay(step["asset_id"], step["action"], step["hours"])["altitude_loss_km"]
        pred = float(step["altitude_loss_km"])
        tol = max(DECAY_REL_TOL * abs(true_loss), DECAY_ABS_TOL_KM)
        return abs(pred - true_loss) <= tol, f"predicted {pred:.1f} km vs physics {true_loss:.1f} km"
