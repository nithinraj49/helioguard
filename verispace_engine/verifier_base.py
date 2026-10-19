"""Verifier base: rules keyed by step type, each returning (ok, reason).

The verifier is independent code; the agent never grades itself.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable

Rule = Callable[[dict[str, Any], Any], tuple[bool, str]]


@dataclass
class StepResult:
    reward: int
    reason: str


@dataclass
class EpisodeResult:
    step_rewards: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    process_reward: float = 0.0


class VerifierBase(ABC):
    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def add_rule(self, step_type: str, rule: Rule) -> None:
        if step_type in self._rules:
            raise ValueError(f"rule for '{step_type}' already registered")
        self._rules[step_type] = rule

    def score_step(self, step: dict[str, Any], context: Any) -> StepResult:
        kind = step.get("type")
        rule = self._rules.get(kind)
        if rule is None:
            return StepResult(0, f"unknown step type '{kind}'")
        try:
            ok, reason = rule(step, context)
        except (KeyError, TypeError, ValueError) as exc:   # malformed claim never earns reward
            return StepResult(0, f"malformed '{kind}' step: {type(exc).__name__} {exc}")
        return StepResult(int(ok), reason)

    def score_episode(self, trace: list[dict[str, Any]], context: Any) -> EpisodeResult:
        results = [self.score_step(s, context) for s in trace]
        rewards = [r.reward for r in results]
        return EpisodeResult(rewards, [r.reason for r in results],
                             sum(rewards) / len(rewards) if rewards else 0.0)


def group_centered(rewards: list[float]) -> list[float]:
    """Group-relative rewards (r - mean) for a group of traces on the same prompt."""
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]
