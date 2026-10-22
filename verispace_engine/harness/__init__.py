"""Model harness: prompts, strict parser, rollouts, backends and logging for any EnvBase task.

Domain-neutral: nothing here knows about satellites. A domain supplies an environment whose
reset() returns the briefing text, whose step() takes dict actions and returns StepOutcome, and
that exposes tool_specs(). Final rewards are read from outcome.info["final"]["final_reward"].
"""
from .backends import HFBackend, ModelBackend, OpenAICompatibleBackend, ScriptedBackend
from .parser import ActionParseError, parse_action
from .prompts import system_prompt
from .rollout import EpisodeResult, RolloutConfig, rollout_task, run_rollouts

__all__ = ["ActionParseError", "EpisodeResult", "HFBackend", "ModelBackend", "OpenAICompatibleBackend",
           "RolloutConfig", "ScriptedBackend", "parse_action", "rollout_task", "run_rollouts", "system_prompt"]
