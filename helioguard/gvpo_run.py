"""HelioGuard's plug-in for GVPO training: group sampling on the real environment, and the run entry point.

Sampling uses the harness pieces unchanged (strict parser, environment, rewards, compact observation view); only the
model calls of the 8 attempts of a task are batched (`verispace_engine.harness.group_rollout`). Train tasks only.
"""
from __future__ import annotations

from typing import Any

from verispace_engine.gvpo import Episode, GVPOConfig, train_gvpo
from verispace_engine.harness.group_rollout import HFBatchGenerator, rollout_group
from verispace_engine.harness.rollout import RolloutConfig

from .env import HelioGuardEnv
from .harness_env import ALLOWED_TYPES, EXTRA_ACTIONS, compact_observation
from .sft_traces import train_tasks


def rollout_config(cfg: GVPOConfig) -> RolloutConfig:
    return RolloutConfig(attempts=1, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, seed=cfg.seed,
                         allowed_types=tuple(ALLOWED_TYPES), extra_actions=tuple(EXTRA_ACTIONS),
                         compactor=compact_observation)


def make_sampler(cfg: GVPOConfig):
    """sample_fn(model, tokenizer, device, task_id, group_size, seed, cfg) -> list[Episode], on the train tasks."""
    tasks = train_tasks()                                    # refuses eval tasks and eval storms
    envs: list[HelioGuardEnv] = []

    def sample(model: Any, tok: Any, device: str, task_id: str, group_size: int, seed: int, _cfg: GVPOConfig):
        while len(envs) < group_size:
            envs.append(HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20))
        gen = HFBatchGenerator(model, tok, device, top_p=cfg.top_p)
        group = rollout_group(envs[:group_size], task_id, gen, rollout_config(cfg), seed=seed)
        return [Episode(messages=g.messages, final_reward=g.result.final_reward, total_return=g.result.total_return,
                        valid=g.result.valid_submission, info={"failure": g.result.failure, "tool_calls": g.result.tool_calls})
                for g in group]
    return sample


def run_training(cfg: GVPOConfig, hub=None, stop_after_steps: int | None = None) -> dict:
    ids = [t["id"] for t in train_tasks()]
    return train_gvpo(cfg, ids, make_sampler(cfg), hub=hub, stop_after_steps=stop_after_steps)
