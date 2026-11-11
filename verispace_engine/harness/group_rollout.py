"""Group rollout: the 8 attempts of one task advance in lock-step so a model can answer them as one batch.

Each attempt follows exactly the logic of `rollout_task` (same parser, same environment, same rewards, same
failure labels); only the model calls are batched. A test replays scripted attempts through both and compares the
results field by field. Domain-neutral; torch is imported only by `HFBatchGenerator`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .parser import ActionParseError, parse_action
from .prompts import briefing_message, observation_message, system_prompt
from .rollout import EpisodeResult, RolloutConfig

# generate_batch(messages_list, max_new_tokens, temperature, seed, indices) -> one reply per messages list.
# `indices` are the attempt numbers of the lists (scripted test generators use them; model generators ignore them).
BatchGenerate = Callable[[Sequence[Sequence[dict]], int, float, int, Sequence[int]], list[str]]


@dataclass
class GroupEpisode:
    result: EpisodeResult
    messages: list[dict[str, str]]          # the whole chat, ending with the model's last output


def rollout_group(envs: Sequence[Any], task_id: str, generate_batch: BatchGenerate, cfg: RolloutConfig,
                  seed: int = 0) -> list[GroupEpisode]:
    """Run len(envs) attempts of `task_id` (one environment per attempt) and return one GroupEpisode each."""
    started = time.time()
    n = len(envs)
    messages, tool_names, done = [], [], [False] * n
    for env in envs:
        briefing = env.reset(task_id=task_id)
        messages.append([{"role": "system", "content": system_prompt(env.tool_specs(), cfg.extra_actions)},
                         {"role": "user", "content": briefing_message(briefing)}])
        tool_names.append(set(env.tool_names()))
    total, turns, steps = [0.0] * n, [0] * n, [[] for _ in range(n)]
    failure: list[str | None] = [None] * n
    final: list[dict | None] = [None] * n
    transcripts: list[list[dict]] = [[] for _ in range(n)]
    max_turns = getattr(envs[0], "max_turns", 20) + 1
    for turn in range(1, max_turns + 1):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        raws = generate_batch([messages[i] for i in active], cfg.max_new_tokens, cfg.temperature, seed * 1000 + turn, active)
        for i, raw in zip(active, raws):
            turns[i] = turn
            transcripts[i] = [*messages[i], {"role": "assistant", "content": raw}]
            try:
                action = parse_action(raw, cfg.allowed_types, tool_names[i])
            except ActionParseError as exc:
                total[i] += cfg.malformed_reward
                steps[i].append(cfg.malformed_reward)
                failure[i] = f"malformed:{exc.kind}"
                done[i] = True
                continue
            out = envs[i].step(action)
            total[i] += out.reward
            steps[i].append(out.reward)
            if out.done:
                final[i] = out.info.get("final")
                if out.info.get("timeout"):
                    failure[i] = "never_submits"
                done[i] = True
                continue
            messages[i] += [{"role": "assistant", "content": raw},
                            {"role": "user", "content": observation_message(out.observation, compactor=cfg.compactor)}]
    elapsed = (time.time() - started) / max(1, n)
    episodes = []
    for i in range(n):
        valid = bool(final[i]) and final[i].get("error") is None
        if final[i] and final[i].get("error"):
            failure[i] = "rejected_submission"
        if failure[i] is None and not valid:
            failure[i] = "never_submits"
        result = EpisodeResult(task_id, i, total[i], final[i]["final_reward"] if valid else cfg.failure_final_reward,
                               valid, getattr(envs[i], "executed_tool_calls", 0), turns[i], failure[i], steps[i], elapsed)
        episodes.append(GroupEpisode(result, transcripts[i]))
    return episodes


class HFBatchGenerator:
    """Batched sampling from an in-memory causal LM (left-padded). One torch seed per call."""

    def __init__(self, model: Any, tokenizer: Any, device: str, top_p: float = 0.95) -> None:
        import torch
        self._torch, self.model, self.tok, self.device, self.top_p = torch, model, tokenizer, device, top_p
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, messages_list: Sequence[Sequence[dict]], max_new_tokens: int, temperature: float, seed: int,
                 indices: Sequence[int] = ()) -> list[str]:
        torch = self._torch
        texts = [self.tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True) for m in messages_list]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        torch.manual_seed(seed)
        kwargs = ({"do_sample": True, "temperature": temperature, "top_p": self.top_p} if temperature > 0
                  else {"do_sample": False})
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                          use_cache=True, **kwargs)
        finally:
            self.model.train(was_training)
        width = enc["input_ids"].shape[1]
        return [self.tok.decode(row[width:], skip_special_tokens=True) for row in out]
