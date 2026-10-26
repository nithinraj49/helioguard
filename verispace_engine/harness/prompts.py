"""Prompt templates: the system prompt (protocol + tool schemas) and observation messages."""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

PROTOCOL = """You are an operations analyst agent. Work in turns. Each turn, reply with exactly ONE JSON object and nothing else (no prose, no markdown).

Action formats:
  call a tool:   {"type": "tool_call", "name": "<tool name>", "arguments": {...}}
@@EXTRA@@  finish:        {"type": "submit_decision", ...}   (this ends the episode; see the briefing for its fields)

Rules: tool calls are limited and each has a small cost; repeating a call is penalised; a reply that is not a single valid JSON object ends the episode as a failure. You recommend; a human decides.

Tools (name, description, argument schema):
@@TOOLS@@"""


def system_prompt(tool_specs: Sequence[dict[str, Any]], extra_actions: Sequence[str] = ()) -> str:
    tools = "\n".join(f"- {s['name']}: {s['description']}\n  arguments schema: {json.dumps(s.get('parameters', {}), separators=(',', ':'))}"
                      for s in tool_specs)
    extra = "".join(f"  {line}\n" for line in extra_actions)
    return PROTOCOL.replace("@@EXTRA@@", extra).replace("@@TOOLS@@", tools)


def briefing_message(briefing: str) -> str:
    return briefing + "\n\nReply with one JSON action."


def observation_message(observation: Any, max_chars: int = 3000, compactor: Callable[[Any], Any] | None = None) -> str:
    """The environment's answer to the last action, shown to the model as the next user turn.
    `compactor` (a domain hook) may shorten the observation first, e.g. drop long time series."""
    if compactor is not None:
        observation = compactor(observation)
    text = json.dumps(observation, separators=(",", ":"), default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "...(truncated)"
    return f"Environment result: {text}\nReply with one JSON action."
