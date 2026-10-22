"""Strict parser: the model's whole reply must be one JSON action. Anything else is a failure.

Accepted: a single JSON object, optionally inside one ``` fence, with only whitespace around it.
Not accepted: prose before or after, several objects, trailing text, unknown action types, tool
calls without a name or arguments object, a submit_decision without an assets object. There is no
repair step: a malformed reply ends the episode (the rollout runner scores it as a failure).
"""
from __future__ import annotations

import json
import re
from typing import Any, Collection

FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*?)\n?```\Z", re.DOTALL)


class ActionParseError(ValueError):
    """kind is a short failure label used in reports: empty, no_json, extra_text, not_object,
    missing_type, unknown_type, bad_tool_call, bad_submit."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def parse_action(text: str, allowed_types: Collection[str], tool_names: Collection[str] | None = None) -> dict[str, Any]:
    body = (text or "").strip()
    if not body:
        raise ActionParseError("empty", "the reply was empty")
    fence = FENCE.match(body)
    if fence:
        body = fence.group(1).strip()
    try:
        obj, end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as exc:
        raise ActionParseError("no_json", f"the reply is not valid JSON ({exc.msg})") from None
    if body[end:].strip():
        raise ActionParseError("extra_text", "text found after the JSON object; reply with the JSON object alone")
    if not isinstance(obj, dict):
        raise ActionParseError("not_object", "the reply must be a JSON object")
    kind = obj.get("type")
    if not isinstance(kind, str):
        raise ActionParseError("missing_type", "the object needs a string field 'type'")
    if kind not in allowed_types:
        raise ActionParseError("unknown_type", f"unknown action type {kind!r}")
    if kind == "tool_call":
        name, args = obj.get("name"), obj.get("arguments", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            raise ActionParseError("bad_tool_call", "a tool call needs a string 'name' and an 'arguments' object")
        if tool_names is not None and name not in tool_names:
            raise ActionParseError("bad_tool_call", f"unknown tool {name!r}")
    if kind == "submit_decision" and not isinstance(obj.get("assets"), dict):
        raise ActionParseError("bad_submit", "submit_decision needs an 'assets' object")
    return obj
