"""Tool protocol and registry. A tool is anything with a name, JSON schema and run()."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .schema import ToolCall, ToolResult

_JSON_TYPES = {"number": (int, float), "integer": (int,), "string": (str,),
               "boolean": (bool,), "array": (list,), "object": (dict,)}


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    schema: dict[str, Any]          # JSON schema of the arguments

    def run(self, **arguments: Any) -> dict[str, Any]: ...


def _check_arguments(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    for key, val in args.items():
        if key not in props:
            return f"unexpected argument '{key}'"
        want = props[key].get("type")
        if want in _JSON_TYPES:
            is_bool = isinstance(val, bool)
            if not isinstance(val, _JSON_TYPES[want]) or (is_bool and want != "boolean"):
                return f"argument '{key}' must be {want}"
    return None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("object does not satisfy the Tool protocol")
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.schema}
                for t in self._tools.values()]

    def call(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(name=call.name, ok=False, error=f"unknown tool '{call.name}'")
        problem = _check_arguments(tool.schema, call.arguments)
        if problem:
            return ToolResult(name=call.name, ok=False, error=problem)
        try:
            return ToolResult(name=call.name, ok=True, output=tool.run(**call.arguments))
        except Exception as exc:  # a failing tool must not crash the episode
            return ToolResult(name=call.name, ok=False, error=f"{type(exc).__name__}: {exc}")
