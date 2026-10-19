"""Pydantic schemas for decision traces and tool calls."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceStep(BaseModel):
    """One reasoning step. Only `type` is fixed; other fields belong to the domain."""
    model_config = ConfigDict(extra="allow")
    type: str


class DecisionTrace(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)

    @classmethod
    def from_steps(cls, steps: list[dict[str, Any]]) -> DecisionTrace:
        return cls(steps=[TraceStep.model_validate(s) for s in steps])

    def as_dicts(self) -> list[dict[str, Any]]:
        return [s.model_dump() for s in self.steps]


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None
