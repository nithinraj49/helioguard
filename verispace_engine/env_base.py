"""Environment base: reset() returns an observation, step(action) returns a StepOutcome."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepOutcome:
    observation: Any
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class EnvBase(ABC):
    @abstractmethod
    def reset(self) -> Any: ...

    @abstractmethod
    def step(self, action: Any) -> StepOutcome: ...
