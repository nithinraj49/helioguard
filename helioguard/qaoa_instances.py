"""Random HelioGuard fleet-scheduling instances for the Phase 7 QAOA scaling benchmark.

Rewards come from the physics (`action_rewards`: Feb 2022 storm, Starlink-like craft). Spacecraft, storm
and altitude values are ILLUSTRATIVE. The fleet has 3 actions per asset, so its native qubit counts are
multiples of 3 (family A). Family B drops `hold` (2 actions per asset) to reach 8, 10, 14, 16 qubits.
Do not compare results across families: they have different numbers of constraint bits per asset.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

from verispace_engine import qubo as _engine
from verispace_engine.qubo import Qubo

from .scenarios import FEB_2022, STARLINK_LIKE
from .tools.qaoa_schedule import ACTION_NAMES, action_rewards

__all__ = ["FAMILIES", "ALT_RANGE_KM", "make_instance", "make_qubo"]

ALT_RANGE_KM = (190.0, 320.0)             # ILLUSTRATIVE altitude spread of the fleet
FAMILIES: dict[str, dict] = {
    "A": {"actions": ACTION_NAMES[:3], "sizes": [6, 9, 12, 15]},
    "B": {"actions": ACTION_NAMES[:2], "sizes": [6, 8, 10, 12, 14, 16]},
}
EXTENSION_SIZES = [18, 20]                  # family B only, harness arm, reduced grid (see docs/phase7_gates.md)


@lru_cache(maxsize=4096)
def _rewards_at(alt_km: float) -> tuple[tuple[str, float], ...]:
    r = action_rewards(STARLINK_LIKE, {**FEB_2022, "alt_km": alt_km})
    return tuple(r.items())


def make_instance(family: str, n_qubits: int, k: int) -> tuple[list[dict[str, float]], list[int], list[str]]:
    """Instance k of `family` with `n_qubits` qubits: (rewards, windows, actions)."""
    actions = list(FAMILIES[family]["actions"])
    if n_qubits % len(actions):
        raise ValueError(f"{n_qubits} qubits is not a multiple of {len(actions)} actions (family {family})")
    items = n_qubits // len(actions)
    rng = random.Random(7_000_000 + 10_000 * ord(family) + 100 * n_qubits + k)
    n_windows = max(2, math.ceil(items / 2))
    rewards, windows = [], []
    for _ in range(items):
        alt = round(rng.uniform(*ALT_RANGE_KM), 1)
        full = dict(_rewards_at(alt))
        rewards.append({a: full[a] for a in actions})
        windows.append(rng.randrange(n_windows))
    return rewards, windows, actions


def make_qubo(family: str, n_qubits: int, k: int) -> Qubo:
    rewards, windows, actions = make_instance(family, n_qubits, k)
    return _engine.build_qubo(rewards, windows, actions, actions[0])
