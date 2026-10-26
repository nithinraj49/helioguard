"""Small statistics helpers for evaluation reports: percentile bootstrap over tasks."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, low, high): percentile bootstrap CI of the mean, resampling the entries of `values`
    (one entry per task, so tasks are the unit of resampling)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
