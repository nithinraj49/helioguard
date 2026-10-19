"""Density from NRLMSIS (pymsis) driven by real F10.7 / Ap, with the reference model as fallback.

The result is a `density_fn(alt_km, hours_since_start)`. Density is a crude orbit average
over a small lat/lon grid (ILLUSTRATIVE: not a propagated ground track).
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from .atmosphere import density as reference_density
from .physics import DensityFn

_ALTS = np.arange(120.0, 605.0, 5.0)
_LATS = [-50.0, -25.0, 0.0, 25.0, 50.0]
_LONS = [0.0, 90.0, 180.0, 270.0]


class MsisDensity:
    def __init__(self, hours: int, log_rho: np.ndarray):
        self.hours = float(hours)
        self._log_rho = log_rho            # shape (hours + 1, n_alts)

    @classmethod
    def from_file(cls, sw_path: Path, start: datetime, hours: float,
                  lats: list[float] | None = None,
                  lons: list[float] | None = None) -> "MsisDensity":
        import pymsis                       # imported here so the fallback works without it
        pymsis.utils.use_space_weather_file(Path(sw_path))
        n = int(np.ceil(hours))
        dates = np.datetime64(start, "m") + (np.arange(n + 1) * 60).astype("timedelta64[m]")
        out = pymsis.calculate(dates, lons or _LONS, lats or _LATS, _ALTS)
        rho = out[..., 0].mean(axis=(1, 2))   # average over lon, lat -> (time, alt)
        return cls(n, np.log(rho))

    def __call__(self, alt_km: float, t_h: float) -> float:
        t = min(max(t_h, 0.0), self.hours)
        i = min(int(t), len(self._log_rho) - 2)
        w = t - i
        row = (1 - w) * self._log_rho[i] + w * self._log_rho[i + 1]
        return float(np.exp(np.interp(alt_km, _ALTS, row)))


def get_density_fn(sw_path: Path, start: datetime, hours: float,
                   storm_multiplier: float = 1.0) -> tuple[DensityFn, str]:
    """Return (density_fn, source). Falls back to the reference atmosphere if pymsis or the
    space-weather file is unavailable; `source` says which one you got."""
    try:
        return MsisDensity.from_file(sw_path, start, hours), "pymsis"
    except (ImportError, OSError, ValueError) as exc:
        warnings.warn(f"pymsis density unavailable ({exc}); using reference atmosphere")
        return (lambda alt_km, t_h: reference_density(alt_km, storm_multiplier)), "reference"
