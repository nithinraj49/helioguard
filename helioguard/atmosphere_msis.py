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
                  lats: list[float] | None = None, lons: list[float] | None = None,
                  storm_time: bool = False) -> "MsisDensity":
        """`storm_time=True` drives MSIS with the 3-hourly ap history (geomagnetic_activity=-1)
        instead of the default daily Ap, which smears a short storm over the whole day."""
        import pymsis                       # imported here so the fallback works without it
        pymsis.utils.use_space_weather_file(Path(sw_path))
        n = int(np.ceil(hours))
        dates = np.datetime64(start, "m") + (np.arange(n + 1) * 60).astype("timedelta64[m]")
        out = pymsis.calculate(dates, lons or _LONS, lats or _LATS, _ALTS,
                               geomagnetic_activity=-1 if storm_time else 1)
        rho = out[..., 0].mean(axis=(1, 2))   # average over lon, lat -> (time, alt)
        return cls(n, np.log(rho))

    def __call__(self, alt_km: float, t_h: float) -> float:
        t = min(max(t_h, 0.0), self.hours)
        i = min(int(t), len(self._log_rho) - 2)
        w = t - i
        row = (1 - w) * self._log_rho[i] + w * self._log_rho[i + 1]
        return float(np.exp(np.interp(alt_km, _ALTS, row)))


def get_density_fn(sw_path: Path, start: datetime, hours: float,
                   storm_multiplier: float = 1.0, storm_time: bool = False) -> tuple[DensityFn, str]:
    """Return (density_fn, source). Falls back to the reference atmosphere if pymsis or the
    space-weather file is unavailable; `source` says which one you got."""
    try:
        return MsisDensity.from_file(sw_path, start, hours, storm_time=storm_time), "pymsis"
    except (ImportError, OSError, ValueError) as exc:
        warnings.warn(f"pymsis density unavailable ({exc}); using reference atmosphere")
        return (lambda alt_km, t_h: reference_density(alt_km, storm_multiplier)), "reference"


def storm_density_change(sw_path: Path, alt_km: float, storm_time: bool,
                         baseline_day: datetime = datetime(2022, 2, 2),
                         hours: int = 72) -> dict:
    """Density change at `alt_km` over the `hours` after the day following the baseline day,
    relative to the baseline day's 24-hour mean (same grid point for the local peak).

    peak_local_ratio: largest hourly density at any single lat/lon point / its baseline mean.
    peak_orbit_avg_ratio: largest hourly lat/lon-averaged density / baseline orbit average.
    """
    import pymsis
    pymsis.utils.use_space_weather_file(Path(sw_path))
    kw = {"geomagnetic_activity": -1 if storm_time else 1}
    day0 = np.datetime64(baseline_day, "m")

    def grid(start, n):
        dates = start + (np.arange(n) * 60).astype("timedelta64[m]")
        return pymsis.calculate(dates, _LONS, _LATS, [alt_km], **kw)[..., 0, 0]  # (t, lon, lat)

    base = grid(day0, 24).mean(axis=0)                       # (lon, lat)
    storm = grid(day0 + np.timedelta64(24, "h"), hours)
    local = (storm / base).max(axis=(1, 2))
    avg = storm.mean(axis=(1, 2)) / base.mean()
    return {"baseline_day": str(baseline_day.date()), "storm_time_mode": storm_time,
            "alt_km": alt_km, "baseline_density": float(base.mean()),
            "peak_local_ratio": float(local.max()), "peak_orbit_avg_ratio": float(avg.max()),
            "peak_hour": int(avg.argmax()), "mean_orbit_avg_ratio": float(avg.mean())}
