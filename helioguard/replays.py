"""Replays of real storms driven by real space-weather indices.

Spacecraft parameters (scenarios.STARLINK_LIKE) and the replay altitudes are ILLUSTRATIVE.
Density comes from NRLMSIS with the real F10.7 / Ap of the event days.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from .atmosphere_msis import get_density_fn
from .data.noaa import fetch_space_weather, load_space_weather, storm_summary

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

FEB_2022_REAL = {
    "id": "starlink_feb_2022_real",
    "start": datetime(2022, 2, 3, 18, 0),      # deployment on 3 Feb (time illustrative)
    "window": (date(2022, 2, 3), date(2022, 2, 5)),
    "alt_km": 210.0,                           # deployment altitude of the event
    "horizon_h": 48,
}
MAY_2024_REAL = {
    "id": "storm_may_2024_real",
    "start": datetime(2024, 5, 10, 12, 0),
    "window": (date(2024, 5, 10), date(2024, 5, 12)),
    "alt_km": 350.0,                           # ILLUSTRATIVE: typical low parking orbit
    "horizon_h": 48,
}


def build_replay(spec: dict, sw_path: Path | None = None) -> dict:
    """Scenario dict usable by the verifier. `sw_path=None` downloads (and caches) SW-All."""
    path = Path(sw_path) if sw_path else fetch_space_weather(CACHE_DIR)
    summary = storm_summary(load_space_weather(path), *spec["window"])
    fn, source = get_density_fn(path, spec["start"], spec["horizon_h"])
    return {"id": spec["id"], "start": spec["start"], "alt_km": spec["alt_km"],
            "horizon_h": spec["horizon_h"], "kp": summary["kp_max"], "g_scale": summary["g_scale"],
            "ap_max": summary["ap_max"], "storm_multiplier": 1.0,   # density already in density_fn
            "density_fn": fn, "data_source": source, "illustrative_spacecraft": True}
