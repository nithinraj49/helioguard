"""Space-weather indices: Kp, Ap and F10.7 from the CelesTrak SW-All file (NOAA/GFZ data)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..atmosphere import kp_to_g_scale
from .cache import Downloader, fetch_cached, http_download

SW_ALL_URL = "https://celestrak.org/SpaceData/SW-All.csv"


@dataclass(frozen=True)
class DailyIndices:
    day: date
    kp_3h: tuple[float, ...]      # eight 3-hourly Kp values (file stores Kp x 10)
    ap_3h: tuple[int, ...]        # eight 3-hourly ap values
    ap_daily: int
    f107_obs: float


def fetch_space_weather(cache_dir: Path, *, max_age_days: float = 1.0,
                        downloader: Downloader = http_download) -> Path:
    return fetch_cached(SW_ALL_URL, Path(cache_dir) / "SW-All.csv",
                        max_age_days=max_age_days, downloader=downloader)


def load_space_weather(path: Path) -> list[DailyIndices]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            try:
                rows.append(DailyIndices(
                    day=date.fromisoformat(rec["DATE"]),
                    kp_3h=tuple(int(rec[f"KP{i}"]) / 10 for i in range(1, 9)),
                    ap_3h=tuple(int(rec[f"AP{i}"]) for i in range(1, 9)),
                    ap_daily=int(rec["AP_AVG"]),
                    f107_obs=float(rec["F10.7_OBS"]),
                ))
            except (KeyError, ValueError):
                continue          # incomplete (future) rows carry no usable indices
    return rows


def storm_summary(rows: list[DailyIndices], start: date, end: date) -> dict:
    """Strongest Kp / ap in [start, end] inclusive, with the NOAA G-scale for that Kp."""
    window = [r for r in rows if start <= r.day <= end]
    if not window:
        raise ValueError(f"no space-weather data between {start} and {end}")
    kp_max = max(max(r.kp_3h) for r in window)
    return {"kp_max": kp_max, "g_scale": kp_to_g_scale(kp_max),
            "ap_max": max(max(r.ap_3h) for r in window),
            "f107_mean": sum(r.f107_obs for r in window) / len(window)}
