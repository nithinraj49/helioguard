"""Real geomagnetic storms (from the Kp record) and an illustrative density profile.

Storm dates, onsets and 3-hourly Kp series are real (SW-All, NOAA/GFZ). The mapping from
Kp to a density multiplier is ILLUSTRATIVE: NRLMSIS alone understates storm-time density
(see the README, Phase 2), so peak multipliers follow 1 + 0.35 * (Kp - 4), which gives about
x1.5 for the Feb 2022 storm (the Phase 1 value) and x2.75 for Kp 9.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

from .atmosphere import kp_to_g_scale
from .data.noaa import DailyIndices

SERIES_BEFORE_H = 48
SERIES_SLOTS = 48                      # 3-hourly: 48 h before onset to 96 h after
STORM_KP = 5.0                         # NOAA G1 and above
DEFAULT_FIRST_DAY = date(1996, 1, 1)
DEFAULT_LAST_DAY = date(2026, 9, 30)   # later rows of SW-All are predictions, not observations


def peak_multiplier(kp_max: float) -> float:
    """ILLUSTRATIVE peak density multiplier for a storm of this Kp."""
    return max(1.0, 1.0 + 0.35 * (kp_max - 4.0))


@dataclass(frozen=True)
class Storm:
    id: str
    onset: datetime                      # first 3-hour slot with Kp >= 5 (UTC)
    kp_max: float
    g_scale: str
    kp_series: tuple[float, ...]         # 48 slots from onset - 48 h

    @property
    def series_start(self) -> datetime:
        return self.onset - timedelta(hours=SERIES_BEFORE_H)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "onset": self.onset.isoformat(), "kp_max": self.kp_max,
                "g_scale": self.g_scale, "kp_series": list(self.kp_series), "real": True}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Storm":
        return cls(d["id"], datetime.fromisoformat(d["onset"]), d["kp_max"], d["g_scale"],
                   tuple(d["kp_series"]))


def storm_catalog(rows: Sequence[DailyIndices], min_kp: float = STORM_KP,
                  first_day: date = DEFAULT_FIRST_DAY, last_day: date = DEFAULT_LAST_DAY) -> list[Storm]:
    """One entry per storm cluster (days with Kp >= min_kp, gaps of at most one day merged).
    Storms without a complete 144 h Kp series around them are skipped."""
    by_day = {r.day: r for r in rows if first_day <= r.day <= last_day}
    strong = sorted(d for d, r in by_day.items() if max(r.kp_3h) >= min_kp)
    clusters: list[list[date]] = []
    for d in strong:
        if clusters and (d - clusters[-1][-1]).days <= 1:
            clusters[-1].append(d)
        else:
            clusters.append([d])
    out = []
    for c in clusters:
        slot = next(i for i, k in enumerate(by_day[c[0]].kp_3h) if k >= min_kp)
        onset = datetime(c[0].year, c[0].month, c[0].day, 3 * slot, tzinfo=timezone.utc)
        series = []
        for k in range(SERIES_SLOTS):
            when = onset - timedelta(hours=SERIES_BEFORE_H) + timedelta(hours=3 * k)
            row = by_day.get(when.date())
            if row is None:
                series = []
                break
            series.append(row.kp_3h[when.hour // 3])
        if not series:
            continue
        kp_max = max(series)
        out.append(Storm(f"storm_{onset:%Y%m%d}", onset, kp_max, kp_to_g_scale(kp_max), tuple(series)))
    return out


class StormProfile:
    """Density multiplier over time since the forecast was issued.

    The storm onset arrives `arrival_h` hours after t = 0. `stretch` > 1 lengthens the storm
    (time after onset is divided by it). Kp below 5 gives no enhancement; the strongest slot
    gives the full `peak`.
    """

    def __init__(self, storm: Storm, arrival_h: float, peak: float, stretch: float = 1.0) -> None:
        self.storm, self.arrival_h, self.peak, self.stretch = storm, arrival_h, peak, stretch

    def __call__(self, t_h: float) -> float:
        u = t_h - self.arrival_h
        if u > 0:
            u /= self.stretch
        idx = int(math.floor((u + SERIES_BEFORE_H) / 3.0))
        if not 0 <= idx < SERIES_SLOTS:
            return 1.0
        kp = self.storm.kp_series[idx]
        if kp < STORM_KP or self.storm.kp_max <= 4.0:
            return 1.0
        f = min(1.0, (kp - 4.0) / (self.storm.kp_max - 4.0))
        return 1.0 + (self.peak - 1.0) * f
