"""Propose training storms and passive tracking objects for the QNN density check.

Storms come from the Kp record; objects are filtered from Space-Track histories. Passive
objects (rocket bodies) are preferred because thrusting satellites hide their own drag.
The output is a draft for a human to review before any training.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

from ..data.celestrak import TLE
from ..data.noaa import DailyIndices
from .decay_density import _decay_rate, _pairs

MIN_INTERVAL_H = 3.0          # shorter intervals are dominated by TLE fit noise
MAX_RAISE_MS = 0.002          # da/dt above this (m/s) looks like a manoeuvre, not drag


@dataclass(frozen=True)
class StormCandidate:
    start: date
    end: date
    onset: datetime
    kp_max: float
    ap_max: int

    @property
    def name(self) -> str:
        return f"storm_{self.onset:%Y_%m_%d}"


def find_storms(rows: Sequence[DailyIndices], min_kp: float = 7.0,
                exclude: Sequence[tuple[date, date]] = (), exclusion_margin_days: int = 10,
                first_day: date = date(2015, 1, 1), max_events: int = 10) -> list[StormCandidate]:
    """Clusters of days with Kp >= min_kp, ranked by strength. Windows near `exclude` are skipped."""
    days = [r for r in sorted(rows, key=lambda r: r.day) if r.day >= first_day]
    strong = [r for r in days if max(r.kp_3h) >= min_kp]
    clusters: list[list[DailyIndices]] = []
    for r in strong:
        if clusters and (r.day - clusters[-1][-1].day).days <= 2:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    by_day = {r.day: r for r in days}
    out = []
    for c in clusters:
        onset = None
        for d in (c[0].day - timedelta(days=1), c[0].day):
            row = by_day.get(d)
            if row is None:
                continue
            slot = next((i for i, k in enumerate(row.kp_3h) if k >= 5.0), None)
            if slot is not None:
                onset = datetime(d.year, d.month, d.day, 3 * slot, tzinfo=timezone.utc)
                break
        if onset is None:
            continue
        start, end = onset.date() - timedelta(days=3), onset.date() + timedelta(days=4)
        margin = timedelta(days=exclusion_margin_days)
        if any(start <= ex_end + margin and end >= ex_start - margin for ex_start, ex_end in exclude):
            continue
        out.append(StormCandidate(start, end, onset, max(max(r.kp_3h) for r in c),
                                  max(max(r.ap_3h) for r in c)))
    out.sort(key=lambda s: (s.kp_max, s.ap_max), reverse=True)
    return out[:max_events]


def usable_objects(histories: dict[int, Sequence[TLE]], onset: datetime, min_pre: int = 3,
                   min_post: int = 4, max_objects: int = 12,
                   min_interval_h: float = MIN_INTERVAL_H) -> list[int]:
    """Objects with enough element sets before and after onset and no manoeuvre-like rise."""
    scored = []
    for norad, tles in histories.items():
        pre = sum(t.epoch < onset for t in tles)
        post = sum(t.epoch >= onset for t in tles)
        if pre < min_pre or post < min_post:
            continue
        try:
            rates = [r for r, _, dt in (_decay_rate(a, b) for a, b in _pairs(tles))
                     if dt >= min_interval_h * 3600]
        except ValueError:
            continue
        if any(r > MAX_RAISE_MS for r in rates):
            continue
        scored.append((len(tles), norad))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored[:max_objects]]


def _event(name: str, storm: StormCandidate, ids: list[int], role: str, ballistic: float | None = None,
           **extra: Any) -> dict[str, Any]:
    return {"name": name, "role": role, "start": storm.start.isoformat(), "end": storm.end.isoformat(),
            "onset": storm.onset.isoformat(), "kp_max": storm.kp_max, "ap_max": storm.ap_max,
            "norad_ids": ids, "ballistic": ballistic, **extra}


def build_config(training: list[tuple[StormCandidate, list[int]]], may_2024: tuple[StormCandidate, list[int]],
                 feb_2022_intdes: str, ballistic_feb_2022: float | None,
                 ballistic_band_feb_2022: tuple[float, float] | None = None) -> dict[str, Any]:
    """Draft config. Feb 2022 (Group 4-7) is held out; May 2024 is a second test-only storm."""
    feb = {"name": "feb_2022", "role": "heldout_test", "start": "2022-02-03", "end": "2022-02-12",
           "onset": "2022-02-03T12:00:00+00:00", "launch_intdes": feb_2022_intdes,
           "ballistic": ballistic_feb_2022,
           "ballistic_band": list(ballistic_band_feb_2022) if ballistic_band_feb_2022 else None}
    events = [feb, _event("may_2024", *may_2024, role="test_only")]
    events += [_event(s.name, s, ids, role="train") for s, ids in training]
    return {"_comment": "DRAFT for human review. Not yet used for training. Passive objects "
                        "(rocket bodies) are used for the training and May 2024 storms; "
                        "ballistic=null means estimate from the pre-storm element sets.",
            "heldout": "feb_2022", "events": events}
