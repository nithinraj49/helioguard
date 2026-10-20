"""Decay-derived effective density from TLE histories, and the decay-prediction check.

Circular-orbit drag decay:  da/dt = -rho * B * sqrt(mu * a),  B = Cd*A/m  (m^2/kg).
From the change in mean motion between two element sets we get da/dt, hence an effective
density rho_eff for that interval; the learning target is ln(rho_eff / rho_reference).
B is estimated per satellite from a quiet period. TLE mean motion is treated as the mean
motion of a circular orbit (small Kozai/Brouwer difference ignored): an approximation.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Protocol, Sequence

from sgp4.api import Satrec
from sgp4.propagation import gstime

from ..data.celestrak import TLE
from ..data.noaa import DailyIndices
from ..physics import MU, R_EARTH

RhoFn = Callable[[float, float], float]        # (alt_km, hours since t0) -> kg/m^3


class Corrector(Protocol):
    def log_correction(self, features: Sequence[float]) -> float: ...


@dataclass(frozen=True)
class Sample:
    features: tuple[float, ...]   # alt_km, F10.7, ap, hours since onset, local solar time h, |lat| deg
    target: float                 # ln(rho_eff / rho_reference)
    epoch: datetime               # interval midpoint
    hours_from_start: float


def tle_mean_motion(tle: TLE) -> float:
    return float(tle.line2[52:63])


def _sma_m(tle: TLE) -> float:
    n = tle_mean_motion(tle) * 2 * math.pi / 86400.0
    return (MU / n ** 2) ** (1 / 3)


def subpoint(tle: TLE) -> tuple[float, float]:
    """(geocentric latitude, longitude) in degrees under the satellite at the TLE epoch."""
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    err, r, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if err:
        raise ValueError(f"SGP4 error code {err}")
    lat = math.degrees(math.asin(r[2] / math.sqrt(sum(c * c for c in r))))
    lon = math.degrees(math.atan2(r[1], r[0]) - gstime(sat.jdsatepoch + sat.jdsatepochF))
    return lat, (lon + 180.0) % 360.0 - 180.0


def _pairs(tles: Sequence[TLE]):
    if len(tles) < 2:
        raise ValueError("need at least two element sets")
    for a, b in zip(tles, tles[1:]):
        if b.epoch <= a.epoch:
            raise ValueError("element sets must be in increasing epoch order")
        yield a, b


def _decay_rate(a: TLE, b: TLE) -> tuple[float, float, float]:
    """(da/dt m/s, mid semi-major axis m, interval seconds)."""
    dt = (b.epoch - a.epoch).total_seconds()
    a1, a2 = _sma_m(a), _sma_m(b)
    return (a2 - a1) / dt, (a1 + a2) / 2, dt


def _hours(t0: datetime, when: datetime) -> float:
    return (when - t0).total_seconds() / 3600


def estimate_ballistic_coefficient(quiet_tles: Sequence[TLE], rho_fn: RhoFn, t0: datetime) -> float:
    """Median B = -(da/dt) / (rho_ref * sqrt(mu*a)) over quiet intervals."""
    values = []
    for a, b in _pairs(quiet_tles):
        rate, a_mid, _ = _decay_rate(a, b)
        mid = a.epoch + (b.epoch - a.epoch) / 2
        rho = rho_fn((a_mid - R_EARTH) / 1e3, _hours(t0, mid))
        if rate < 0:
            values.append(-rate / (rho * math.sqrt(MU * a_mid)))
    if not values:
        raise ValueError("no decaying intervals to estimate the ballistic coefficient")
    return statistics.median(values)


def interval_ballistic(tles: Sequence[TLE], rho_fn: RhoFn, t0: datetime,
                       min_interval_s: float = 0.0) -> list[tuple[datetime, float]]:
    """(interval midpoint, B) for every decaying interval: B = -(da/dt) / (rho_ref * sqrt(mu*a)).
    Intervals shorter than `min_interval_s` are skipped (TLE fit noise dominates them)."""
    out = []
    for a, b in _pairs(tles):
        rate, a_mid, dt = _decay_rate(a, b)
        if rate < 0 and dt >= min_interval_s:
            mid = a.epoch + (b.epoch - a.epoch) / 2
            rho = rho_fn((a_mid - R_EARTH) / 1e3, _hours(t0, mid))
            out.append((mid, -rate / (rho * math.sqrt(MU * a_mid))))
    return out


def summarize_ballistic(values: Sequence[float]) -> dict:
    """Central value (median) and a 16-84 percentile band of per-interval B estimates."""
    if not values:
        raise ValueError("no ballistic coefficient estimates")
    v = sorted(values)

    def pct(q: float) -> float:
        pos = q * (len(v) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return {"central": statistics.median(v), "p16": pct(0.16), "p84": pct(0.84), "n": len(v)}


def _features(a: TLE, b: TLE, alt_km: float, sw: Sequence[DailyIndices], onset: datetime) -> tuple[float, ...]:
    mid = a.epoch + (b.epoch - a.epoch) / 2
    row = next((r for r in sw if r.day == mid.date()), None)
    if row is None:
        raise ValueError(f"no space-weather data for {mid.date()}")
    lat, lon = subpoint(a)
    lst = (mid.hour + mid.minute / 60 + lon / 15.0) % 24.0
    return (alt_km, row.f107_obs, float(row.ap_3h[mid.hour // 3]), _hours(onset, mid), lst, abs(lat))


def effective_density_samples(tles: Sequence[TLE], ballistic: float, rho_fn: RhoFn, t0: datetime,
                              sw: Sequence[DailyIndices], onset: datetime,
                              min_interval_s: float = 0.0) -> list[Sample]:
    samples = []
    for a, b in _pairs(tles):
        rate, a_mid, dt = _decay_rate(a, b)
        if rate >= 0 or dt < min_interval_s:
            continue                          # raising or noise: no drag signal in this interval
        alt = (a_mid - R_EARTH) / 1e3
        mid = a.epoch + (b.epoch - a.epoch) / 2
        rho_eff = -rate / (ballistic * math.sqrt(MU * a_mid))
        target = math.log(rho_eff / rho_fn(alt, _hours(t0, mid)))
        samples.append(Sample(_features(a, b, alt, sw, onset), target, mid, _hours(t0, mid)))
    return samples


def evaluate_decay_prediction(histories: dict[int, Sequence[TLE]], ballistic: dict[int, float],
                              rho_fn: RhoFn, t0: datetime, sw: Sequence[DailyIndices],
                              onset: datetime, corrector: Corrector | None = None,
                              dt_s: float = 60.0, min_interval_s: float = 0.0,
                              decaying_only: bool = False) -> dict:
    """Predict each observed altitude change over each interval, starting from the observed
    altitude, with the reference density (times exp(correction) if a corrector is given).
    Intervals shorter than `min_interval_s` are skipped; with `decaying_only`, so are intervals
    where the object was observed to rise (thrusting). Returns the mean absolute error of the
    altitude change in km."""
    errors = []
    for norad, tles in histories.items():
        b_coef = ballistic[norad]
        for a, b in _pairs(tles):
            if (b.epoch - a.epoch).total_seconds() < min_interval_s:
                continue
            if decaying_only and _sma_m(b) >= _sma_m(a):
                continue
            alt_mid = ((_sma_m(a) + _sma_m(b)) / 2 - R_EARTH) / 1e3
            c = corrector.log_correction(_features(a, b, alt_mid, sw, onset)) if corrector else 0.0
            k = math.exp(c)
            sma, t = _sma_m(a), 0.0
            span = (b.epoch - a.epoch).total_seconds()
            while t < span:
                step = min(dt_s, span - t)
                rho = k * rho_fn((sma - R_EARTH) / 1e3, _hours(t0, a.epoch + timedelta(seconds=t)))
                sma -= rho * b_coef * math.sqrt(MU * sma) * step
                t += step
            errors.append(abs((sma - _sma_m(b)) / 1e3))
    if not errors:
        raise ValueError("no intervals to evaluate")
    return {"mae_km": sum(errors) / len(errors), "n_intervals": len(errors),
            "interval_errors_km": errors}


def decay_gate(uncorrected_mae: float, corrected_mae: float, min_improvement: float = 0.05) -> bool:
    """True only if the corrected density reproduces observed decay better by a real margin."""
    return uncorrected_mae > 0 and corrected_mae <= (1.0 - min_improvement) * uncorrected_mae - 1e-12
