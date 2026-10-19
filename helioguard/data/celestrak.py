"""CelesTrak orbit data: GP elements, TLE history (sgp4) and SATCAT decay records.

CelesTrak serves current elements only; historical TLEs need Space-Track, so the history
reader accepts TLE text from any source.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sgp4.api import Satrec

from ..physics import MU, R_EARTH
from .cache import Downloader, fetch_cached, http_download

GP_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT={fmt}"
SATCAT_URL = "https://celestrak.org/satcat/records.php?INTDES={intdes}&FORMAT=csv"


@dataclass(frozen=True)
class GPRecord:
    name: str
    norad_id: int
    epoch: datetime
    mean_motion_rev_day: float
    eccentricity: float
    inclination_deg: float


@dataclass(frozen=True)
class TLE:
    name: str
    line1: str
    line2: str
    epoch: datetime


@dataclass(frozen=True)
class SatcatRow:
    name: str
    intl_designator: str
    norad_id: int
    object_type: str
    launch_date: date | None
    decay_date: date | None


def fetch_gp(catnr: int, cache_dir: Path, *, fmt: str = "csv", max_age_days: float = 1.0,
             downloader: Downloader = http_download) -> Path:
    return fetch_cached(GP_URL.format(catnr=catnr, fmt=fmt),
                        Path(cache_dir) / f"gp_{catnr}.{fmt}",
                        max_age_days=max_age_days, downloader=downloader)


def fetch_satcat(intdes: str, cache_dir: Path, *, max_age_days: float = 7.0,
                 downloader: Downloader = http_download) -> Path:
    return fetch_cached(SATCAT_URL.format(intdes=intdes),
                        Path(cache_dir) / f"satcat_{intdes}.csv",
                        max_age_days=max_age_days, downloader=downloader)


def parse_gp_csv(text: str) -> list[GPRecord]:
    return [GPRecord(name=r["OBJECT_NAME"], norad_id=int(r["NORAD_CAT_ID"]),
                     epoch=datetime.fromisoformat(r["EPOCH"]),
                     mean_motion_rev_day=float(r["MEAN_MOTION"]),
                     eccentricity=float(r["ECCENTRICITY"]),
                     inclination_deg=float(r["INCLINATION"]))
            for r in csv.DictReader(io.StringIO(text))]


def mean_altitude_km(rec: GPRecord) -> float:
    """Altitude of the mean semi-major axis above the equatorial radius (km)."""
    n = rec.mean_motion_rev_day * 2 * math.pi / 86400.0
    return (MU / n ** 2) ** (1 / 3) / 1e3 - R_EARTH / 1e3


def _tle_epoch(line1: str) -> datetime:
    yy, day = int(line1[18:20]), float(line1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)


def parse_tle_history(text: str) -> list[TLE]:
    """Parse 2- or 3-line TLE sets (optionally many) into records sorted by epoch."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out, name, i = [], "", 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            l1, l2 = lines[i], lines[i + 1]
            if len(l1) < 69 or len(l2) < 69:
                raise ValueError(f"malformed TLE lines: {l1!r}")
            out.append(TLE(name, l1, l2, _tle_epoch(l1)))
            name, i = "", i + 2
        elif lines[i].startswith(("1 ", "2 ")):
            raise ValueError(f"malformed TLE near: {lines[i]!r}")
        else:
            name, i = lines[i].strip(), i + 1
    return sorted(out, key=lambda t: t.epoch)


def sgp4_altitude_km(tle: TLE) -> float:
    """Geocentric altitude at the TLE epoch from SGP4 (spherical Earth, km)."""
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    err, r, _ = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
    if err:
        raise ValueError(f"SGP4 error code {err}")
    return math.sqrt(sum(c * c for c in r)) - R_EARTH / 1e3


def _opt_date(s: str) -> date | None:
    return date.fromisoformat(s) if s else None


def load_satcat(path: Path) -> list[SatcatRow]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [SatcatRow(r["OBJECT_NAME"], r["OBJECT_ID"], int(r["NORAD_CAT_ID"]),
                          r["OBJECT_TYPE"], _opt_date(r["LAUNCH_DATE"]), _opt_date(r["DECAY_DATE"]))
                for r in csv.DictReader(fh)]


def decayed_between(rows: list[SatcatRow], start: date, end: date) -> list[SatcatRow]:
    return [r for r in rows if r.decay_date is not None and start <= r.decay_date <= end]
