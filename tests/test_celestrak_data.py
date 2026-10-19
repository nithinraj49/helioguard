"""CelesTrak orbit data: GP elements, TLE/sgp4 altitude, SATCAT decay records. Offline fixtures.

Fixtures are real CelesTrak records fetched 2026-10-19 for STARLINK-3181 (NORAD 51461) and
the 2022-010 launch. CelesTrak serves current elements only; true orbit histories need
Space-Track, so the history reader takes TLE text but is fed a one-epoch fixture here.
"""
from datetime import date
from pathlib import Path

import pytest

from helioguard.data.celestrak import (decayed_between, load_satcat, mean_altitude_km,
                                       parse_gp_csv, parse_tle_history, sgp4_altitude_km)

FIX = Path(__file__).parent / "fixtures"


def test_gp_csv_parses():
    recs = parse_gp_csv((FIX / "gp_51461.csv").read_text())
    assert len(recs) == 1
    r = recs[0]
    assert r.norad_id == 51461 and r.name == "STARLINK-3181"
    assert r.mean_motion_rev_day == pytest.approx(15.34387661)
    assert r.inclination_deg == pytest.approx(53.1594)


def test_mean_altitude_matches_satcat_apogee_perigee():
    r = parse_gp_csv((FIX / "gp_51461.csv").read_text())[0]
    assert mean_altitude_km(r) == pytest.approx(463, abs=4)       # SATCAT: 462 x 464 km


def test_sgp4_altitude_agrees_with_mean_motion_altitude():
    hist = parse_tle_history((FIX / "tle_51461.txt").read_text())
    assert len(hist) == 1
    assert sgp4_altitude_km(hist[0]) == pytest.approx(463, abs=10)


def test_tle_history_is_sorted_by_epoch():
    text = (FIX / "tle_51461.txt").read_text()
    both = parse_tle_history(text + text)
    assert len(both) == 2
    assert both[0].epoch <= both[1].epoch


def test_malformed_tle_is_rejected():
    with pytest.raises(ValueError):
        parse_tle_history("NAME\n1 garbage\n2 garbage\n")


def test_satcat_decay_records():
    rows = load_satcat(FIX / "satcat_2022-010.csv")
    pay = [r for r in rows if r.object_type == "PAY"]
    assert len(pay) == 17
    lost = decayed_between(pay, date(2022, 2, 3), date(2022, 3, 1))
    assert len(lost) == 6                    # only what this SATCAT listing shows
    assert all(r.decay_date is not None for r in lost)
