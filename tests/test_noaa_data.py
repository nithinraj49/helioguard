"""Space-weather indices (CelesTrak SW-All, compiled from NOAA/GFZ). Offline fixtures only."""
from datetime import date
from pathlib import Path

import pytest

from helioguard.data.noaa import load_space_weather, storm_summary

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def feb2022():
    return load_space_weather(FIX / "sw_feb2022.csv")


@pytest.fixture(scope="module")
def may2024():
    return load_space_weather(FIX / "sw_may2024.csv")


def test_rows_parse_into_daily_records(feb2022):
    row = next(r for r in feb2022 if r.day == date(2022, 2, 4))
    assert len(row.kp_3h) == 8 and len(row.ap_3h) == 8
    assert row.f107_obs == pytest.approx(129.6)
    assert row.ap_daily == 32
    assert max(row.kp_3h) == pytest.approx(5.3)


def test_feb_2022_is_a_minor_g1_storm(feb2022):
    s = storm_summary(feb2022, date(2022, 2, 3), date(2022, 2, 5))
    assert s["kp_max"] == pytest.approx(5.3)
    assert s["g_scale"] == "G1"


def test_may_2024_is_a_g5_storm(may2024):
    s = storm_summary(may2024, date(2024, 5, 10), date(2024, 5, 12))
    assert s["kp_max"] == pytest.approx(9.0)
    assert s["g_scale"] == "G5"
    assert s["ap_max"] >= 300


def test_quiet_period_is_g0(feb2022):
    assert storm_summary(feb2022, date(2022, 1, 10), date(2022, 1, 12))["g_scale"] == "G0"


def test_empty_window_is_an_error(feb2022):
    with pytest.raises(ValueError):
        storm_summary(feb2022, date(2030, 1, 1), date(2030, 1, 2))
