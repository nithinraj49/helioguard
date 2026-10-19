"""pymsis (NRLMSIS) density driven by real F10.7 / Ap, with the old model as fallback."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pymsis
import pytest

from helioguard.atmosphere import density as ref_density
from helioguard.atmosphere_msis import MsisDensity, get_density_fn
from helioguard.physics import simulate
from helioguard.scenarios import STARLINK_LIKE as SC

FIX = Path(__file__).parent / "fixtures"
FEB_START = datetime(2022, 2, 3, 18, 0)


@pytest.fixture(scope="module")
def feb():
    return MsisDensity.from_file(FIX / "sw_feb2022.csv", FEB_START, hours=48)


def test_density_positive_and_falls_with_altitude(feb):
    vals = [feb(alt, 0.0) for alt in (160, 200, 250, 300, 400, 500)]
    assert all(v > 0 for v in vals)
    assert vals == sorted(vals, reverse=True)


def test_density_responds_to_the_real_storm(feb):
    series = [feb(350.0, h) for h in range(0, 49)]
    assert max(series) / min(series) > 1.1          # real indices change density with time


def test_single_point_matches_pymsis_directly():
    m = MsisDensity.from_file(FIX / "sw_feb2022.csv", FEB_START, hours=48, lats=[0.0], lons=[0.0])
    when = np.datetime64("2022-02-04T12:00", "m")    # 18 h after the start
    direct = float(np.ravel(pymsis.calculate([when], [0.0], [0.0], [300.0])[..., 0])[0])
    assert m(300.0, 18.0) == pytest.approx(direct, rel=1e-3)


def test_storm_time_mode_matches_pymsis_directly():
    m = MsisDensity.from_file(FIX / "sw_feb2022.csv", FEB_START, hours=48, lats=[0.0], lons=[0.0],
                              storm_time=True)
    when = np.datetime64("2022-02-04T12:00", "m")
    direct = float(np.ravel(pymsis.calculate([when], [0.0], [0.0], [300.0],
                                             geomagnetic_activity=-1)[..., 0])[0])
    assert m(300.0, 18.0) == pytest.approx(direct, rel=1e-3)


def test_storm_time_mode_differs_from_daily_ap_mode():
    daily = MsisDensity.from_file(FIX / "sw_feb2022.csv", FEB_START, hours=48)
    storm = MsisDensity.from_file(FIX / "sw_feb2022.csv", FEB_START, hours=48, storm_time=True)
    assert any(abs(daily(210.0, h) / storm(210.0, h) - 1) > 0.01 for h in range(49))


def test_density_change_report_structure_and_consistency():
    from helioguard.atmosphere_msis import storm_density_change
    for storm_time in (False, True):
        r = storm_density_change(FIX / "sw_feb2022.csv", 210.0, storm_time=storm_time)
        assert r["baseline_day"] == "2022-02-02" and r["storm_time_mode"] is storm_time
        assert r["baseline_density"] > 0
        assert r["peak_local_ratio"] >= r["peak_orbit_avg_ratio"] > 0
        assert 0 <= r["peak_hour"] <= 72


def test_clamps_outside_the_time_window(feb):
    assert feb(300.0, -5.0) == feb(300.0, 0.0)
    assert feb(300.0, 500.0) == feb(300.0, 48.0)


def test_simulate_with_constant_density_fn_equals_storm_multiplier():
    a = simulate(SC, 210.0, "edge_on", 1.5, 48)
    b = simulate(SC, 210.0, "edge_on", 1.0, 48, density_fn=lambda alt, t: ref_density(alt, 1.5))
    assert a == b


def test_fallback_to_reference_atmosphere_when_data_missing(tmp_path):
    fn, source = get_density_fn(tmp_path / "nope.csv", FEB_START, 48, storm_multiplier=1.5)
    assert source == "reference"
    assert fn(210.0, 3.0) == ref_density(210.0, 1.5)


def test_real_source_is_labelled():
    fn, source = get_density_fn(FIX / "sw_feb2022.csv", FEB_START, 48)
    assert source == "pymsis"
