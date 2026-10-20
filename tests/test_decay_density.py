"""Decay-derived effective density from TLE histories, and the decay-prediction check.

All histories here are SYNTHETIC: built from a known ballistic coefficient and a known density
multiplier, to prove the pipeline recovers them. They say nothing about real satellites.
"""
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from helioguard.atmosphere import density
from helioguard.data.celestrak import parse_tle_history
from helioguard.data.noaa import load_space_weather
from helioguard.physics import MU, R_EARTH
from helioguard.tools.decay_density import (decay_gate, effective_density_samples,
                                            estimate_ballistic_coefficient, evaluate_decay_prediction,
                                            subpoint)
from tle_builder import B_TRUE, synth_history

FIX = Path(__file__).parent / "fixtures"
T0 = datetime(2022, 2, 3, 12, 0, tzinfo=timezone.utc)
ONSET = datetime(2022, 2, 4, 6, 0, tzinfo=timezone.utc)


def rho_ref(alt_km, t_h):
    return density(alt_km, 1.0)


def quiet_then_storm(h):                # density x1 for the first 24 h, x1.6 afterwards
    return 1.0 if h < 24 else 1.6


@pytest.fixture(scope="module")
def sw():
    return load_space_weather(FIX / "sw_feb2022.csv")


@pytest.fixture(scope="module")
def history():
    return synth_history(80001, T0, quiet_then_storm)


def test_synthetic_history_decays(history):
    assert len(history) == 10 and history[0].epoch < history[-1].epoch


def test_ballistic_coefficient_is_recovered_from_the_quiet_segment(history):
    quiet = [t for t in history if t.epoch <= T0 + timedelta(hours=24)]
    b = estimate_ballistic_coefficient(quiet, rho_ref, T0)
    assert b == pytest.approx(B_TRUE, rel=0.05)


def test_effective_density_ratio_is_recovered(history, sw):
    samples = effective_density_samples(history, B_TRUE, rho_ref, T0, sw, ONSET)
    quiet = [s.target for s in samples if s.hours_from_start < 24 - 8]
    storm = [s.target for s in samples if s.hours_from_start > 24 + 4]
    assert quiet and storm
    assert sum(quiet) / len(quiet) == pytest.approx(0.0, abs=0.06)
    assert sum(storm) / len(storm) == pytest.approx(math.log(1.6), abs=0.06)


def test_samples_carry_the_six_documented_features(history, sw):
    s = effective_density_samples(history, B_TRUE, rho_ref, T0, sw, ONSET)[0]
    assert len(s.features) == 6
    alt, f107, ap, hrs, lst, lat = s.features
    assert 300 < alt < 335 and 100 < f107 < 140 and ap >= 0
    assert 0 <= lst < 24 and 0 <= lat <= 53.5
    assert hrs == pytest.approx((s.epoch - ONSET).total_seconds() / 3600 - 0, abs=6)


def test_subpoint_stays_within_the_inclination_band(history):
    for t in history:
        lat, lon = subpoint(t)
        assert abs(lat) <= 53.3 and -180 <= lon <= 180


def test_single_epoch_or_unordered_history_is_rejected(history, sw):
    with pytest.raises(ValueError):
        effective_density_samples(history[:1], B_TRUE, rho_ref, T0, sw, ONSET)


# --- decay prediction check --------------------------------------------------
class ConstantCorrector:
    def __init__(self, ln_k):
        self.ln_k = ln_k

    def log_correction(self, features):
        return self.ln_k


def test_corrected_density_reproduces_observed_decay_better_than_uncorrected(history, sw):
    hist = {80001: history}
    base = evaluate_decay_prediction(hist, {80001: B_TRUE}, rho_ref, T0, sw, ONSET, corrector=None)
    good = evaluate_decay_prediction(hist, {80001: B_TRUE}, rho_ref, T0, sw, ONSET,
                                     corrector=ConstantCorrector(math.log(1.6)))
    assert base["n_intervals"] == good["n_intervals"] == 9
    assert good["mae_km"] < base["mae_km"]


def test_a_wrong_corrector_is_worse_and_the_gate_notices(history, sw):
    hist = {80001: history}
    base = evaluate_decay_prediction(hist, {80001: B_TRUE}, rho_ref, T0, sw, ONSET)
    bad = evaluate_decay_prediction(hist, {80001: B_TRUE}, rho_ref, T0, sw, ONSET,
                                    corrector=ConstantCorrector(1.5))
    assert bad["mae_km"] > base["mae_km"]
    assert decay_gate(base["mae_km"], bad["mae_km"]) is False


def test_gate_requires_a_real_margin():
    assert decay_gate(2.0, 1.0) is True
    assert decay_gate(2.0, 1.99) is False            # 0.5 % better is not "better"
    assert decay_gate(2.0, 2.0) is False
    assert decay_gate(0.0, 0.0) is False


# --- ballistic coefficient calibration ---------------------------------------------
def test_interval_ballistic_and_band_on_a_clean_history():
    from helioguard.tools.decay_density import interval_ballistic, summarize_ballistic
    hist = synth_history(80002, T0, lambda h: 1.0)
    vals = interval_ballistic(hist, rho_ref, T0)
    assert len(vals) == len(hist) - 1
    s = summarize_ballistic([b for _, b in vals])
    assert s["central"] == pytest.approx(B_TRUE, rel=0.05)
    assert s["p16"] <= s["central"] <= s["p84"]
    assert (s["p84"] - s["p16"]) / s["central"] < 0.2            # clean data: a tight band
    assert s["n"] == len(vals)


def test_summarize_requires_data_and_widens_with_scatter():
    from helioguard.tools.decay_density import summarize_ballistic
    with pytest.raises(ValueError):
        summarize_ballistic([])
    tight = summarize_ballistic([0.03, 0.031, 0.029, 0.03])
    wide = summarize_ballistic([0.01, 0.05, 0.02, 0.04])
    assert wide["p84"] - wide["p16"] > tight["p84"] - tight["p16"]


def test_short_intervals_are_ignored_when_a_minimum_length_is_set():
    from helioguard.tools.decay_density import interval_ballistic
    hist = synth_history(80003, T0, lambda h: 1.0, step_h=1, hours=24)       # hourly element sets
    assert len(interval_ballistic(hist, rho_ref, T0)) == 24
    assert len(interval_ballistic(hist, rho_ref, T0, min_interval_s=3 * 3600)) == 0
    sparse = synth_history(80003, T0, lambda h: 1.0, step_h=8, hours=24)
    assert len(interval_ballistic(sparse, rho_ref, T0, min_interval_s=3 * 3600)) == 3
