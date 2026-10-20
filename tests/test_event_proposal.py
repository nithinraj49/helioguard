"""Proposing training storms and passive objects for the QNN check (offline, SYNTHETIC histories)."""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from helioguard.data.noaa import load_space_weather
from helioguard.tools.event_proposal import build_config, find_storms, usable_objects
from tle_builder import synth_history

FIX = Path(__file__).parent / "fixtures"
UTC = timezone.utc


@pytest.fixture(scope="module")
def sw_may():
    return load_space_weather(FIX / "sw_may2024.csv")


def test_finds_the_may_2024_storm_first_with_its_onset(sw_may):
    storms = find_storms(sw_may, min_kp=7.0, exclude=[])
    assert [s.onset.month for s in storms] == [5, 6, 4]          # ranked by Kp: May 9.0, Jun 7.7, Apr 7.0
    s = storms[0]
    assert s.kp_max == pytest.approx(9.0) and date(2024, 5, 10) <= s.onset.date() <= date(2024, 5, 11)
    assert s.onset.tzinfo is not None
    assert s.start == s.onset.date() - timedelta(days=3) and s.end == s.onset.date() + timedelta(days=4)


def test_excluded_windows_are_not_proposed(sw_may):
    rest = find_storms(sw_may, min_kp=7.0, exclude=[(date(2024, 5, 8), date(2024, 5, 14))])
    assert [s.onset.month for s in rest] == [6, 4]
    wide = find_storms(sw_may, min_kp=7.0, exclude=[(date(2024, 5, 8), date(2024, 5, 14))],
                       exclusion_margin_days=60)
    assert wide == []                                            # margin reaches April and June


def test_threshold_filters_weaker_storms(sw_may):
    assert [s.kp_max for s in find_storms(sw_may, min_kp=8.0, exclude=[])] == [9.0]
    assert find_storms(sw_may, min_kp=9.5, exclude=[]) == []


# --- object filtering ---------------------------------------------------------------
T0 = datetime(2024, 5, 8, tzinfo=UTC)
ONSET = datetime(2024, 5, 10, 17, tzinfo=UTC)


def hist(norad, hours=168, step=8, **kw):
    return synth_history(norad, T0, lambda h: 1.0, hours=hours, step_h=step, **kw)


def test_usable_objects_keeps_passive_decaying_objects_with_enough_sets():
    good = hist(90001)
    out = usable_objects({90001: good}, ONSET)
    assert out == [90001]


def test_objects_with_too_few_pre_or_post_storm_sets_are_dropped():
    only_post = [t for t in hist(90002) if t.epoch >= ONSET]
    only_pre = [t for t in hist(90003) if t.epoch < ONSET]
    sparse = hist(90004, step=48)
    assert usable_objects({90002: only_post, 90003: only_pre, 90004: sparse}, ONSET) == []


def test_manoeuvring_objects_are_dropped():
    from dataclasses import replace
    from tle_builder import make_tle
    h = hist(90005)
    k = len(h) // 2
    lines = make_tle(90005, 24, 130.0, 15.9, 53.2)   # a jump to a much lower altitude, then up
    bumped = list(h)
    t = bumped[k]
    bumped[k] = replace(t, line1=lines[0], line2=lines[1])
    assert usable_objects({90005: bumped}, ONSET) == []


def test_max_objects_cap_prefers_better_tracked_objects():
    a, b = hist(90006, step=8), hist(90007, step=12)
    assert usable_objects({90006: a, 90007: b}, ONSET, max_objects=1) == [90006]


# --- config ---------------------------------------------------------------------------
def test_config_roles_may_is_the_held_out_test_and_feb_is_a_quiet_check(sw_may):
    storm = find_storms(sw_may, min_kp=8.0, exclude=[])[0]
    cfg = build_config(training=[(storm, [90001, 90002])], may_2024=(storm, [90003]),
                       feb_2022_intdes="2022-010", ballistic_feb_2022=0.125,
                       ballistic_band_feb_2022=(0.071, 0.192))
    json.dumps(cfg)                                          # serialisable
    names = {e["name"]: e for e in cfg["events"]}
    assert names["may_2024"]["role"] == "heldout_test"
    feb = names["feb_2022"]
    assert feb["role"] == "quiet_check" and feb["launch_intdes"] == "2022-010"
    assert feb["ballistic"] == 0.125 and feb["ballistic_band"] == [0.071, 0.192]
    train = [e for e in cfg["events"] if e["role"] == "train"]
    assert len(train) == 1 and train[0]["ballistic"] is None
    assert all(e["name"] not in ("feb_2022", "may_2024") for e in train)
    assert "heldout" not in cfg                              # roles live on the events


def test_noisy_short_intervals_do_not_make_an_object_look_like_it_manoeuvres():
    from dataclasses import replace
    from tle_builder import make_tle
    h = hist(90008, step=8)
    k = len(h) // 2
    t = h[k]
    near = make_tle(90008, t.epoch.year % 100, t.epoch.timetuple().tm_yday + (t.epoch.hour * 60 + t.epoch.minute + 20) / 1440,
                    float(t.line2[52:63]) - 0.0005)       # extra set 20 min later, mean motion slightly lower
    noisy = h[:k + 1] + [replace(t, line1=near[0], line2=near[1], epoch=t.epoch + timedelta(minutes=20))] + h[k + 1:]
    assert usable_objects({90008: noisy}, ONSET) == [90008]
