"""Held-out evaluation of the QNN density correction, on SYNTHETIC histories.

A fake Space-Track transport serves generated element sets, so these tests prove the wiring:
roles, object-overlap exclusion, leave-one-storm-out folds, the quiet-time check across a
ballistic band, the gates, and the same-size MLP reported next to every number. They say
nothing about whether the QNN helps on real satellites; the real run is in the README.
"""
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from helioguard.data.spacetrack import Credentials, SpaceTrackClient
from helioguard.tools.density_pipeline import EventSpec, evaluate_gates, run_density_pipeline
from tle_builder import B_TRUE, synth_history

FIX = Path(__file__).parent / "fixtures"
UTC = timezone.utc


class SyntheticWorld:
    """Serves 3LE text for any gp_history query; enhancement peaks 36 h after the window opens."""

    def __init__(self):
        self.gets = []

    def login(self, identity, password):
        pass

    def get(self, url):
        self.gets.append(url)
        ids = [int(i) for i in re.search(r"NORAD_CAT_ID/([\d,]+)/", url).group(1).split(",")]
        start, end = re.search(r"EPOCH/(\d{4}-\d{2}-\d{2})--(\d{4}-\d{2}-\d{2})", url).groups()
        t0 = datetime.fromisoformat(start).replace(tzinfo=UTC)
        hours = (datetime.fromisoformat(end).replace(tzinfo=UTC) - t0).days * 24
        text = []
        for n in ids:
            for t in synth_history(n, t0, lambda h: 1 + 0.7 * math.exp(-((h - 36) / 10) ** 2),
                                   start_alt=330.0 + (n % 5), hours=hours):
                text += [t.name, t.line1, t.line2]
        return "\n".join(text) + "\n"


def spec(name, start, ids, role="train", ballistic=None, band=None):
    s = datetime.fromisoformat(start).replace(tzinfo=UTC)
    return EventSpec(name, s.date(), s.date() + timedelta(days=4), onset=s + timedelta(hours=24),
                     norad_ids=ids, ballistic=ballistic, role=role, ballistic_band=band)


def events():
    return [spec("synth_a", "2022-01-10", [81001, 81002, 81003]),
            spec("synth_b", "2022-01-20", [82001, 82002, 81002]),          # 81002 also in synth_a
            spec("synth_c", "2022-02-10", [83001, 83002, 83003]),
            spec("test_storm", "2022-02-14", [81001, 84001, 84002], role="heldout_test"),   # 81001 overlaps
            spec("quiet_check", "2022-02-03", [80001, 80002], role="quiet_check",
                 ballistic=B_TRUE, band=(0.5 * B_TRUE, 1.5 * B_TRUE))]


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    world = SyntheticWorld()
    client = SpaceTrackClient(Credentials("x@example.com", "pw"), tmp_path_factory.mktemp("st"),
                              transport=world, min_interval_s=0)
    res = run_density_pipeline(client, events(), FIX / "sw_feb2022.csv", epochs=15, seeds=(0,))
    return res, world


# --- roles ---------------------------------------------------------------------------
def test_unknown_role_or_missing_roles_are_rejected(tmp_path):
    client = SpaceTrackClient(Credentials("x@example.com", "pw"), tmp_path,
                              transport=SyntheticWorld(), min_interval_s=0)
    bad = events()
    bad[0].role = "validation"
    with pytest.raises(ValueError, match="role"):
        run_density_pipeline(client, bad, FIX / "sw_feb2022.csv", epochs=2, seeds=(0,))
    with pytest.raises(ValueError, match="held-out"):
        run_density_pipeline(client, [e for e in events() if e.role != "heldout_test"],
                             FIX / "sw_feb2022.csv", epochs=2, seeds=(0,))
    with pytest.raises(ValueError, match="training"):
        run_density_pipeline(client, [e for e in events() if e.role != "train"],
                             FIX / "sw_feb2022.csv", epochs=2, seeds=(0,))


def test_test_and_quiet_events_are_never_trained_on(outcome):
    rep = outcome[0]["report"]
    assert sorted(rep["training_events"]) == ["synth_a", "synth_b", "synth_c"]
    assert list(rep["heldout"]) == ["test_storm"]
    assert "quiet_check" not in rep["training_events"] and "test_storm" not in rep["training_events"]


# --- object overlap -------------------------------------------------------------------
def test_objects_seen_in_test_storms_are_excluded_from_training(outcome):
    ov = outcome[0]["report"]["overlap"]
    assert ov["excluded"]["synth_a"] == [81001]            # also tracked in the test storm
    assert ov["excluded"]["synth_b"] == [] and ov["excluded"]["synth_c"] == []
    assert ov["n_unique_objects_excluded"] == 1 and ov["n_object_event_pairs_excluded"] == 1
    rep = outcome[0]["report"]
    assert rep["n_train_samples"] < rep["n_train_samples_before_exclusion"]


# --- leave-one-storm-out ---------------------------------------------------------------
def test_leave_one_storm_out_has_one_fold_per_training_storm(outcome):
    folds = outcome[0]["report"]["loso"]["folds"]
    assert sorted(f["test_event"] for f in folds) == ["synth_a", "synth_b", "synth_c"]
    for f in folds:
        assert f["test_event"] not in f["train_events"] and len(f["train_events"]) == 2
        for key in ("uncorrected_mae_km", "qnn_mae_km", "mlp_mae_km", "qnn_improvement",
                    "mlp_improvement", "n_intervals"):
            assert key in f


def test_loso_excludes_objects_shared_with_the_fold_test_storm(outcome):
    fold_a = next(f for f in outcome[0]["report"]["loso"]["folds"] if f["test_event"] == "synth_a")
    assert fold_a["overlap_excluded"]["synth_b"] == [81002]   # 81002 is in synth_a too


# --- held-out and quiet-time reports ------------------------------------------------------
def test_mlp_is_reported_next_to_every_number(outcome):
    rep = outcome[0]["report"]
    for scores in list(rep["heldout"].values()) + list(rep["quiet_check"]["by_ballistic"].values()):
        for key in ("uncorrected_mae_km", "qnn_mae_km", "mlp_mae_km", "qnn_improvement", "mlp_improvement"):
            assert key in scores
    assert rep["qnn_params"] == 61 and rep["mlp_params"] == 57
    assert "mean_mlp_improvement" in rep["loso"] and "mean_qnn_improvement" in rep["loso"]


def test_quiet_check_runs_at_both_ends_and_the_centre_of_the_band(outcome):
    by = outcome[0]["report"]["quiet_check"]["by_ballistic"]
    assert set(by) == {"low", "central", "high"}
    assert by["low"]["ballistic"] < by["central"]["ballistic"] < by["high"]["ballistic"]
    assert all(v["n_intervals"] > 0 for v in by.values())


# --- gates ---------------------------------------------------------------------------------
def test_corrector_is_enabled_only_if_every_gate_passes(outcome):
    res = outcome[0]
    gates = res["report"]["gates"]
    assert res["corrector"].enabled == gates["all_passed"]
    assert gates["all_passed"] == (gates["heldout_pass"] and gates["loso_pass"] and gates["quiet_pass"])


def good():
    return dict(heldout={"may": {"uncorrected_mae_km": 2.0, "qnn_mae_km": 1.5}},
                loso_improvements=[0.10, 0.02, 0.12],
                quiet={"low": {"uncorrected_mae_km": 1.0, "qnn_mae_km": 0.9},
                       "central": {"uncorrected_mae_km": 1.0, "qnn_mae_km": 1.0},
                       "high": {"uncorrected_mae_km": 1.0, "qnn_mae_km": 1.0}})


def test_gates_all_pass_together():
    g = evaluate_gates(**good())
    assert g["all_passed"] and g["heldout_pass"] and g["loso_pass"] and g["quiet_pass"]


def test_each_gate_can_block_enabling_on_its_own():
    a = good(); a["heldout"]["may"]["qnn_mae_km"] = 1.95            # only 2.5 % better on May 2024
    assert not evaluate_gates(**a)["all_passed"] and not evaluate_gates(**a)["heldout_pass"]
    b = good(); b["loso_improvements"] = [0.10, -0.20, 0.06]        # mean 0.0 across folds
    assert not evaluate_gates(**b)["all_passed"] and not evaluate_gates(**b)["loso_pass"]
    c = good(); c["quiet"]["high"]["qnn_mae_km"] = 1.2              # worse at one end of the band
    assert not evaluate_gates(**c)["all_passed"] and not evaluate_gates(**c)["quiet_pass"]
    d = good(); d["quiet"]["low"]["qnn_mae_km"] = 1.2
    assert not evaluate_gates(**d)["quiet_pass"]


def test_the_centre_of_the_band_is_reported_but_the_ends_decide():
    c = good(); c["quiet"]["central"]["qnn_mae_km"] = 1.5
    g = evaluate_gates(**c)
    assert g["quiet_pass"] and g["quiet"]["central"]["no_worse"] is False


def test_gates_need_five_percent_and_ignore_the_mlp():
    g = good(); g["heldout"]["may"]["qnn_mae_km"] = 1.9            # exactly 5 % better passes
    assert evaluate_gates(**g)["heldout_pass"]
    g["heldout"]["may"]["qnn_mae_km"] = 1.91                       # 4.5 % does not
    assert not evaluate_gates(**g)["heldout_pass"]
    g = good(); g["heldout"]["may"]["mlp_mae_km"] = 0.1             # a great MLP changes nothing
    assert evaluate_gates(**g)["heldout_pass"]
    assert not evaluate_gates(heldout={}, loso_improvements=[0.2], quiet=good()["quiet"])["all_passed"]
    assert not evaluate_gates(good()["heldout"], [], good()["quiet"])["all_passed"]
