"""End-to-end plumbing of the QNN density-correction check, on SYNTHETIC histories.

A fake Space-Track transport serves generated element sets, so this proves the pipeline wires
together (fetch -> effective density -> train on other storms -> held-out Feb 2022 -> decay-
prediction gate). It says nothing about whether the QNN helps on real satellites: that needs
real Group 4-7 histories (Space-Track credentials) and has NOT been run.
"""
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from helioguard.data.spacetrack import Credentials, SpaceTrackClient
from helioguard.tools.density_pipeline import EventSpec, run_density_pipeline
from tle_builder import B_TRUE, synth_history

FIX = Path(__file__).parent / "fixtures"
UTC = timezone.utc


class SyntheticWorld:
    """Serves 3LE text for any gp_history query; storm enhancement peaks 36 h after the start."""

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


def events():
    def spec(name, start, ids, ballistic):
        s = datetime.fromisoformat(start).replace(tzinfo=UTC)
        return EventSpec(name, s.date(), date.fromordinal(s.date().toordinal() + 4),
                         onset=s + timedelta(hours=24),
                         norad_ids=ids, ballistic=ballistic)
    return [spec("synth_a", "2022-01-10", [81001, 81002], B_TRUE),
            spec("synth_b", "2022-01-20", [82001, 82002], None),        # B from the quiet segment
            spec("synth_c", "2022-02-10", [83001, 83002], B_TRUE),
            spec("feb_2022", "2022-02-03", [80001, 80002, 80003], B_TRUE)]


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    world = SyntheticWorld()
    client = SpaceTrackClient(Credentials("x@example.com", "pw"), tmp_path_factory.mktemp("st"),
                              transport=world, min_interval_s=0)
    return run_density_pipeline(client, events(), "feb_2022", FIX / "sw_feb2022.csv",
                                epochs=40, seed=0), world


def test_one_batched_query_per_event(outcome):
    _, world = outcome
    assert len(world.gets) == 4 and all("gp_history" in u for u in world.gets)


def test_feb_2022_is_held_out_of_training(outcome):
    res, _ = outcome
    rep = res["report"]
    assert rep["heldout_event"] == "feb_2022"
    assert "feb_2022" not in rep["train_events"]
    assert sorted(rep["train_events"]) == ["synth_a", "synth_b", "synth_c"]
    assert rep["n_train"] > 0 and rep["n_heldout"] > 0


def test_report_has_the_decay_check_for_all_three_densities(outcome):
    rep = outcome[0]["report"]
    for key in ("uncorrected_mae_km", "qnn_mae_km", "mlp_mae_km", "qnn_params", "mlp_params",
                "qnn_decay_gate", "qnn_rmse_gate", "success_criterion_met", "uncorrected_rmse", "qnn_rmse"):
        assert key in rep
    assert isinstance(rep["success_criterion_met"], bool)
    assert rep["n_decay_intervals"] > 0


def test_corrector_is_enabled_only_if_both_gates_pass(outcome):
    res = outcome[0]
    rep = res["report"]
    both = rep["qnn_decay_gate"] and rep["qnn_rmse_gate"]
    assert res["corrector"].enabled == both == rep["success_criterion_met"]


def test_unknown_heldout_event_is_an_error(tmp_path):
    client = SpaceTrackClient(Credentials("x@example.com", "pw"), tmp_path,
                              transport=SyntheticWorld(), min_interval_s=0)
    with pytest.raises(ValueError, match="held"):
        run_density_pipeline(client, events(), "may_2024", FIX / "sw_feb2022.csv", epochs=2)
