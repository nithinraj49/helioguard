"""Hybrid QNN density correction vs a same-size classical network, with Feb 2022 held out.

Datasets here are SYNTHETIC (a known smooth function plus noise). They test the machinery
and the gate; they are not evidence that a QNN helps on real satellite decay.
"""
import math

import numpy as np
import pytest
import torch

from helioguard.tools.qnn_density import (DensityCorrector, QnnDensityCorrection,
                                          build_classical_mlp, build_hybrid_qnn, compare_on_heldout,
                                          count_parameters, rmse, scale_features, train_model)
from verispace_engine.schema import ToolCall
from verispace_engine.tools import Tool, ToolRegistry


def synthetic_event(seed, n=70, noise=0.02, signal=True):
    """Features: alt, F10.7, Ap, hours since onset, local solar time, |lat|. Target: log ratio."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(200, 450, n), rng.uniform(80, 220, n), rng.uniform(5, 300, n),
                         rng.uniform(0, 48, n), rng.uniform(0, 24, n), rng.uniform(0, 60, n)])
    y = (0.45 * np.tanh((X[:, 2] - 80) / 80) + 0.15 * np.cos(2 * math.pi * X[:, 4] / 24)
         if signal else np.zeros(n))
    return X, y + rng.normal(0, noise if signal else 0.25, n)


# --- sizes and scaling --------------------------------------------------------
def test_hybrid_qnn_has_the_documented_parameter_count():
    qnn = build_hybrid_qnn(seed=0)
    assert count_parameters(qnn) == 54 + 7            # 3 layers x 6 qubits x 3 angles, + linear head


def test_classical_baseline_is_the_same_size():
    qnn, mlp = build_hybrid_qnn(seed=0), build_classical_mlp(seed=0)
    assert abs(count_parameters(mlp) - count_parameters(qnn)) / count_parameters(qnn) <= 0.10


def test_scaling_maps_features_into_zero_to_pi_and_clips():
    X = np.array([[200, 70, 0, 0, 0, 0], [600, 300, 400, 100, 24, 90], [325, 150, 100, 20, 12, 30]])
    s = scale_features(X)
    assert s.min() >= 0 and s.max() <= math.pi + 1e-9
    assert s.shape == (3, 6)
    far = scale_features(np.array([[1e5, 1e5, 1e5, 1e5, 1e5, 1e5]]))
    assert far.max() <= math.pi + 1e-9                 # out-of-range inputs are clipped


# --- training -----------------------------------------------------------------
def test_training_reduces_error_for_both_models():
    X, y = synthetic_event(1)
    Xs = scale_features(X)
    for model in (build_hybrid_qnn(seed=0), build_classical_mlp(seed=0)):
        before = rmse(model, Xs, y)
        train_model(model, Xs, y, epochs=60, lr=0.05, seed=0)
        assert rmse(model, Xs, y) < before


# --- held-out comparison and gate ----------------------------------------------
def storms(signal=True):
    return {f"storm_{i}": synthetic_event(10 + i, signal=signal) for i in range(3)}


def test_report_compares_qnn_mlp_and_uncorrected_on_the_held_out_storm():
    rep = compare_on_heldout(storms(), "feb_2022", synthetic_event(99), epochs=80, seed=0)
    for key in ("uncorrected_rmse", "qnn_rmse", "mlp_rmse", "qnn_params", "mlp_params",
                "qnn_train_s", "mlp_train_s", "qnn_beats_uncorrected", "qnn_beats_mlp",
                "heldout_event", "train_events"):
        assert key in rep
    assert rep["heldout_event"] == "feb_2022" and "feb_2022" not in rep["train_events"]
    assert rep["uncorrected_rmse"] == pytest.approx(math.sqrt(np.mean(synthetic_event(99)[1] ** 2)))
    assert rep["qnn_rmse"] < rep["uncorrected_rmse"]          # the synthetic signal is learnable


def test_training_on_the_held_out_event_is_refused():
    train = storms()
    train["feb_2022"] = synthetic_event(5)
    with pytest.raises(ValueError, match="held"):
        compare_on_heldout(train, "feb_2022", synthetic_event(99), epochs=5)


def test_correction_is_not_enabled_when_there_is_no_signal():
    rep = compare_on_heldout(storms(signal=False), "feb_2022", synthetic_event(99, signal=False),
                             epochs=80, seed=0)
    assert not rep["qnn_beats_uncorrected"]
    corr = DensityCorrector.from_report(build_hybrid_qnn(seed=0), rep)
    assert corr.enabled is False and corr.log_correction([300, 130, 50, 10, 12, 20]) == 0.0


def test_corrector_defaults_to_disabled_and_needs_a_passing_report():
    model = build_hybrid_qnn(seed=0)
    assert DensityCorrector(model).enabled is False
    fake = {"qnn_beats_uncorrected": True, "heldout_event": "feb_2022", "train_events": ["a"],
            "uncorrected_rmse": 0.2, "qnn_rmse": 0.195}
    assert DensityCorrector.from_report(model, fake).enabled is False        # 2.5 % better: margin not met
    fake["qnn_rmse"] = 0.1
    assert DensityCorrector.from_report(model, fake).enabled is True
    assert DensityCorrector.from_report(model, {**fake, "heldout_event": None}).enabled is False
    assert DensityCorrector.from_report(model, {**fake, "train_events": ["feb_2022"]}).enabled is False


# --- the tool -------------------------------------------------------------------
ARGS = dict(altitude_km=210.0, f107=130.0, ap=60.0, hours_since_onset=12.0,
            local_solar_time_h=14.0, lat_band_deg=30.0)


def call_tool(corrector):
    reg = ToolRegistry()
    tool = QnnDensityCorrection(corrector, base_density_fn=lambda alt, t: 2.0e-10)
    assert isinstance(tool, Tool) and tool.name == "qnn_density_correction"
    reg.register(tool)
    return reg.call(ToolCall(name="qnn_density_correction", arguments=ARGS))


def test_disabled_tool_returns_the_uncorrected_density_and_says_why():
    out = call_tool(DensityCorrector(build_hybrid_qnn(seed=0))).output
    assert out["enabled"] is False and out["log_correction"] == 0.0
    assert out["corrected_density"] == out["base_density"] == 2.0e-10
    assert "held-out" in out["reason"]


def test_enabled_tool_applies_the_correction():
    class Fixed(DensityCorrector):
        def __init__(self):
            super().__init__(build_hybrid_qnn(seed=0), enabled=True)

        def log_correction(self, features):
            return math.log(1.3)

    out = call_tool(Fixed()).output
    assert out["enabled"] is True
    assert out["corrected_density"] == pytest.approx(2.0e-10 * 1.3)


def test_tool_rejects_missing_features():
    reg = ToolRegistry()
    reg.register(QnnDensityCorrection(DensityCorrector(build_hybrid_qnn(seed=0)),
                                      base_density_fn=lambda a, t: 1e-10))
    assert not reg.call(ToolCall(name="qnn_density_correction", arguments={"altitude_km": 200})).ok
