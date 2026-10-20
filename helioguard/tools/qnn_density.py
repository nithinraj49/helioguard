"""qnn_density_correction: hybrid quantum-classical correction to the NRLMSIS density.

Target: ln(rho_effective / rho_NRLMSIS), with rho_effective derived from orbit decay.
Model: AngleEmbedding + StronglyEntanglingLayers on 6 qubits (qml.qnn.TorchLayer) and a
linear head, compared with a classical MLP of about the same size on the same data.
No quantum advantage is claimed; whatever the comparison shows is reported.

The correction is OFF until a report shows it beating uncorrected NRLMSIS on a held-out
storm (the held-out event is never in the training set). Until then the tool returns the
uncorrected density, so the verifier never depends on an unproven model.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

N_QUBITS = 6
N_LAYERS = 3
MIN_IMPROVEMENT = 0.05                  # corrected RMSE must be >5 % below uncorrected
FEATURE_NAMES = ["altitude_km", "f107", "ap", "hours_since_onset", "local_solar_time_h",
                 "lat_band_deg"]
# (low, high) used to scale each raw feature into [0, pi] for angle embedding; clipped outside.
FEATURE_RANGES = [(150.0, 500.0), (60.0, 300.0), (0.0, 400.0), (0.0, 72.0), (0.0, 24.0), (0.0, 90.0)]


def scale_features(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    lo = np.array([r[0] for r in FEATURE_RANGES])
    hi = np.array([r[1] for r in FEATURE_RANGES])
    return np.clip((X - lo) / (hi - lo), 0.0, 1.0) * math.pi


def build_hybrid_qnn(seed: int = 0, device: str = "default.qubit") -> torch.nn.Module:
    """54 quantum + 7 classical parameters. `default.qubit` with backprop keeps training fast;
    pass device="lightning.qubit" to use the lightning simulator."""
    import pennylane as qml
    torch.manual_seed(seed)
    dev = qml.device(device, wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

    layer = qml.qnn.TorchLayer(circuit, {"weights": (N_LAYERS, N_QUBITS, 3)},
                               init_method={"weights": lambda t: torch.nn.init.normal_(t, std=0.1)})
    return torch.nn.Sequential(layer, torch.nn.Linear(N_QUBITS, 1))


def build_classical_mlp(seed: int = 0) -> torch.nn.Module:
    """6 -> 7 -> 1 with tanh: 57 parameters, within 7 % of the hybrid model's 61."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(N_QUBITS, 7), torch.nn.Tanh(), torch.nn.Linear(7, 1))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _tensors(X: np.ndarray, y: np.ndarray | None = None):
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    return Xt if y is None else (Xt, torch.as_tensor(np.asarray(y), dtype=torch.float32))


def train_model(model: torch.nn.Module, X_scaled: np.ndarray, y: np.ndarray, epochs: int = 60,
                lr: float = 0.05, seed: int = 0) -> list[float]:
    """Full-batch Adam on mean squared error. Returns the loss per epoch."""
    torch.manual_seed(seed)
    Xt, yt = _tensors(X_scaled, y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(Xt).squeeze(-1) - yt) ** 2)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def rmse(model: torch.nn.Module, X_scaled: np.ndarray, y: np.ndarray) -> float:
    Xt, yt = _tensors(X_scaled, y)
    with torch.no_grad():
        return float(torch.sqrt(torch.mean((model(Xt).squeeze(-1) - yt) ** 2)))


def compare_on_heldout(train_events: dict[str, tuple[np.ndarray, np.ndarray]], heldout_name: str,
                       heldout_data: tuple[np.ndarray, np.ndarray], epochs: int = 80,
                       lr: float = 0.05, seed: int = 0, return_models: bool = False) -> dict[str, Any]:
    """Train the hybrid QNN and the same-size MLP on `train_events`, test on the held-out storm.

    `uncorrected_rmse` is the error of predicting no correction (ln ratio 0): plain NRLMSIS."""
    if heldout_name in train_events:
        raise ValueError(f"event '{heldout_name}' is held out and must not be in the training events")
    X = np.vstack([e[0] for e in train_events.values()])
    y = np.concatenate([e[1] for e in train_events.values()])
    Xs, (Xh, yh) = scale_features(X), (scale_features(heldout_data[0]), np.asarray(heldout_data[1]))

    models: dict[str, torch.nn.Module] = {}
    report: dict[str, Any] = {"heldout_event": heldout_name, "train_events": sorted(train_events),
                              "n_train": int(len(y)), "n_heldout": int(len(yh)),
                              "uncorrected_rmse": float(np.sqrt(np.mean(yh ** 2)))}
    for key, build in (("qnn", build_hybrid_qnn), ("mlp", build_classical_mlp)):
        model = build(seed)
        t0 = time.perf_counter()
        train_model(model, Xs, y, epochs=epochs, lr=lr, seed=seed)
        report[f"{key}_train_s"] = time.perf_counter() - t0
        report[f"{key}_rmse"] = rmse(model, Xh, yh)
        report[f"{key}_params"] = count_parameters(model)
        models[key] = model
    report["qnn_beats_uncorrected"] = report["qnn_rmse"] < report["uncorrected_rmse"]
    report["mlp_beats_uncorrected"] = report["mlp_rmse"] < report["uncorrected_rmse"]
    report["qnn_beats_mlp"] = report["qnn_rmse"] < report["mlp_rmse"]
    return (report, models) if return_models else report


class DensityCorrector:
    """Wraps a trained model. Disabled unless built from a passing held-out report."""

    def __init__(self, model: torch.nn.Module, enabled: bool = False, report: dict | None = None):
        self.model, self.enabled, self.report = model, enabled, report

    @classmethod
    def from_report(cls, model: torch.nn.Module, report: dict,
                    min_improvement: float = MIN_IMPROVEMENT) -> "DensityCorrector":
        held = report.get("heldout_event")
        ok = (bool(report.get("qnn_beats_uncorrected")) and bool(held)
              and held not in report.get("train_events", [])
              and report["uncorrected_rmse"] > 0
              and (report["uncorrected_rmse"] - report["qnn_rmse"]) / report["uncorrected_rmse"]
              > min_improvement)
        return cls(model, enabled=ok, report=report)

    def log_correction(self, features: Sequence[float]) -> float:
        if not self.enabled:
            return 0.0
        Xt = _tensors(scale_features(np.asarray(features, dtype=float)))
        with torch.no_grad():
            return float(self.model(Xt).squeeze())


class QnnDensityCorrection:
    name = "qnn_density_correction"
    description = ("Corrected neutral density for the given conditions: NRLMSIS density times "
                   "exp(learned log correction). Returns the uncorrected density, with the "
                   "reason, until the model has beaten NRLMSIS on a held-out storm.")
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {n: {"type": "number"} for n in FEATURE_NAMES},
        "required": list(FEATURE_NAMES),
    }

    def __init__(self, corrector: DensityCorrector, base_density_fn: Callable[[float, float], float]):
        """`base_density_fn(altitude_km, hours_since_onset)` gives the NRLMSIS density (kg/m^3)."""
        self.corrector, self.base_density_fn = corrector, base_density_fn

    def run(self, **features: float) -> dict[str, Any]:
        raw = [features[n] for n in FEATURE_NAMES]
        base = self.base_density_fn(features["altitude_km"], features["hours_since_onset"])
        if not self.corrector.enabled:
            return {"enabled": False, "base_density": base, "log_correction": 0.0,
                    "corrected_density": base,
                    "reason": "Correction not enabled: it has not yet beaten uncorrected NRLMSIS "
                              "on a held-out storm."}
        c = self.corrector.log_correction(raw)
        rep = self.corrector.report or {}
        return {"enabled": True, "base_density": base, "log_correction": c,
                "corrected_density": base * math.exp(c),
                "reason": f"Validated on held-out storm {rep.get('heldout_event', '?')}."}
