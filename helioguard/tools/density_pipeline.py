"""Train the QNN density correction on other storms and test it on a held-out one (Feb 2022).

Success criterion: with the correction, NRLMSIS reproduces the observed decay of the held-out
satellites (Starlink Group 4-7) better than uncorrected NRLMSIS, by a real margin. Until a
run on real Space-Track histories passes this, the correction stays disabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from ..atmosphere_msis import get_density_fn
from ..data.celestrak import group_by_norad
from ..data.noaa import load_space_weather
from ..data.spacetrack import SpaceTrackClient
from .decay_density import (decay_gate, effective_density_samples, estimate_ballistic_coefficient,
                            evaluate_decay_prediction)
from .qnn_density import DensityCorrector, compare_on_heldout

MIN_QUIET_ELEMENT_SETS = 3


@dataclass
class EventSpec:
    name: str
    start: date
    end: date
    onset: datetime                         # storm onset (UTC, timezone-aware)
    norad_ids: list[int]
    ballistic: float | None = None          # Cd*A/m in m^2/kg; None = estimate from the quiet part
    skipped: dict[int, str] = field(default_factory=dict)


def _prepare(client: SpaceTrackClient, spec: EventSpec, sw_path, sw_rows) -> dict[str, Any]:
    t0 = datetime(spec.start.year, spec.start.month, spec.start.day, tzinfo=timezone.utc)
    hours = (spec.end - spec.start).days * 24
    rho_fn, source = get_density_fn(sw_path, t0.replace(tzinfo=None), hours, storm_time=True)
    if source != "pymsis":
        raise RuntimeError("NRLMSIS density unavailable; refusing to build targets from the fallback")
    histories = group_by_norad(client.gp_history(spec.norad_ids, spec.start, spec.end))
    ballistic: dict[int, float] = {}
    for norad, tles in list(histories.items()):
        if spec.ballistic is not None:
            ballistic[norad] = spec.ballistic
            continue
        quiet = [t for t in tles if t.epoch < spec.onset]
        if len(quiet) < MIN_QUIET_ELEMENT_SETS:
            spec.skipped[norad] = "too few pre-storm element sets to estimate the ballistic coefficient"
            del histories[norad]
            continue
        ballistic[norad] = estimate_ballistic_coefficient(quiet, rho_fn, t0)
    X, y = [], []
    for norad, tles in histories.items():
        for s in effective_density_samples(tles, ballistic[norad], rho_fn, t0, sw_rows, spec.onset):
            X.append(s.features)
            y.append(s.target)
    if not y:
        raise ValueError(f"event {spec.name}: no usable decay intervals")
    return {"X": np.array(X), "y": np.array(y), "histories": histories, "ballistic": ballistic,
            "rho_fn": rho_fn, "t0": t0, "spec": spec}


def run_density_pipeline(client: SpaceTrackClient, events: list[EventSpec], heldout_name: str,
                         sw_path, epochs: int = 80, seed: int = 0) -> dict[str, Any]:
    """Returns {"report", "corrector", "models", "skipped"}. The held-out event is never trained on."""
    if heldout_name not in {e.name for e in events}:
        raise ValueError(f"held-out event '{heldout_name}' is not among the events")
    sw_rows = load_space_weather(sw_path)
    prepared = {e.name: _prepare(client, e, sw_path, sw_rows) for e in events}
    held = prepared.pop(heldout_name)
    report, models = compare_on_heldout({n: (p["X"], p["y"]) for n, p in prepared.items()},
                                        heldout_name, (held["X"], held["y"]),
                                        epochs=epochs, seed=seed, return_models=True)
    args = (held["histories"], held["ballistic"], held["rho_fn"], held["t0"], sw_rows,
            held["spec"].onset)
    base = evaluate_decay_prediction(*args, corrector=None)
    qnn_corr = DensityCorrector(models["qnn"], enabled=True)
    qnn = evaluate_decay_prediction(*args, corrector=qnn_corr)
    mlp = evaluate_decay_prediction(*args, corrector=DensityCorrector(models["mlp"], enabled=True))
    corrector = DensityCorrector.from_report(models["qnn"], report)
    report.update({
        "uncorrected_mae_km": base["mae_km"], "qnn_mae_km": qnn["mae_km"], "mlp_mae_km": mlp["mae_km"],
        "n_decay_intervals": base["n_intervals"],
        "qnn_decay_gate": decay_gate(base["mae_km"], qnn["mae_km"]),
        "mlp_decay_gate": decay_gate(base["mae_km"], mlp["mae_km"]),
        "qnn_rmse_gate": corrector.enabled,
    })
    report["success_criterion_met"] = bool(report["qnn_decay_gate"] and report["qnn_rmse_gate"])
    corrector.enabled = report["success_criterion_met"]
    corrector.report = report
    return {"report": report, "corrector": corrector, "models": models,
            "skipped": {e.name: dict(e.skipped) for e in events if e.skipped}}
