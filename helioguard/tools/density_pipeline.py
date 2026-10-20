"""Train and test the QNN density correction on held-out storms, with a same-size MLP beside it.

Roles of an event:
  train         training storm; also one fold of leave-one-storm-out (train on the others)
  heldout_test  primary held-out storm; never trained on
  quiet_check   quiet-time "do no harm" check across a ballistic-coefficient band; never trained on

Objects that appear in a test storm are removed from training storms (and, in each
leave-one-storm-out fold, objects shared with the fold's test storm are removed from the other
storms). Decay-prediction error is the mean absolute error, in km, of the altitude change over
each element-set interval of at least 3 h, starting from the observed altitude.

Gates for enabling the correction (all must pass): at least 5 percent lower error than
uncorrected NRLMSIS on every held-out test storm AND on average across the leave-one-storm-out
folds, AND no worse than uncorrected on the quiet check at both ends of the ballistic band.
The MLP never affects the gates; it is reported next to every number.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Sequence

import numpy as np

from ..atmosphere_msis import get_density_fn
from ..data.celestrak import group_by_norad
from ..data.noaa import load_space_weather
from ..data.spacetrack import SpaceTrackClient
from .decay_density import (effective_density_samples, estimate_ballistic_coefficient,
                            evaluate_decay_prediction)
from .qnn_density import (DensityCorrector, build_classical_mlp, build_hybrid_qnn, count_parameters,
                          scale_features, train_model)

ROLES = ("train", "heldout_test", "quiet_check")
MIN_QUIET_ELEMENT_SETS = 3
MIN_INTERVAL_S = 3 * 3600.0           # shorter intervals are dominated by TLE fit noise
MIN_IMPROVEMENT = 0.05
MIN_TRAIN_SAMPLES = 20


@dataclass
class EventSpec:
    name: str
    start: date
    end: date
    onset: datetime                         # storm onset (UTC, timezone-aware)
    norad_ids: list[int]
    ballistic: float | None = None          # Cd*A/m, m^2/kg; None = estimate from pre-storm sets
    role: str = "train"
    ballistic_band: tuple[float, float] | None = None     # (low, high) for the quiet check
    skipped: dict[int, str] = field(default_factory=dict)


def relative_improvement(uncorrected: float, new: float) -> float:
    return (uncorrected - new) / uncorrected if uncorrected > 0 else 0.0


def evaluate_gates(heldout: dict[str, dict], loso_improvements: Sequence[float],
                   quiet: dict[str, dict], min_improvement: float = MIN_IMPROVEMENT) -> dict[str, Any]:
    """Apply the enabling gates to QNN numbers. `heldout[name]` and `quiet[label]` hold
    uncorrected_mae_km and qnn_mae_km; `quiet` needs "low" and "high" (the ends decide)."""
    tol = 1e-12
    held = {}
    for name, s in heldout.items():
        imp = relative_improvement(s["uncorrected_mae_km"], s["qnn_mae_km"])
        held[name] = {"improvement": imp, "pass": imp >= min_improvement - tol}
    heldout_pass = bool(held) and all(h["pass"] for h in held.values())
    mean_imp = statistics.fmean(loso_improvements) if loso_improvements else None
    loso_pass = mean_imp is not None and mean_imp >= min_improvement - tol
    q = {label: {"no_worse": s["qnn_mae_km"] <= s["uncorrected_mae_km"] + 1e-9} for label, s in quiet.items()}
    quiet_pass = all(label in q and q[label]["no_worse"] for label in ("low", "high"))
    return {"min_improvement": min_improvement, "heldout": held, "heldout_pass": heldout_pass,
            "loso": {"improvements": list(loso_improvements), "mean_improvement": mean_imp,
                     "pass": loso_pass},
            "loso_pass": loso_pass, "quiet": q, "quiet_pass": quiet_pass,
            "all_passed": bool(heldout_pass and loso_pass and quiet_pass)}


# --- data preparation -------------------------------------------------------------------------
def _prepare(client: SpaceTrackClient, spec: EventSpec, sw_path, sw_rows) -> dict[str, Any]:
    t0 = datetime(spec.start.year, spec.start.month, spec.start.day, tzinfo=timezone.utc)
    hours = (spec.end - spec.start).days * 24
    rho_fn, source = get_density_fn(sw_path, t0.replace(tzinfo=None), hours, storm_time=True)
    if source != "pymsis":
        raise RuntimeError("NRLMSIS density unavailable; refusing to build targets from the fallback")
    histories = group_by_norad(client.gp_history(spec.norad_ids, spec.start, spec.end))
    ballistic: dict[int, float] = {}
    for norad, tles in list(histories.items()):
        if len(tles) < 2:
            spec.skipped[norad] = "fewer than two element sets"
        elif spec.role == "quiet_check":
            continue                                    # B comes from the band, set at scoring time
        elif spec.ballistic is not None:
            ballistic[norad] = spec.ballistic
            continue
        else:
            quiet = [t for t in tles if t.epoch < spec.onset]
            if len(quiet) < MIN_QUIET_ELEMENT_SETS:
                spec.skipped[norad] = "too few pre-storm element sets to estimate the ballistic coefficient"
            else:
                try:
                    ballistic[norad] = estimate_ballistic_coefficient(quiet, rho_fn, t0)
                    continue
                except ValueError as exc:
                    spec.skipped[norad] = str(exc)
        histories.pop(norad, None)
    return {"spec": spec, "t0": t0, "rho_fn": rho_fn, "histories": histories, "ballistic": ballistic}


def _samples(prep: dict, sw_rows, exclude: frozenset[int] = frozenset()):
    X, y = [], []
    for norad, tles in prep["histories"].items():
        if norad in exclude:
            continue
        for s in effective_density_samples(tles, prep["ballistic"][norad], prep["rho_fn"], prep["t0"],
                                           sw_rows, prep["spec"].onset, min_interval_s=MIN_INTERVAL_S):
            X.append(s.features)
            y.append(s.target)
    return np.array(X).reshape(-1, 6), np.array(y)


def _train_pair(X: np.ndarray, y: np.ndarray, epochs: int, seed: int):
    if len(y) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"only {len(y)} training samples after excluding overlapping objects")
    Xs = scale_features(X)
    qnn, mlp = build_hybrid_qnn(seed), build_classical_mlp(seed)
    train_model(qnn, Xs, y, epochs=epochs, seed=seed)
    train_model(mlp, Xs, y, epochs=epochs, seed=seed)
    return qnn, mlp


def _score(prep: dict, sw_rows, models: Sequence[tuple], ballistic: dict[int, float] | None = None,
           decaying_only: bool = False) -> dict[str, Any]:
    b = ballistic if ballistic is not None else prep["ballistic"]
    hist = {k: v for k, v in prep["histories"].items() if k in b}
    args = (hist, b, prep["rho_fn"], prep["t0"], sw_rows, prep["spec"].onset)
    kw = {"min_interval_s": MIN_INTERVAL_S, "decaying_only": decaying_only}
    base = evaluate_decay_prediction(*args, **kw)
    qnn = [evaluate_decay_prediction(*args, corrector=DensityCorrector(q, enabled=True), **kw)["mae_km"]
           for q, _ in models]
    mlp = [evaluate_decay_prediction(*args, corrector=DensityCorrector(m, enabled=True), **kw)["mae_km"]
           for _, m in models]
    q_mean, m_mean = statistics.fmean(qnn), statistics.fmean(mlp)
    return {"n_intervals": base["n_intervals"], "n_objects": len(hist),
            "uncorrected_mae_km": base["mae_km"],
            "qnn_mae_km": q_mean, "qnn_mae_std": statistics.pstdev(qnn),
            "mlp_mae_km": m_mean, "mlp_mae_std": statistics.pstdev(mlp),
            "qnn_improvement": relative_improvement(base["mae_km"], q_mean),
            "mlp_improvement": relative_improvement(base["mae_km"], m_mean)}


# --- the run ------------------------------------------------------------------------------------
def run_density_pipeline(client: SpaceTrackClient, events: list[EventSpec], sw_path, epochs: int = 100,
                         seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    """Returns {"report", "corrector", "skipped"}. Test and quiet-check events are never trained on."""
    for e in events:
        if e.role not in ROLES:
            raise ValueError(f"event {e.name}: unknown role '{e.role}' (use one of {ROLES})")
    by_role = {r: [e for e in events if e.role == r] for r in ROLES}
    if not by_role["train"]:
        raise ValueError("need at least one training event")
    if not by_role["heldout_test"]:
        raise ValueError("need at least one held-out test event")
    if len(by_role["quiet_check"]) != 1:
        raise ValueError("need exactly one quiet-check event")
    quiet_spec = by_role["quiet_check"][0]
    if quiet_spec.ballistic is None or quiet_spec.ballistic_band is None:
        raise ValueError("the quiet check needs a central ballistic coefficient and a (low, high) band")

    sw_rows = load_space_weather(sw_path)
    prep = {e.name: _prepare(client, e, sw_path, sw_rows) for e in events}
    train_names = [e.name for e in by_role["train"]]
    test_names = [e.name for e in by_role["heldout_test"]] + [quiet_spec.name]

    # object overlap: anything tracked in a test storm is removed from the training storms
    test_ids = frozenset().union(*(prep[n]["histories"].keys() for n in test_names))
    excluded = {n: sorted(set(prep[n]["histories"]) & test_ids) for n in train_names}
    overlap = {"test_objects": len(test_ids), "excluded": excluded,
               "n_object_event_pairs_excluded": sum(len(v) for v in excluded.values()),
               "n_unique_objects_excluded": len({i for v in excluded.values() for i in v})}

    parts = {n: _samples(prep[n], sw_rows, frozenset(excluded[n])) for n in train_names}
    X = np.vstack([p[0] for p in parts.values()])
    y = np.concatenate([p[1] for p in parts.values()])
    n_before = sum(len(_samples(prep[n], sw_rows)[1]) for n in train_names)

    # main models: trained on all training storms (after exclusion), one pair per seed
    models = [_train_pair(X, y, epochs, s) for s in seeds]
    heldout = {n: _score(prep[n], sw_rows, models) for n in [e.name for e in by_role["heldout_test"]]}

    lo, hi = quiet_spec.ballistic_band
    quiet_by_b = {}
    for label, b in (("low", lo), ("central", quiet_spec.ballistic), ("high", hi)):
        score = _score(prep[quiet_spec.name], sw_rows, models,
                       ballistic={k: b for k in prep[quiet_spec.name]["histories"]}, decaying_only=True)
        quiet_by_b[label] = {"ballistic": b, **score}

    # leave-one-storm-out over the training storms
    folds = []
    for k in train_names:
        others = [n for n in train_names if n != k]
        k_ids = frozenset(prep[k]["histories"])
        ex = {n: sorted(set(prep[n]["histories"]) & k_ids) for n in others}
        px = [_samples(prep[n], sw_rows, frozenset(ex[n])) for n in others]
        fold_models = [_train_pair(np.vstack([p[0] for p in px]), np.concatenate([p[1] for p in px]),
                                   epochs, s) for s in seeds]
        folds.append({"test_event": k, "train_events": others, "overlap_excluded": ex,
                      "n_train_samples": int(sum(len(p[1]) for p in px)),
                      **_score(prep[k], sw_rows, fold_models)})
    loso = {"folds": folds,
            "mean_qnn_improvement": statistics.fmean(f["qnn_improvement"] for f in folds),
            "mean_mlp_improvement": statistics.fmean(f["mlp_improvement"] for f in folds)}

    gates = evaluate_gates(heldout, [f["qnn_improvement"] for f in folds], quiet_by_b)
    report = {"seeds": list(seeds), "epochs": epochs, "training_events": train_names,
              "heldout": heldout, "quiet_check": {"event": quiet_spec.name, "by_ballistic": quiet_by_b},
              "loso": loso, "overlap": overlap, "n_train_samples": int(len(y)),
              "n_train_samples_before_exclusion": int(n_before),
              "qnn_params": count_parameters(models[0][0]), "mlp_params": count_parameters(models[0][1]),
              "gates": gates}
    corrector = DensityCorrector(models[0][0], enabled=gates["all_passed"], report=report)
    return {"report": report, "corrector": corrector,
            "skipped": {e.name: dict(e.skipped) for e in events if e.skipped}}
