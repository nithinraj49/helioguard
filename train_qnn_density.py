"""Train the QNN density correction on training storms; test on held-out storms.

Roles come from the events file (configs/qnn_events.json): `heldout_test` (May 2024) is the
primary held-out storm, `quiet_check` (Feb 2022, Starlink Group 4-7) is a quiet-time do-no-harm
check across the ballistic band, `train` storms are trained on and also used for
leave-one-storm-out. Needs Space-Track credentials in .env. Writes reports/qnn_density_report.json.
The correction is enabled only if every gate passes (see helioguard/tools/density_pipeline.py).

    python train_qnn_density.py configs/qnn_events.json
"""
import json
import sys
from datetime import date, datetime
from pathlib import Path

from helioguard.data.noaa import fetch_space_weather
from helioguard.data.spacetrack import (MissingCredentials, SpaceTrackClient, SpaceTrackError,
                                        credentials_from_env)
from helioguard.replays import CACHE_DIR
from helioguard.tools.density_pipeline import EventSpec, run_density_pipeline

LIMITATIONS = [
    "Altitude gap: training objects are at about 280-430 km; Group 4-7 was at about 160-280 km.",
    "Rocket bodies tumble, so their ballistic coefficient is uncertain and may change over time.",
    "Feb 2022 element sets start on 5 Feb, after the storm: it can only test quiet-time behaviour.",
    "Jan 2026 training data is recent and may be incomplete or revised.",
    "Ballistic coefficients absorb any NRLMSIS density bias; targets are noisy.",
    "The Feb 2022 ballistic band rests on 2 intervals of 1 object plus an assumed 15 percent.",
    "The Feb 2022 quiet check is dominated by runaway integrations of objects near re-entry.",
]

if len(sys.argv) != 2:
    sys.exit(__doc__)
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    client = SpaceTrackClient(credentials_from_env(), CACHE_DIR / "spacetrack")
    events = []
    for e in cfg["events"]:
        ids = list(e.get("norad_ids") or [])
        if not ids and e.get("launch_intdes"):
            ids = [int(o["NORAD_CAT_ID"]) for o in client.launch_objects(e["launch_intdes"])
                   if o.get("OBJECT_TYPE") == "PAYLOAD"]
        if not ids:
            sys.exit(f"event {e['name']}: no satellites (set norad_ids or launch_intdes)")
        band = tuple(e["ballistic_band"]) if e.get("ballistic_band") else None
        events.append(EventSpec(e["name"], date.fromisoformat(e["start"]), date.fromisoformat(e["end"]),
                                datetime.fromisoformat(e["onset"]), ids, e.get("ballistic"),
                                e["role"], band))
    result = run_density_pipeline(client, events, fetch_space_weather(CACHE_DIR))
except (MissingCredentials, SpaceTrackError, ValueError, RuntimeError) as exc:
    sys.exit(f"NOT run: {exc}")

report = {**result["report"], "tool_enabled": result["corrector"].enabled,
          "limitations": LIMITATIONS, "skipped_objects": result["skipped"]}
out = Path("reports")
out.mkdir(exist_ok=True)
(out / "qnn_density_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def row(label, s):
    return (f"{label:28s} n={s['n_intervals']:4d}  uncorrected {s['uncorrected_mae_km']:.3f} km | "
            f"QNN {s['qnn_mae_km']:.3f} ({s['qnn_improvement']:+.1%}) | "
            f"MLP {s['mlp_mae_km']:.3f} ({s['mlp_improvement']:+.1%})")


g = report["gates"]
print(f"seeds {report['seeds']}, epochs {report['epochs']}, params QNN {report['qnn_params']} / MLP {report['mlp_params']}")
print(f"overlap: {report['overlap']['n_unique_objects_excluded']} unique objects "
      f"({report['overlap']['n_object_event_pairs_excluded']} object-storm pairs) excluded from training; "
      f"samples {report['n_train_samples_before_exclusion']} -> {report['n_train_samples']}")
for name, s in report["heldout"].items():
    print(row(f"HELD-OUT {name}", s), "PASS" if g["heldout"][name]["pass"] else "FAIL")
for f in report["loso"]["folds"]:
    print(row(f"LOSO test {f['test_event']}", f))
print(f"LOSO mean improvement: QNN {report['loso']['mean_qnn_improvement']:+.1%}, "
      f"MLP {report['loso']['mean_mlp_improvement']:+.1%}", "PASS" if g["loso_pass"] else "FAIL")
for label, s in report["quiet_check"]["by_ballistic"].items():
    print(row(f"QUIET feb_2022 B={s['ballistic']:.3f} ({label})", s),
          ("no worse" if g["quiet"][label]["no_worse"] else "WORSE"))
print("quiet gate (both ends):", "PASS" if g["quiet_pass"] else "FAIL")
print("\nqnn_density_correction ENABLED" if report["tool_enabled"] else "\nqnn_density_correction stays DISABLED")
print("skipped objects:", result["skipped"] or "none")
