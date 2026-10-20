"""Train the QNN density correction on other storms; test on Feb 2022 (Starlink Group 4-7).

Needs Space-Track credentials in `.env` and an events file (see configs/qnn_events.example.json).
Writes reports/qnn_density_report.json. The correction is enabled only if it reproduces the
held-out decay better than uncorrected NRLMSIS. Not yet run on real data.

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

if len(sys.argv) != 2:
    sys.exit(__doc__)
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    client = SpaceTrackClient(credentials_from_env(), CACHE_DIR / "spacetrack")
    events = []
    for e in cfg["events"]:
        ids = e.get("norad_ids") or [
            int(o["NORAD_CAT_ID"]) for o in client.launch_objects(e["launch_intdes"])
            if o.get("OBJECT_TYPE") == "PAYLOAD"] if e.get("launch_intdes") else []
        if not ids:
            sys.exit(f"event {e['name']}: no satellites (set norad_ids or launch_intdes)")
        events.append(EventSpec(e["name"], date.fromisoformat(e["start"]), date.fromisoformat(e["end"]),
                                datetime.fromisoformat(e["onset"]), ids, e.get("ballistic")))
    result = run_density_pipeline(client, events, cfg["heldout"], fetch_space_weather(CACHE_DIR))
except (MissingCredentials, SpaceTrackError, ValueError, RuntimeError) as exc:
    sys.exit(f"NOT run: {exc}")

report = result["report"]
out = Path("reports")
out.mkdir(exist_ok=True)
(out / "qnn_density_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k: report[k] for k in (
    "heldout_event", "train_events", "uncorrected_mae_km", "qnn_mae_km", "mlp_mae_km",
    "qnn_params", "mlp_params", "qnn_decay_gate", "success_criterion_met")}, indent=2))
print("skipped satellites:", result["skipped"] or "none")
