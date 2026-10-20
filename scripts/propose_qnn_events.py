"""Draft configs/qnn_events.json for human review. Does NOT train anything.

Feb 2022 (Starlink Group 4-7) stays held out; May 2024 is a second, test-only storm. Training
storms are the strongest other storms (Kp >= 7) since 2016 that have enough passive tracking
objects: rocket bodies near 340-430 km in near-circular orbits, with enough element sets
before and after onset and no manoeuvre-like rise. Needs Space-Track credentials in .env.
Queries are batched and cached under .cache/spacetrack/ (git-ignored); the client spaces
requests out to respect Space-Track rate limits.
"""
import json
from datetime import date, timedelta
from pathlib import Path

from helioguard.data.celestrak import group_by_norad
from helioguard.data.noaa import fetch_space_weather, load_space_weather
from helioguard.data.spacetrack import SpaceTrackClient, credentials_from_env
from helioguard.replays import CACHE_DIR
from helioguard.tools.decay_density import _sma_m
from helioguard.tools.event_proposal import StormCandidate, build_config, find_storms, usable_objects
from helioguard.physics import R_EARTH

N_TRAIN, MAX_ATTEMPTS, MIN_OBJECTS = 5, 9, 4
EXCLUDE = [(date(2022, 2, 3), date(2022, 2, 12)), (date(2024, 5, 8), date(2024, 5, 14))]
BAND = "MEAN_MOTION/15.45--16.00/ECCENTRICITY/%3C0.005/OBJECT_TYPE/ROCKET%20BODY"

client = SpaceTrackClient(credentials_from_env(), CACHE_DIR / "spacetrack")
rows = load_space_weather(fetch_space_weather(CACHE_DIR))


def objects_for(storm: StormCandidate):
    """Discover candidate passive objects the day the window opens, then filter their histories."""
    path = (f"/basicspacedata/query/class/gp_history/EPOCH/{storm.start}--{storm.start + timedelta(days=1)}"
            f"/{BAND}/orderby/NORAD_CAT_ID%20asc/limit/200/format/json")
    found = sorted({int(r["NORAD_CAT_ID"]) for r in json.loads(client.query(path, "json"))})[:30]
    if not found:
        return [], {}
    hist = group_by_norad(client.gp_history(found, storm.start, storm.end))
    return usable_objects(hist, storm.onset), hist


def describe(storm, ids, hist):
    alts = [(_sma_m(hist[i][0]) - R_EARTH) / 1e3 for i in ids]
    sets = sum(len(hist[i]) for i in ids)
    return (f"{storm.name}  onset {storm.onset:%Y-%m-%d %H:%M}Z  Kp {storm.kp_max}  ap {storm.ap_max}  "
            f"{len(ids)} objects, {sets} element sets, alt {min(alts):.0f}-{max(alts):.0f} km")


training, notes = [], []
candidates = [s for s in find_storms(rows, 7.0, EXCLUDE, first_day=date(2016, 1, 1)) if s.onset.year != 2024 or s.onset.month != 5]
for storm in candidates[:MAX_ATTEMPTS]:
    if len(training) == N_TRAIN:
        break
    ids, hist = objects_for(storm)
    if len(ids) >= MIN_OBJECTS:
        training.append((storm, ids))
        notes.append("TRAIN " + describe(storm, ids, hist))
    else:
        notes.append(f"skip  {storm.name}  Kp {storm.kp_max}: only {len(ids)} usable passive objects")

may = next(s for s in find_storms(rows, 7.0, [], first_day=date(2024, 1, 1)) if (s.onset.year, s.onset.month) == (2024, 5))
may_ids, may_hist = objects_for(may)
notes.append("TEST  " + describe(may, may_ids, may_hist) if may_ids else f"TEST  {may.name}: NO usable passive objects")

cal = Path("reports/ballistic_calibration.json")
central, band = None, None
if cal.exists():
    c = json.loads(cal.read_text(encoding="utf-8"))
    central, band = c["central_m2_per_kg"], tuple(c["total_band_with_assumed_density_uncertainty"])
cfg = build_config(training, (may, may_ids), "2022-010", central, band)
Path("configs").mkdir(exist_ok=True)
Path("configs/qnn_events.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("\n".join(notes))
print("\nwrote configs/qnn_events.json (DRAFT, not used for training)")
