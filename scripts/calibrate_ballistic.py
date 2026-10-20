"""Calibrate Cd*A/m (m^2/kg) for Starlink Group 4-7 from the element sets that exist.

Uses only intervals of at least 3 h (shorter ones are TLE fit noise) on objects in free
decay (no manoeuvre-like rise) and NRLMSIS density in storm-time
mode. CAVEAT: B is derived with NRLMSIS density, so a density bias is absorbed into B; the
reported band adds an ASSUMED 15 percent density uncertainty on top of the measured scatter.
Needs Space-Track credentials in .env. Writes reports/ballistic_calibration.json.
"""
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

from helioguard.atmosphere_msis import get_density_fn
from helioguard.data.celestrak import group_by_norad
from helioguard.data.noaa import fetch_space_weather, load_space_weather
from helioguard.data.spacetrack import SpaceTrackClient, credentials_from_env
from helioguard.replays import CACHE_DIR
from helioguard.tools.decay_density import _decay_rate, _pairs, interval_ballistic, summarize_ballistic
from helioguard.tools.event_proposal import MAX_RAISE_MS, MIN_INTERVAL_H

START, END = date(2022, 2, 3), date(2022, 2, 12)
DENSITY_UNCERTAINTY = 0.15          # ASSUMPTION, not measured

client = SpaceTrackClient(credentials_from_env(), CACHE_DIR / "spacetrack")
ids = [int(o["NORAD_CAT_ID"]) for o in client.launch_objects("2022-010") if o["OBJECT_TYPE"] == "PAYLOAD"]
hist = group_by_norad(client.gp_history(ids, START, END))
sw_path = fetch_space_weather(CACHE_DIR)
t0 = datetime(2022, 2, 3, tzinfo=timezone.utc)
rho_fn, source = get_density_fn(sw_path, t0.replace(tzinfo=None), (END - START).days * 24, storm_time=True)
assert source == "pymsis"
kp = {r.day: r.kp_3h for r in load_space_weather(sw_path)}

used, per_object, all_b, kp_seen = [], {}, [], []
for norad, tles in sorted(hist.items()):
    pairs = list(_pairs(tles)) if len(tles) > 1 else []
    rates = [r for r, _, dt in (_decay_rate(a, b) for a, b in pairs) if dt >= MIN_INTERVAL_H * 3600]
    vals = interval_ballistic(tles, rho_fn, t0, MIN_INTERVAL_H * 3600) if rates else []
    if not rates or any(r > MAX_RAISE_MS for r in rates) or len(vals) < 2:
        why = "no interval of at least 3 h" if not rates else (
            "manoeuvre-like rise (thrusting)" if any(r > MAX_RAISE_MS for r in rates) else "fewer than 2 decaying intervals")
        print(f"  skip {norad}: {why}  ({len(tles)} element sets)")
        continue
    used.append(norad)
    per_object[norad] = summarize_ballistic([b for _, b in vals])
    all_b += [b for _, b in vals]
    for mid, _ in vals:
        kp_seen.append(max(kp[mid.date()]))
    print(f"  use  {norad}: {len(vals)} intervals, median B = {per_object[norad]['central']:.4f} m^2/kg")

s = summarize_ballistic(all_b)
total_lo = s["p16"] * (1 - DENSITY_UNCERTAINTY)
total_hi = s["p84"] * (1 + DENSITY_UNCERTAINTY)
result = {"method": "free-decay intervals of Group 4-7 payloads, NRLMSIS storm-time density",
          "central_m2_per_kg": s["central"], "scatter_band_16_84": [s["p16"], s["p84"]],
          "total_band_with_assumed_density_uncertainty": [total_lo, total_hi],
          "assumed_density_uncertainty": DENSITY_UNCERTAINTY, "n_intervals": s["n"],
          "objects_used": used, "per_object_median": {str(k): v["central"] for k, v in per_object.items()},
          "max_kp_during_used_intervals": max(kp_seen) if kp_seen else None,
          "caveat": "B absorbs any NRLMSIS density bias; the element sets begin after the storm."}
Path("reports").mkdir(exist_ok=True)
Path("reports/ballistic_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"\nCd*A/m central {s['central']:.4f} m^2/kg from {s['n']} intervals on {len(used)} objects")
print(f"  scatter band (16-84 %):            {s['p16']:.4f} to {s['p84']:.4f}")
print(f"  with assumed 15 % density error:   {total_lo:.4f} to {total_hi:.4f}")
print(f"  max Kp over the intervals used:    {result['max_kp_during_used_intervals']}")
