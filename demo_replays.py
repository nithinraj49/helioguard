"""Replay the Feb 2022 and May 2024 storms with real space-weather indices (NRLMSIS via pymsis).

Downloads and caches SW-All on first run. Spacecraft and altitudes are ILLUSTRATIVE.
HelioGuard recommends; humans decide.
"""
from helioguard.physics import simulate
from helioguard.replays import FEB_2022_REAL, MAY_2024_REAL, build_replay
from helioguard.scenarios import STARLINK_LIKE as SC
from helioguard.verifier import ACTIONS, best_action

for spec in (FEB_2022_REAL, MAY_2024_REAL):
    scen = build_replay(spec)
    fn, alt = scen["density_fn"], scen["alt_km"]
    peak = max(fn(alt, h) for h in range(scen["horizon_h"] + 1)) / fn(alt, 0)
    print(f"\n[{scen['id']}]  start {scen['start']}  Kp max {scen['kp']} ({scen['g_scale']})  "
          f"ap max {scen['ap_max']}  data: {scen['data_source']}")
    print(f"  orbit-averaged density at {alt:.0f} km: peak is x{peak:.2f} of the start value")
    for act, (att, thr) in ACTIONS.items():
        out = simulate(SC, alt, att, 1.0, scen["horizon_h"], thrusting=thr, density_fn=fn)
        fate = (f"LOST at hour {out['reentry_hour']:.1f}" if out["reentered"]
                else f"survives, altitude {out['final_alt_km']:.0f} km")
        print(f"  {act:18s} -> {fate}")
    print(f"  physics-optimal action: {best_action(SC, scen)}")
