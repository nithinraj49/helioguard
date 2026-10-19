"""Replay the Feb 2022 Starlink storm and score two LLM decision traces with VPR."""
from helioguard.scenarios import STARLINK_LIKE as SC, FEB_2022, FEB_2022_QUIET
from helioguard.physics import simulate, drag_force
from helioguard.verifier import verify_trace, group_centered, ACTIONS

print("=== Physics: same satellite, same 210 km orbit, 48 h ===")
for scen in (FEB_2022_QUIET, FEB_2022):
    print(f"\n[{scen['id']}]  Kp={scen['kp']}  density x{scen['storm_multiplier']}")
    for act, (att, thr) in ACTIONS.items():
        out = simulate(SC, scen["alt_km"], att, scen["storm_multiplier"], 48, thrusting=thr)
        fate = (f"LOST at hour {out['reentry_hour']:.1f}" if out["reentered"]
                else f"survives, altitude {out['final_alt_km']:.0f} km")
        print(f"  {act:18s} drag {drag_force(210, SC.area_m2[att], SC.cd, scen['storm_multiplier'])*1e3:5.1f} mN -> {fate}")

# Two candidate traces an LLM might produce for the storm scenario.
naive = [
    {"type": "classify_storm", "g_scale": "G3"},
    {"type": "predict_decay", "attitude": "raise", "hours": 24, "altitude_loss_km": 10},
    {"type": "check_feasibility", "action": "raise_orbit", "feasible": True},
    {"type": "choose_action", "action": "raise_orbit"},
]
good = [
    {"type": "classify_storm", "g_scale": "G1"},
    {"type": "predict_decay", "attitude": "raise", "hours": 12, "altitude_loss_km": 40},
    {"type": "check_feasibility", "action": "raise_orbit", "feasible": False},
    {"type": "choose_action", "action": "safe_mode_edge_on"},
]

print("\n=== VPR: step-by-step verification of LLM traces (Feb 2022 storm) ===")
totals = []
for name, trace in (("naive", naive), ("physics-aware", good)):
    res = verify_trace(trace, SC, FEB_2022)
    totals.append(res["process_reward"])
    print(f"\n[{name} trace]  process reward = {res['process_reward']:.2f}")
    for r, why in zip(res["step_rewards"], res["reasons"]):
        print(f"  {'PASS' if r else 'FAIL'}  {why}")

print("\nGVPO group-centred rewards:", [round(x, 2) for x in group_centered(totals)])
