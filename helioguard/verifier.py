"""Verifiable Process Rewards (VPR) for space-weather decisions.

The LLM produces a decision trace: a list of JSON steps. Each step is checked by a
deterministic physics verifier V(s, a) -> {0, 1}, giving dense turn-level rewards.
These rewards feed GVPO training (group-centred rewards across sampled traces).
"""
from .atmosphere import kp_to_g_scale
from .physics import simulate, can_raise

ACTIONS = {
    # action: (attitude, thrusting)
    "raise_orbit": ("raise", True),
    "safe_mode_edge_on": ("edge_on", False),
    "hold": ("raise", False),
}


def _outcome(sc, scen, action):
    attitude, thrusting = ACTIONS[action]
    return simulate(sc, scen["alt_km"], attitude, scen["storm_multiplier"],
                    scen["horizon_h"], thrusting=thrusting)


def _score(out):
    """Higher is better: survive first; if lost, survive longer (more time to recover)."""
    if out["reentered"]:
        return (0, out["reentry_hour"])
    return (1, out["final_alt_km"])


def feasible(sc, scen, action):
    if action == "raise_orbit":
        return can_raise(sc, scen["alt_km"], "raise", scen["storm_multiplier"])
    return action in ACTIONS


def best_action(sc, scen):
    options = [a for a in ACTIONS if feasible(sc, scen, a)]
    return max(options, key=lambda a: _score(_outcome(sc, scen, a)))


def verify_step(step, sc, scen, tol=0.25):
    """Return (reward, reason) for one reasoning step."""
    kind = step.get("type")

    if kind == "classify_storm":
        truth = kp_to_g_scale(scen["kp"])
        ok = step.get("g_scale") == truth
        return int(ok), f"storm level {step.get('g_scale')} vs truth {truth}"

    if kind == "predict_decay":
        attitude = step["attitude"]
        out = simulate(sc, scen["alt_km"], attitude, scen["storm_multiplier"], step["hours"])
        true_loss = scen["alt_km"] - out["final_alt_km"]
        pred = step["altitude_loss_km"]
        ok = abs(pred - true_loss) <= tol * max(true_loss, 1e-6)
        return int(ok), f"predicted loss {pred:.1f} km vs physics {true_loss:.1f} km"

    if kind == "check_feasibility":
        truth = feasible(sc, scen, step["action"])
        ok = step.get("feasible") == truth
        return int(ok), f"'{step['action']}' feasible={step.get('feasible')} vs physics {truth}"

    if kind == "choose_action":
        act = step["action"]
        if not feasible(sc, scen, act):
            return 0, f"'{act}' is physically infeasible under this storm"
        best = best_action(sc, scen)
        return int(act == best), f"chose '{act}', physics-optimal is '{best}'"

    return 0, f"unknown step type '{kind}'"


def verify_trace(trace, sc, scen):
    results = [verify_step(s, sc, scen) for s in trace]
    rewards = [r for r, _ in results]
    return {"step_rewards": rewards, "reasons": [m for _, m in results],
            "process_reward": sum(rewards) / len(rewards) if rewards else 0.0}


def group_centered(rewards):
    """Group-relative rewards (r - mean) for a group of traces on the same prompt,
    the reward term GVPO weights its gradient by."""
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]
