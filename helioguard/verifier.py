"""Verifiable Process Rewards (VPR) for space-weather decisions.

The LLM produces a decision trace: a list of JSON steps. Each step is checked by a
deterministic physics verifier V(s, a) -> {0, 1}, giving dense turn-level rewards.
These rewards feed GVPO training (group-centred rewards across sampled traces).
"""
from verispace_engine.verifier_base import VerifierBase, group_centered  # noqa: F401 (re-export)

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


class HelioGuardVerifier(VerifierBase):
    """VPR rules for storm decisions. Context is the pair (spacecraft, scenario)."""

    def __init__(self, tol: float = 0.25):
        super().__init__()
        self.tol = tol
        self.add_rule("classify_storm", self._classify_storm)
        self.add_rule("predict_decay", self._predict_decay)
        self.add_rule("check_feasibility", self._check_feasibility)
        self.add_rule("choose_action", self._choose_action)

    @staticmethod
    def _classify_storm(step, ctx):
        _, scen = ctx
        truth = kp_to_g_scale(scen["kp"])
        return step.get("g_scale") == truth, f"storm level {step.get('g_scale')} vs truth {truth}"

    def _predict_decay(self, step, ctx):
        sc, scen = ctx
        out = simulate(sc, scen["alt_km"], step["attitude"], scen["storm_multiplier"], step["hours"])
        true_loss = scen["alt_km"] - out["final_alt_km"]
        pred = step["altitude_loss_km"]
        ok = abs(pred - true_loss) <= self.tol * max(true_loss, 1e-6)
        return ok, f"predicted loss {pred:.1f} km vs physics {true_loss:.1f} km"

    @staticmethod
    def _check_feasibility(step, ctx):
        sc, scen = ctx
        truth = feasible(sc, scen, step["action"])
        return (step.get("feasible") == truth,
                f"'{step['action']}' feasible={step.get('feasible')} vs physics {truth}")

    @staticmethod
    def _choose_action(step, ctx):
        sc, scen = ctx
        act = step["action"]
        if not feasible(sc, scen, act):
            return False, f"'{act}' is physically infeasible under this storm"
        best = best_action(sc, scen)
        return act == best, f"chose '{act}', physics-optimal is '{best}'"


def verify_step(step, sc, scen, tol=0.25):
    """Return (reward, reason) for one reasoning step."""
    res = HelioGuardVerifier(tol).score_step(step, (sc, scen))
    return res.reward, res.reason


def verify_trace(trace, sc, scen):
    res = HelioGuardVerifier().score_episode(trace, (sc, scen))
    return {"step_rewards": res.step_rewards, "reasons": res.reasons,
            "process_reward": res.process_reward}
