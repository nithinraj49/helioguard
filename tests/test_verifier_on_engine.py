"""HelioGuard's verifier now sits on VerifierBase; results must not change."""
from helioguard.scenarios import FEB_2022, STARLINK_LIKE as SC
from helioguard.verifier import HelioGuardVerifier, verify_step, verify_trace
from verispace_engine.verifier_base import VerifierBase

from test_phase1_baseline import NAIVE, PHYSICS_AWARE


def test_helioguard_verifier_is_a_verifier_base():
    assert issubclass(HelioGuardVerifier, VerifierBase)


def test_traces_score_identically_through_engine():
    v = HelioGuardVerifier()
    ctx = (SC, FEB_2022)
    assert v.score_episode(NAIVE, ctx).step_rewards == [0, 0, 0, 0]
    assert v.score_episode(PHYSICS_AWARE, ctx).step_rewards == [1, 1, 1, 1]


def test_legacy_functions_match_engine():
    for trace in (NAIVE, PHYSICS_AWARE):
        legacy = verify_trace(trace, SC, FEB_2022)
        eng = HelioGuardVerifier().score_episode(trace, (SC, FEB_2022))
        assert legacy["step_rewards"] == eng.step_rewards
        assert legacy["reasons"] == eng.reasons
        assert legacy["process_reward"] == eng.process_reward


def test_loopholes_vague_and_malformed_steps_fail():
    assert verify_step({"type": "predict_decay", "attitude": "raise", "hours": 12,
                        "altitude_loss_km": 0}, SC, FEB_2022)[0] == 0
    assert verify_step({"type": "predict_decay", "attitude": "raise", "hours": 12}, SC, FEB_2022)[0] == 0
    assert verify_step({"type": "choose_action", "action": "hold"}, SC, FEB_2022)[0] == 0
    assert verify_step({"type": "choose_action", "action": "teleport"}, SC, FEB_2022)[0] == 0
    assert verify_step({"type": "classify_storm"}, SC, FEB_2022)[0] == 0
