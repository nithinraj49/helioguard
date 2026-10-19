"""Phase 1 'done when': any tool or verifier rule plugs in through one interface,
and the engine contains nothing HelioGuard-specific."""
import pathlib

import pytest
from pydantic import ValidationError

from verispace_engine.env_base import EnvBase, StepOutcome
from verispace_engine.schema import DecisionTrace, ToolCall, ToolResult, TraceStep
from verispace_engine.tools import Tool, ToolRegistry
from verispace_engine.verifier_base import VerifierBase, group_centered


class Add:
    name = "add"
    description = "Add two numbers"
    schema = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
              "required": ["a", "b"]}

    def run(self, a, b):
        return {"sum": a + b}


class Boom:
    name = "boom"
    description = "Always fails"
    schema = {"type": "object", "properties": {}, "required": []}

    def run(self):
        raise RuntimeError("kaboom")


# --- tools -------------------------------------------------------------------
def test_tool_protocol_is_structural():
    assert isinstance(Add(), Tool)
    assert not isinstance(object(), Tool)


def test_registry_register_and_call():
    reg = ToolRegistry()
    reg.register(Add())
    assert reg.names() == ["add"]
    res = reg.call(ToolCall(name="add", arguments={"a": 1, "b": 2}))
    assert res == ToolResult(name="add", ok=True, output={"sum": 3})
    spec = reg.specs()[0]
    assert spec["name"] == "add" and spec["parameters"]["required"] == ["a", "b"]


def test_registry_rejects_duplicates():
    reg = ToolRegistry()
    reg.register(Add())
    with pytest.raises(ValueError):
        reg.register(Add())


def test_registry_unknown_tool_and_bad_arguments_fail_cleanly():
    reg = ToolRegistry()
    reg.register(Add())
    reg.register(Boom())
    assert not reg.call(ToolCall(name="nope", arguments={})).ok
    missing = reg.call(ToolCall(name="add", arguments={"a": 1}))
    assert not missing.ok and "b" in missing.error
    extra = reg.call(ToolCall(name="add", arguments={"a": 1, "b": 2, "c": 3}))
    assert not extra.ok
    wrong_type = reg.call(ToolCall(name="add", arguments={"a": "x", "b": 2}))
    assert not wrong_type.ok
    crashed = reg.call(ToolCall(name="boom", arguments={}))
    assert not crashed.ok and "kaboom" in crashed.error


# --- schema ------------------------------------------------------------------
def test_trace_step_keeps_extra_fields_and_requires_type():
    step = TraceStep.model_validate({"type": "classify_storm", "g_scale": "G1"})
    assert step.type == "classify_storm" and step.model_dump()["g_scale"] == "G1"
    with pytest.raises(ValidationError):
        TraceStep.model_validate({"g_scale": "G1"})


def test_decision_trace_roundtrip():
    raw = [{"type": "a", "x": 1}, {"type": "b"}]
    trace = DecisionTrace.from_steps(raw)
    assert trace.as_dicts() == raw


def test_tool_call_requires_name():
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"arguments": {}})


# --- verifier ----------------------------------------------------------------
class ToyVerifier(VerifierBase):
    def __init__(self):
        super().__init__()
        self.add_rule("eq", self._eq)

    @staticmethod
    def _eq(step, ctx):
        return step["value"] == ctx["truth"], f"{step['value']} vs {ctx['truth']}"


def test_verifier_scores_steps_and_episode():
    v = ToyVerifier()
    ctx = {"truth": 3}
    assert v.score_step({"type": "eq", "value": 3}, ctx).reward == 1
    assert v.score_step({"type": "eq", "value": 4}, ctx).reward == 0
    ep = v.score_episode([{"type": "eq", "value": 3}, {"type": "eq", "value": 4}], ctx)
    assert ep.step_rewards == [1, 0] and ep.process_reward == 0.5
    assert len(ep.reasons) == 2


def test_verifier_loopholes_unknown_malformed_and_empty_fail():
    v = ToyVerifier()
    ctx = {"truth": 3}
    assert v.score_step({"type": "made_up"}, ctx).reward == 0
    assert v.score_step({}, ctx).reward == 0
    assert v.score_step({"type": "eq"}, ctx).reward == 0          # missing field
    assert v.score_episode([], ctx).process_reward == 0.0         # empty trace earns nothing


def test_verifier_rule_registration_is_the_only_extension_point():
    v = ToyVerifier()
    with pytest.raises(ValueError):
        v.add_rule("eq", lambda s, c: (True, ""))                 # no silent override


def test_group_centered():
    assert group_centered([0.0, 1.0]) == [-0.5, 0.5]


# --- env ---------------------------------------------------------------------
class CountEnv(EnvBase):
    def reset(self):
        self.n = 0
        return "start"

    def step(self, action):
        self.n += 1
        return StepOutcome(observation=f"n={self.n}", reward=0.0, done=self.n >= 2)


def test_env_base_contract():
    env = CountEnv()
    assert env.reset() == "start"
    assert not env.step("x").done
    assert env.step("x").done
    with pytest.raises(TypeError):
        EnvBase()                                                 # abstract


# --- independence ------------------------------------------------------------
def test_engine_has_no_helioguard_code():
    root = pathlib.Path(__file__).parent.parent / "verispace_engine"
    files = list(root.glob("*.py"))
    assert files
    for f in files:
        text = f.read_text(encoding="utf-8").lower()
        assert "helioguard" not in text, f.name
