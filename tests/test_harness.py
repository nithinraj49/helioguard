"""Phase 5: strict parser, prompts, backends, rollouts, logging and the eval command."""
import json
from pathlib import Path

import pytest

from helioguard import harness_env
from helioguard.env import HelioGuardEnv
from helioguard.tasks import load_tasks, oracle_plan
from verispace_engine.harness import (ActionParseError, OpenAICompatibleBackend, RolloutConfig, ScriptedBackend,
                                      parse_action, rollout_task, run_rollouts, system_prompt)
from verispace_engine.harness import eval as harness_eval
from verispace_engine.harness.backends import load_dotenv_value
from verispace_engine.harness.rollout import JsonlLog, summarize

TYPES = harness_env.ALLOWED_TYPES
TOOLS = {"simulate_decay", "check_feasibility"}


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(Path("tasks") / "eval.jsonl")[:6]


@pytest.fixture()
def env(tasks):
    return HelioGuardEnv(tasks, max_tool_calls=10, max_turns=20)


def kind(text):
    with pytest.raises(ActionParseError) as e:
        parse_action(text, TYPES, TOOLS)
    return e.value.kind


# --- parser --------------------------------------------------------------------------------
def test_parser_accepts_a_bare_object_and_one_fenced_object():
    call = '{"type": "tool_call", "name": "simulate_decay", "arguments": {"asset_id": "A", "hours": 5}}'
    assert parse_action(call, TYPES, TOOLS)["name"] == "simulate_decay"
    assert parse_action(f"  ```json\n{call}\n```  ", TYPES, TOOLS)["type"] == "tool_call"


@pytest.mark.parametrize("text,expected", [
    ("", "empty"), ("   \n", "empty"),
    ("I will call a tool now", "no_json"),
    ('Sure! {"type": "submit_decision", "assets": {}}', "no_json"),
    ('{"type": "submit_decision", "assets": {}} and that is my answer', "extra_text"),
    ('{"type": "tool_call", "name": "check_feasibility"}{"type": "submit_decision", "assets": {}}', "extra_text"),
    ('[{"type": "submit_decision"}]', "not_object"),
    ('{"name": "simulate_decay"}', "missing_type"),
    ('{"type": "launch_missiles"}', "unknown_type"),
    ('{"type": "tool_call", "arguments": {}}', "bad_tool_call"),
    ('{"type": "tool_call", "name": "simulate_decay", "arguments": [1]}', "bad_tool_call"),
    ('{"type": "tool_call", "name": "not_a_tool", "arguments": {}}', "bad_tool_call"),
    ('{"type": "submit_decision", "assets": "all raise"}', "bad_submit"),
    ('{"type": "submit_decision"}', "bad_submit"),
    ('{"type": "tool_call", "name": "simulate_decay", "arguments": {', "no_json"),
])
def test_parser_rejects_malformed_replies_without_repairing_them(text, expected):
    assert kind(text) == expected


# --- prompts -------------------------------------------------------------------------------
def test_system_prompt_lists_every_tool_with_its_schema(env, tasks):
    env.reset(task_id=tasks[0]["id"])
    text = system_prompt(env.tool_specs(), harness_env.EXTRA_ACTIONS)
    for spec in env.tool_specs():
        assert spec["name"] in text
    assert "arguments schema" in text and "ONE JSON object" in text and "submit_decision" in text


# --- backends ------------------------------------------------------------------------------
class _Resp:
    def __init__(self, code, payload):
        self.status_code, self._p = code, payload

    def json(self):
        return self._p


def test_openai_backend_posts_chat_request_and_returns_text(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-secret-123")
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, body=json, headers=headers)
        return _Resp(200, {"choices": [{"message": {"content": '{"type": "submit_decision", "assets": {}}'}}]})

    b = OpenAICompatibleBackend("m", base_url="http://x/v1/", key_var="TEST_KEY", post=fake_post)
    out = b.generate([{"role": "user", "content": "hi"}], 50, 0.0, 3)
    assert "submit_decision" in out and seen["url"] == "http://x/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-secret-123" and seen["body"]["seed"] == 3


def test_openai_backend_error_does_not_leak_the_key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-secret-123")
    b = OpenAICompatibleBackend("m", key_var="TEST_KEY",
                                post=lambda *a, **k: _Resp(401, {"error": "bad sk-secret-123"}))
    with pytest.raises(RuntimeError) as e:
        b.generate([{"role": "user", "content": "hi"}])
    assert "sk-secret-123" not in str(e.value)


def test_key_is_read_from_dotenv_and_dotenv_is_ignored_by_git(tmp_path, monkeypatch):
    monkeypatch.delenv("DOT_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text('OTHER=1\nDOT_KEY="abc"\n', encoding="utf-8")
    assert load_dotenv_value("DOT_KEY", f) == "abc"
    assert ".env" in Path(".gitignore").read_text().split()


# --- rollouts ------------------------------------------------------------------------------
def submit_text(plan):
    return json.dumps({"type": "submit_decision", **plan})


def test_a_correct_submission_scores_the_exact_optimum_and_is_logged(env, tasks, tmp_path):
    t = tasks[0]
    backend = ScriptedBackend([json.dumps({"type": "tool_call", "name": "min_safe_altitude", "arguments": {}}),
                               submit_text(oracle_plan(t))])
    log_path = tmp_path / "run.jsonl"
    r = rollout_task(env, t["id"], backend, RolloutConfig(attempts=1), 0, JsonlLog(log_path))
    assert r.valid_submission and r.failure is None and r.final_reward == pytest.approx(1.0)
    assert r.tool_calls == 1 and r.turns == 2
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    turns = [x for x in lines if x["event"] == "turn"]
    assert [x["turn"] for x in turns] == [1, 2]
    assert turns[0]["action"]["name"] == "min_safe_altitude" and "reward" in turns[0] and "observation" in turns[0]
    assert turns[0]["raw_output"]
    assert lines[-1]["event"] == "episode" and lines[-1]["final"]["final_reward"] == pytest.approx(1.0)


@pytest.mark.parametrize("reply,failure", [("I think we should raise everything.", "malformed:no_json"),
                                           ('{"type": "submit_decision", "assets": {}} done', "malformed:extra_text")])
def test_malformed_output_fails_the_episode_at_once(env, tasks, reply, failure):
    r = rollout_task(env, tasks[0]["id"], ScriptedBackend([reply]), RolloutConfig(attempts=1))
    assert r.failure == failure and not r.valid_submission and r.turns == 1
    assert r.final_reward == -0.5 and r.total_return == -0.5


def test_endless_tool_calls_hit_the_env_caps_and_fail(env, tasks):
    t = tasks[0]
    counter = iter(range(1, 1000))

    def always_a_new_call(messages):
        h = next(counter)
        return json.dumps({"type": "tool_call", "name": "simulate_decay",
                           "arguments": {"asset_id": t["fleet"][0]["id"], "action": "hold", "hours": h % 90 + 1}})

    r = rollout_task(env, t["id"], ScriptedBackend(always_a_new_call), RolloutConfig(attempts=1))
    assert r.failure == "never_submits" and not r.valid_submission
    assert r.tool_calls == 10 and r.turns == 20 and r.final_reward == -0.5


def test_an_unplayable_submission_is_a_rejected_failure(env, tasks):
    bad = submit_text({"assets": {"no_such_asset": "raise_orbit"}})
    r = rollout_task(env, tasks[0]["id"], ScriptedBackend([bad]), RolloutConfig(attempts=1))
    assert r.failure == "rejected_submission" and not r.valid_submission


def test_n_attempts_per_task_and_summary(env, tasks):
    reply = submit_text(oracle_plan(tasks[0]))
    ids = [tasks[0]["id"]]
    results = run_rollouts(env, ids, ScriptedBackend(lambda m: reply), RolloutConfig(attempts=3), progress=False)
    assert [r.attempt for r in results] == [0, 1, 2]
    s = summarize(results + [rollout_task(env, ids[0], ScriptedBackend(["nope"]), RolloutConfig())])
    assert s["episodes"] == 4 and s["valid_submission_rate"] == 0.75
    assert s["failure_types"] == {"malformed:no_json": 1}


def test_the_model_never_sees_the_stored_optimum(env, tasks):
    seen = []

    def spy(messages):
        seen.append(json.dumps(messages))
        return "stop"

    rollout_task(env, tasks[0]["id"], ScriptedBackend(spy), RolloutConfig())
    assert all(f'\\"{k}\\"' not in seen[0] and f'"{k}"' not in seen[0] for k in ("optimal", "baselines", "nominal"))


# --- the one command -----------------------------------------------------------------------
def test_eval_command_runs_a_baseline_and_writes_log_and_summary(tmp_path):
    s = harness_eval.main(["--model", "baseline:tool-informed", "--split", "eval", "--n", "3", "--out", str(tmp_path)])
    assert s["episodes"] == 3 and s["valid_submission_rate"] == 1.0
    assert (tmp_path / "baseline_tool-informed_eval_summary.json").exists()
    assert len((tmp_path / "baseline_tool-informed_eval.jsonl").read_text().splitlines()) == 3
