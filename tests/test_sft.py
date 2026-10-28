"""Phase 6 session 1: SFT traces, loss mask, resumable training, evaluation hook, notebook hygiene."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from helioguard.env import HelioGuardEnv
from helioguard.harness_env import ALLOWED_TYPES, EXTRA_ACTIONS, compact_observation
from helioguard.sft_traces import (OPTIMAL, THRESHOLD, action_text, data_card, generate_traces, record_trace,
                                   train_tasks)
from helioguard.tasks import load_tasks
from verispace_engine.harness import RolloutConfig, ScriptedBackend, parse_action, rollout_task
from verispace_engine.sft import IGNORE, SFTConfig, build_labels, load_config, tokenize_rows, train

ROOT = Path(__file__).parent.parent
PARSER_SHA256 = "3a54310a3eeb45b1f1ef0f4a0d6974864ec21a730277507be648fc55aa8f39ff"   # Phase 5 parser, LF-normalised
TOOLS = {"simulate_decay", "check_feasibility", "min_safe_altitude", "qaoa_schedule", "qnn_density_correction"}


@pytest.fixture(scope="module")
def tasks():
    return train_tasks()


@pytest.fixture(scope="module")
def sample(tasks):
    sub = tasks[:10]
    kept, everything = generate_traces(sub)
    return sub, kept, everything


def assistant_turns(trace):
    return [m["content"] for m in trace["messages"] if m["role"] == "assistant"]


# --- 1. every assistant turn passes the STRICT parser -----------------------------------------------
def test_every_assistant_turn_round_trips_through_the_strict_parser(sample):
    _, kept, everything = sample
    assert everything
    for trace in everything:
        turns = assistant_turns(trace)
        assert turns[-1].startswith('{"type":"submit_decision"')
        for text in turns:
            action = parse_action(text, ALLOWED_TYPES, TOOLS)
            assert action_text(action) == text                        # canonical, nothing else in the turn


def test_messages_alternate_system_user_then_assistant_user_pairs(sample):
    for trace in sample[2]:
        roles = [m["role"] for m in trace["messages"]]
        assert roles[:2] == ["system", "user"] and roles[-1] == "assistant"
        assert all(r == ("assistant" if i % 2 == 0 else "user") for i, r in enumerate(roles[2:]))


# --- 2. determinism: replaying a trace's calls reproduces its rewards -------------------------------
def test_replaying_a_trace_through_the_environment_reproduces_its_rewards(sample):
    sub, _, everything = sample
    env = HelioGuardEnv(sub, max_tool_calls=10, max_turns=20)
    for trace in everything:
        env.reset(task_id=trace["task_id"])
        rewards, final = [], None
        for text in assistant_turns(trace):
            out = env.step(parse_action(text, ALLOWED_TYPES, TOOLS))
            rewards.append(out.reward)
            final = out.info.get("final")
        assert rewards == trace["step_rewards"]
        assert final["final_reward"] == trace["final_reward"]


def test_the_trace_is_what_the_harness_shows_a_model(sample):
    """Feed the trace's own replies through the rollout runner: the model-visible messages must match."""
    sub, _, everything = sample
    trace = everything[0]
    env = HelioGuardEnv(sub, max_tool_calls=10, max_turns=20)
    seen = []

    def replay(messages):
        seen.append([dict(m) for m in messages])
        return trace["messages"][len(messages)]["content"]

    cfg = RolloutConfig(attempts=1, allowed_types=tuple(ALLOWED_TYPES), extra_actions=tuple(EXTRA_ACTIONS),
                        compactor=compact_observation)
    result = rollout_task(env, trace["task_id"], ScriptedBackend(replay), cfg)
    assert result.valid_submission and result.final_reward == trace["final_reward"]
    for messages in seen:
        assert messages == trace["messages"][: len(messages)]


def test_the_expert_decides_from_the_compacted_observations_the_model_sees(sample):
    _, _, everything = sample
    for trace in everything:
        for m in trace["messages"]:
            if m["role"] == "user" and m["content"].startswith("Environment result:"):
                assert "altitudes_km" not in m["content"] and "times_h" not in m["content"]


def test_compaction_shortens_observations_and_keeps_decision_fields(tasks):
    env = HelioGuardEnv(tasks[:2])
    env.reset(task_id=tasks[0]["id"])
    asset = tasks[0]["fleet"][0]["id"]
    out = env.step({"type": "tool_call", "name": "simulate_decay",
                    "arguments": {"asset_id": asset, "action": "hold", "hours": 48.0}})
    raw, small = out.observation, compact_observation(out.observation)
    assert len(json.dumps(small)) < len(json.dumps(raw)) / 2
    for k in ("final_alt_km", "reentered", "reentry_hour"):
        assert (round(raw["result"][k], 3) if isinstance(raw["result"][k], float) else raw["result"][k]) == small["result"][k]


# --- 3. no train / eval overlap -----------------------------------------------------------------------
def test_no_task_or_storm_is_shared_between_train_and_eval(tasks):
    ev = load_tasks(ROOT / "tasks" / "eval.jsonl")
    assert not {t["id"] for t in tasks} & {t["id"] for t in ev}
    assert not {t["storm"]["id"] for t in tasks} & {t["storm"]["id"] for t in ev}
    assert {t["split"] for t in tasks} == {"train"} and {t["split"] for t in ev} == {"eval"}


def test_trace_generation_refuses_eval_tasks(tasks):
    ev = load_tasks(ROOT / "tasks" / "eval.jsonl")
    with pytest.raises(ValueError):
        generate_traces(ev[:2])
    with pytest.raises(ValueError):
        generate_traces(tasks[:2] + ev[:1])


def test_sample_traces_are_train_only(sample):
    ev_storms = {t["storm"]["id"] for t in load_tasks(ROOT / "tasks" / "eval.jsonl")}
    for trace in sample[2]:
        assert trace["split"] == "train" and trace["storm_id"] not in ev_storms


def test_selection_threshold_and_card_counts(sample):
    _, kept, everything = sample
    assert all(t["valid"] and t["final_reward"] >= THRESHOLD for t in kept)
    card = data_card(kept, everything, THRESHOLD, seed=0)
    assert card["traces_kept"] == len(kept) and card["tasks_in"] == len(everything)
    assert card["kept_optimal"] + card["kept_non_optimal"] == len(kept)
    assert card["kept_optimal"] == sum(t["final_reward"] >= OPTIMAL for t in kept)
    assert card["seed"] == 0 and "disruption_mix_kept" in card and "histogram" in card["kept_final_reward"]
    assert sum(card["kept_final_reward"]["histogram"].values()) == len(kept)


# --- 4. loss mask: assistant tokens only -------------------------------------------------------------
class CharTok:
    """Qwen-style chat rendering over single-character tokens: enough to test the mask logic."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        ids = [ord(c) for c in text]
        return {"input_ids": ids} if return_dict else ids

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


CHAT = [{"role": "system", "content": "SYS rules"}, {"role": "user", "content": "briefing text"},
        {"role": "assistant", "content": '{"type":"tool_call","name":"x","arguments":{}}'},
        {"role": "user", "content": "Environment result: {}"},
        {"role": "assistant", "content": '{"type":"submit_decision","assets":{}}'}]


def test_loss_mask_covers_only_assistant_turns_with_a_fake_tokenizer():
    tok = CharTok()
    ids, labels = build_labels(CHAT, tok)
    assert len(ids) == len(labels)
    assert all(l in (IGNORE, i) for l, i in zip(labels, ids))
    trained = tok.decode([l for l in labels if l != IGNORE])
    assert trained == "".join(m["content"] + "<|im_end|>" for m in CHAT if m["role"] == "assistant")
    full = tok.decode(ids)
    for m in CHAT:                                                  # nothing outside the assistant turns is trained
        if m["role"] != "assistant":
            start = full.index(m["content"])
            assert all(l == IGNORE for l in labels[start: start + len(m["content"])])


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer is not cached locally")


def test_loss_mask_on_real_traces_with_the_qwen_tokenizer(sample, qwen_tokenizer):
    for trace in sample[1][:4]:
        ids, labels = build_labels(trace["messages"], qwen_tokenizer)
        trained = qwen_tokenizer.decode([l for l in labels if l != IGNORE])
        assert trained == "".join(m["content"] + "<|im_end|>" for m in trace["messages"] if m["role"] == "assistant")
        n_assistant = sum(m["role"] == "assistant" for m in trace["messages"])
        assert sum(l != IGNORE for l in labels) < len(ids) / 3          # most of the sequence is context
        assert labels[0] == IGNORE and labels[len(ids) // 3] == IGNORE   # system prompt and briefing are never trained
        assert n_assistant == trace["tool_calls"] + 1


def test_traces_fit_the_context_window(sample, qwen_tokenizer):
    for trace in sample[1]:
        assert len(build_labels(trace["messages"], qwen_tokenizer)[0]) <= 4096


# --- 5. the strict parser is unchanged ---------------------------------------------------------------
def test_strict_parser_source_is_unchanged_since_phase_5():
    src = (ROOT / "verispace_engine" / "harness" / "parser.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == PARSER_SHA256


# --- training: resumable, tiny model (fast) -----------------------------------------------------------
def tiny_build(cfg):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(cfg.seed)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=2048))
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"], modules_to_save=["lm_head"],
                                                       task_type="CAUSAL_LM"))
    return model, CharTok(), "cpu"


def synthetic_rows(n):
    return [{"messages": [{"role": "system", "content": f"rules {i}"}, {"role": "user", "content": f"brief {i}"},
                          {"role": "assistant", "content": '{"type":"tool_call","name":"t","arguments":{"i":%d}}' % i},
                          {"role": "user", "content": "Environment result: ok"},
                          {"role": "assistant", "content": '{"type":"submit_decision","assets":{}}'}]}
            for i in range(n)]


def tiny_cfg(tmp_path, name, **kw):
    base = dict(model="tiny", out_dir=str(tmp_path / name), checkpoint_dir=str(tmp_path / name / "ckpt"),
                max_seq_len=1024, lr=5e-3, epochs=1.0, max_steps=6, grad_accum=2, warmup_steps=1, seed=3,
                checkpoint_minutes=999, trainer="peft")
    base.update(kw)
    return SFTConfig(**base)


def test_resuming_gives_the_same_run_as_never_stopping(tmp_path):
    torch = pytest.importorskip("torch")
    rows = synthetic_rows(24)
    straight = train(tiny_cfg(tmp_path, "straight"), rows, tiny_build)
    first = train(tiny_cfg(tmp_path, "resumed"), rows, tiny_build, stop_after_steps=3)
    assert first["steps_done"] == 3 and not first["finished"]
    second = train(tiny_cfg(tmp_path, "resumed"), rows, tiny_build)
    assert second["resumed_from"] == 3 and second["finished"] and second["steps_done"] == 6
    log_a = [json.loads(line) for line in (tmp_path / "straight" / "train_log.jsonl").read_text().splitlines()]
    log_b = [json.loads(line) for line in (tmp_path / "resumed" / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log_b] == [1, 2, 3, 4, 5, 6]
    assert [r["loss"] for r in log_a] == pytest.approx([r["loss"] for r in log_b], rel=1e-5, abs=1e-6)
    from safetensors.torch import load_file
    wa = load_file(str(tmp_path / "straight" / "adapter" / "adapter_model.safetensors"))
    wb = load_file(str(tmp_path / "resumed" / "adapter" / "adapter_model.safetensors"))
    assert wa.keys() == wb.keys() and all(torch.allclose(wa[k], wb[k], atol=1e-6) for k in wa)


def test_training_reduces_loss_on_a_learnable_set(tmp_path):
    pytest.importorskip("torch")
    s = train(tiny_cfg(tmp_path, "learn", max_steps=40, lr=1e-2), synthetic_rows(16), tiny_build)
    assert s["final_loss"] < s["first_loss"] * 0.8


def test_a_checkpoint_from_another_config_is_refused(tmp_path):
    pytest.importorskip("torch")
    rows = synthetic_rows(24)
    train(tiny_cfg(tmp_path, "x"), rows, tiny_build, stop_after_steps=2)
    with pytest.raises(ValueError):
        train(tiny_cfg(tmp_path, "x", lr=1e-3), rows, tiny_build)


def test_too_long_examples_are_skipped_not_truncated():
    ex, skipped = tokenize_rows(synthetic_rows(3) + [{"messages": CHAT[:2] + [
        {"role": "assistant", "content": "x" * 5000}]}], CharTok(), 1024)
    assert len(ex) == 3 and skipped == 1


def test_engine_sft_imports_without_torch():
    code = ("import sys\nsys.modules['torch'] = None\nsys.modules['peft'] = None\nsys.modules['transformers'] = None\n"
            "import verispace_engine.sft as s\nassert s.build_labels\n")
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert run.returncode == 0, run.stderr


def test_configs_load_and_differ_only_where_intended():
    smoke = load_config(ROOT / "configs" / "sft_smoke_0p5b.json")
    real = load_config(ROOT / "configs" / "sft_1p5b.json")
    assert "0.5B" in smoke.model and "1.5B" in real.model and smoke.max_steps == 10
    assert real.trainer == "peft" and smoke.trainer == "peft" and real.seed == 0 and real.lora_r >= smoke.lora_r and real.epochs >= 1


# --- evaluation hook ---------------------------------------------------------------------------------
def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    from verispace_engine.harness.stats import bootstrap_ci
    vals = [0.9, 1.0, 0.7, 0.95, 0.5, 1.0, 0.8, 0.85]
    a, b = bootstrap_ci(vals), bootstrap_ci(vals)
    assert a == b and a[1] < a[0] < a[2]
    assert bootstrap_ci([0.5] * 10)[1:] == (0.5, 0.5)


def _fake(task_id, reward, valid=True, tools=3, attempt=0):
    from verispace_engine.harness.rollout import EpisodeResult
    return EpisodeResult(task_id, attempt, reward, reward if valid else -0.5, valid, tools, 4,
                         None if valid else "malformed:extra_text")


def test_summarize_condition_reports_primary_and_breakdown():
    from helioguard.sft_eval import summarize_condition
    ev = load_tasks(ROOT / "tasks" / "eval.jsonl")[:8]
    results = [_fake(t["id"], 1.0 if i % 2 == 0 else 0.5, valid=i != 3) for i, t in enumerate(ev)]
    s = summarize_condition(results, ev)
    assert s["valid_submission_rate"]["mean"] == pytest.approx(7 / 8)
    assert s["percent_optimal"]["mean"] == pytest.approx(4 / 8)
    assert s["failure_types"] == {"malformed:extra_text": 1}
    kinds = {d["type"] for t in ev for d in t["disruptions"]}
    assert kinds <= set(s["per_disruption"])
    assert all(0 <= v["valid_submission_rate"] <= 1 for v in s["per_disruption"].values())


def test_seeds_of_a_task_are_averaged_before_bootstrapping():
    from helioguard.sft_eval import per_task
    r = [_fake("a", 1.0, attempt=0), _fake("a", 0.0, attempt=1), _fake("b", 0.5)]
    assert per_task(r, lambda x: x.final_reward) == {"a": 0.5, "b": 0.5}


def ci(mean, lo, hi):
    return {"mean": mean, "ci_low": lo, "ci_high": hi}


def greedy(valid, reward):
    return {"valid_submission_rate": ci(valid, valid, valid), "mean_final_reward": reward}


@pytest.mark.parametrize("valid,reward,a,b,c_word", [
    (0.0, ci(-0.5, -0.5, -0.5), False, False, "fails to beat"),
    (0.96, ci(0.80, 0.76, 0.84), True, True, "above"),
    (0.96, ci(0.78, 0.70, 0.86), True, False, "not distinguishable"),
    (0.90, ci(0.74, 0.60, 0.88), True, False, "not distinguishable"),
    (0.89, ci(0.95, 0.90, 0.99), False, True, "above"),
    (1.00, ci(0.60, 0.50, 0.70), True, False, "fails to beat"),
])
def test_gates_follow_the_preregistered_rules(valid, reward, a, b, c_word):
    from helioguard.sft_eval import gate_report
    g = gate_report(greedy(valid, reward))
    assert g["gate_A_format"]["pass"] is a and g["gate_B_value"]["pass"] is b
    assert c_word in g["gate_C_honesty"]


def test_gate_document_exists_and_states_the_thresholds():
    text = (ROOT / "docs" / "phase6_gates.md").read_text(encoding="utf-8")
    for needle in ("Gate A", "Gate B", "Gate C", "90 percent", "+0.743", "+0.727", "bootstrap"):
        assert needle in text


# --- notebook ---------------------------------------------------------------------------------------
def test_colab_notebook_is_valid_clean_and_has_no_secrets():
    raw = (ROOT / "notebooks" / "phase6_sft_colab.ipynb").read_text(encoding="utf-8")
    nb = json.loads(raw)
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ("userdata.get", "drive.mount", "train_sft.py", "generate_sft_traces.py", "sft_eval"):
        assert needle in code, needle
    assert sum("resume" in "".join(c["source"]).lower() for c in nb["cells"]) >= 1
    for pattern in ("ghp_", "github_pat_", "hf_", "sk-", "Bearer "):
        assert pattern not in raw
    assert "https://github.com/nithinraj49/helioguard" in raw


# --- slow: the CPU smoke test (pytest -m slow) -----------------------------------------------------
@pytest.mark.slow
def test_cpu_smoke_run_10_steps_on_the_0p5b_model(tmp_path, tasks):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    kept, _ = generate_traces(tasks[:30])
    from helioguard.sft_traces import write_jsonl
    write_jsonl(tmp_path / "t.jsonl", kept)
    cfg = load_config(ROOT / "configs" / "sft_smoke_0p5b.json", data=str(tmp_path / "t.jsonl"),
                      out_dir=str(tmp_path / "out"), checkpoint_dir=str(tmp_path / "ckpt"))
    s = train(cfg)
    assert s["finished"] and s["steps_done"] == 10
    log = [json.loads(line) for line in (tmp_path / "out" / "train_log.jsonl").read_text().splitlines()]
    assert len(log) == 10 and all(0 < r["loss"] < 50 and r["grad_norm"] > 0 for r in log)
    assert (tmp_path / "out" / "adapter" / "adapter_model.safetensors").exists()


# --- Amendment 1: secondary analyses and recorded run facts -----------------------------------------
def test_paired_difference_uses_the_same_tasks_and_can_disagree_with_unpaired():
    from helioguard.sft_eval import gate_report, paired_vs_baseline
    from verispace_engine.harness.stats import bootstrap_ci
    ids = [f"t{i}" for i in range(20)]
    base = [_fake(t, 0.30 + 0.035 * i) for i, t in enumerate(ids)]            # wide spread
    model = [_fake(t, 0.35 + 0.035 * i) for i, t in enumerate(ids)]            # +0.05 on every task
    p = paired_vs_baseline(model, base)
    assert p["tasks"] == 20 and p["mean_difference"] == pytest.approx(0.05)
    assert p["ci_low"] == pytest.approx(0.05) and p["ci_excludes_zero_above"] is True
    mean, lo, hi = bootstrap_ci([r.final_reward for r in model])
    assert lo < 0.743 < hi                                                  # unpaired CI cannot separate the means
    assert gate_report(greedy(1.0, ci(mean, lo, hi)))["gate_B_value"]["pass"] is False


def test_quiet_task_action_mix_is_descriptive_and_counts_only_undisrupted_tasks(tmp_path):
    from helioguard.sft_eval import quiet_task_action_mix
    ev = load_tasks(ROOT / "tasks" / "eval.jsonl")
    quiet = [t for t in ev if not t["disruptions"]]
    assert quiet
    t = quiet[0]
    plan = {a["id"]: "safe_mode_edge_on" for a in t["fleet"]}
    log = tmp_path / "g.jsonl"
    log.write_text(json.dumps({"event": "turn", "task_id": t["id"], "attempt": 0,
                               "action": {"type": "submit_decision", "assets": plan}}) + "\n", encoding="utf-8")
    out = quiet_task_action_mix(log, ev)
    assert out["quiet_tasks"] == len(quiet) and out["tasks_with_a_submission"] == 1
    assert out["model_actions"] == {"safe_mode_edge_on": len(plan)} and sum(out["optimum_actions"].values()) > 0


def test_training_summary_records_trainer_and_seed(tmp_path):
    pytest.importorskip("torch")
    s = train(tiny_cfg(tmp_path, "rec", max_steps=2), synthetic_rows(8), tiny_build)
    assert s["training_seed"] == 3 and "trainer" in s


def test_scripted_baselines_are_identical_under_the_compact_view():
    """Amendment 1: baselines re-run under the compact view; the tool-informed expert decides the same."""
    from helioguard.harness_env import baseline_result, load
    env, ids = load("eval", 12)
    from helioguard.policies import ToolInformedPolicy, run_episode
    for tid in ids:
        raw = run_episode(env, ToolInformedPolicy(), tid)["final"]["final_reward"]
        assert baseline_result(env, "tool-informed", tid).final_reward == pytest.approx(raw)


def test_amendment_is_in_the_gate_document_with_no_threshold_changed():
    text = (ROOT / "docs" / "phase6_gates.md").read_text(encoding="utf-8")
    assert "Amendment 1" in text and "No threshold changed" in text
    for needle in ("90 percent", "+0.743", "+0.727", "10,000 resamples"):    # original numbers still present
        assert needle in text


def test_trainer_field_is_validated_and_never_falls_back_silently(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.sft import _build_model
    with pytest.raises(ValueError):
        _build_model(SFTConfig(trainer="deepspeed"))
    with pytest.raises(RuntimeError):                       # unsloth on this CPU-only or unsloth-less machine
        _build_model(SFTConfig(trainer="unsloth"))


# --- session 1d: provenance, config note, notebook cells, amendment 3 ------------------------------------
def test_train_summary_records_the_git_commit(tmp_path):
    pytest.importorskip("torch")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()
    s = train(tiny_cfg(tmp_path, "git", max_steps=2), synthetic_rows(8), tiny_build)
    assert s["git_commit"] == head and len(head) == 40
    assert isinstance(s["git_dirty"], bool)
    on_disk = json.loads((tmp_path / "git" / "train_summary.json").read_text())
    assert on_disk["git_commit"] == head


def test_git_commit_is_none_outside_a_repository(tmp_path, monkeypatch):
    from verispace_engine.sft import git_state
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_state() == (None, None)


def test_real_config_note_records_the_registered_trainer():
    raw = json.loads((ROOT / "configs" / "sft_1p5b.json").read_text(encoding="utf-8"))
    assert raw["trainer"] == "peft" and "T4 smoke test" in raw["_note"]
    for needle in ("20 steps", "no NaN", "3,595", "54 s", "resumed_from 11", "identical"):
        assert needle in raw["_note"], needle


def _nb():
    nb = json.loads((ROOT / "notebooks" / "phase6_sft_colab.ipynb").read_text(encoding="utf-8"))
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_starts_with_a_fresh_clone_and_cds_into_the_repo():
    _, code = _nb()
    first = code[0]
    assert "rm -rf /content/helioguard" in first or "shutil.rmtree" in first
    assert first.index("git") < first.index("%cd /content/helioguard")
    assert "%cd /content/helioguard" in first


def test_notebook_sets_allocator_and_pins_versions_and_prints_them():
    _, code = _nb()
    train_cell = next(c for c in code if "train_sft.py" in c)
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in train_cell
    install = next(c for c in code if "pip" in c and "install" in c)
    for pkg in ("peft==", "transformers==", "accelerate==", "torchao>=0.16.0"):
        assert pkg in install, pkg
    check = next(c for c in code if "importlib.metadata" in c or "__version__" in c)
    for pkg in ("peft", "transformers", "accelerate", "torchao", "torch"):
        assert pkg in check


def test_notebook_copies_traces_from_drive_or_regenerates_and_saves_them():
    _, code = _nb()
    cell = next(c for c in code if "train_traces.jsonl" in c and "generate_sft_traces.py" in c)
    assert "os.path.exists" in cell and "copy" in cell and "/content/drive" in cell
    assert cell.index("copy") < cell.index("generate_sft_traces.py")


def test_gitignore_excludes_smoke_and_base_outputs():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "results/phase6/sft_smoke_0p5b/" in lines


def test_amendment_3_registers_the_run_and_changes_no_threshold():
    text = (ROOT / "docs" / "phase6_gates.md").read_text(encoding="utf-8")
    assert "## Amendment 3" in text
    a3 = text[text.index("## Amendment 3"):]
    for needle in ("No threshold changed", "Qwen2.5-1.5B-Instruct", "trainer `peft`", "seed 0", "2 epochs",
                   "200 steps", "final checkpoint", "without `--greedy-only`", "resumed_from 11", "54 s"):
        assert needle in a3, needle
