"""Phase 6 session 2: GVPO (centred rewards, reference via the disabled adapter, zero-sum weights, replay off),
the GRPO baseline, group rollout equivalence, resumable loop with verified and mirrored checkpoints, precheck, gates.
Tests came first; docs/phase6_gvpo_gates.md was committed before any of this code."""
import json
import math
import random
from pathlib import Path

import pytest

from test_robust_saving import hub
from test_sft import CharTok, tiny_build

ROOT = Path(__file__).parent.parent


# --- pure maths (no torch) -----------------------------------------------------------------------------
def test_centred_rewards_are_never_divided_by_the_group_std():
    from verispace_engine.gvpo import centred, gvpo_weights
    r = [1.0, 0.0, 0.0, 1.0]
    big = [10.0 * x for x in r]
    assert centred(r) == pytest.approx([0.5, -0.5, -0.5, 0.5])
    assert centred(big) == pytest.approx([5.0, -5.0, -5.0, 5.0])         # scale is kept: no std normalisation
    w1, w2 = gvpo_weights(r, [0.0] * 4, [0.0] * 4, beta=0.1), gvpo_weights(big, [0.0] * 4, [0.0] * 4, beta=0.1)
    assert w2 == pytest.approx([10 * x for x in w1])


def test_weights_follow_the_gvpo_formula_and_sum_to_zero():
    from verispace_engine.gvpo import gvpo_weights
    rng = random.Random(1)
    for _ in range(50):
        k = rng.randint(2, 9)
        r = [rng.uniform(-1, 1) for _ in range(k)]
        lp, ref = [rng.uniform(-80, -20) for _ in range(k)], [rng.uniform(-80, -20) for _ in range(k)]
        beta = rng.choice([0.01, 0.1, 1.0])
        w = gvpo_weights(r, lp, ref, beta)
        d = [a - b for a, b in zip(lp, ref)]
        mr, md = sum(r) / k, sum(d) / k
        assert w == pytest.approx([(r[i] - mr) - beta * (d[i] - md) for i in range(k)])
        assert abs(sum(w)) < 1e-9


def test_a_non_zero_sum_is_refused():
    from verispace_engine.gvpo import assert_zero_sum
    assert_zero_sum([0.3, -0.1, -0.2])
    with pytest.raises(AssertionError, match="sum to zero"):
        assert_zero_sum([0.3, 0.1, -0.2])


def test_a_group_with_identical_rewards_has_zero_weights_when_policy_equals_reference():
    from verispace_engine.gvpo import gvpo_weights
    assert gvpo_weights([-0.5] * 8, [-30.0, -31, -29, -30, -32, -28, -30, -30],
                        [-30.0, -31, -29, -30, -32, -28, -30, -30], beta=0.1) == [0.0] * 8


def test_grpo_divides_by_the_group_std_and_gives_zero_for_identical_rewards():
    from verispace_engine.gvpo import grpo_advantages
    a = grpo_advantages([1.0, 0.0, 0.0, 1.0])
    assert a == pytest.approx([1.0, -1.0, -1.0, 1.0], rel=1e-3)
    assert grpo_advantages([10.0, 0.0, 0.0, 10.0]) == pytest.approx(a, rel=1e-3)      # scale-free, unlike GVPO
    assert grpo_advantages([0.3] * 5) == [0.0] * 5


def test_lr_schedule_is_warmup_then_constant():
    from verispace_engine.gvpo import GVPOConfig, lr_at
    cfg = GVPOConfig(lr=5e-5, warmup_steps=3)
    assert [lr_at(cfg, s) for s in range(5)] == pytest.approx([5e-5 / 3, 10e-5 / 3, 5e-5, 5e-5, 5e-5])


def test_config_validation_replay_is_off_and_names_are_checked():
    from verispace_engine.gvpo import GVPOConfig
    assert GVPOConfig().replay is False
    for bad in ({"replay": True}, {"algo": "ppo"}, {"reward": "vibes"}, {"group_size": 1}):
        with pytest.raises(ValueError):
            GVPOConfig(**bad)


def test_engine_gvpo_imports_without_torch():
    import subprocess
    import sys
    code = ("import sys; sys.modules['torch']=None; sys.modules['transformers']=None; sys.modules['peft']=None; "
            "import verispace_engine.gvpo, verispace_engine.harness.group_rollout")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr


# --- the model-level pieces (tiny model) ------------------------------------------------------------------
def make_episode(i, reward=None, n_tools=2):
    from verispace_engine.gvpo import Episode
    msgs = [{"role": "system", "content": "rules"}, {"role": "user", "content": f"brief {i}"}]
    for k in range(n_tools):
        msgs += [{"role": "assistant", "content": '{"type":"tool_call","name":"t%d","arguments":{}}' % (i + k)},
                 {"role": "user", "content": "Environment result: ok"}]
    msgs.append({"role": "assistant", "content": '{"type":"submit_decision","assets":{"b%d":"raise_orbit"}}' % i})
    r = float(i % 5) / 4 if reward is None else reward
    return Episode(messages=msgs, final_reward=r, total_return=r + 0.03 * n_tools, valid=True)


def tiny_cfg(tmp_path, name, **kw):
    from verispace_engine.gvpo import GVPOConfig
    base = dict(model="tiny", out_dir=str(tmp_path / name), checkpoint_dir=str(tmp_path / name / "ckpt"),
                group_size=4, tasks_per_step=2, steps=6, lr=5e-3, warmup_steps=1, beta=0.1, seed=3,
                checkpoint_minutes=999, max_seq_len=2048, dtype="float32", gradient_checkpointing=False)
    base.update(kw)
    return GVPOConfig(**base)


def fake_sampler(calls=None):
    def sample(model, tok, device, task_id, group_size, seed, cfg):
        if calls is not None:
            calls.append((task_id, seed))
        rng = random.Random(f"{task_id}:{seed}")
        return [make_episode(rng.randint(0, 40), reward=rng.choice([0.0, 0.5, 1.0]), n_tools=rng.randint(1, 3))
                for _ in range(group_size)]
    return sample


TASKS = [f"t{i}" for i in range(12)]


def test_reference_is_the_model_with_the_adapter_disabled():
    torch = pytest.importorskip("torch")
    from verispace_engine.gvpo import episode_tensors, ref_logprob, sequence_logprob
    from verispace_engine.sft import IGNORE
    model, tok, dev = tiny_build(SimpleCfg())
    ids, labels = episode_tensors(make_episode(1).messages, tok, dev)
    pol0 = sequence_logprob(model, ids, labels, grad=False)[0]
    ref0 = ref_logprob(model, ids, labels)
    assert float(pol0) == pytest.approx(float(ref0), abs=1e-4)             # fresh adapter: policy == reference exactly
    with torch.no_grad():                                                   # move the adapter
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(0.5)
    pol1 = sequence_logprob(model, ids, labels, grad=False)[0]
    ref1 = ref_logprob(model, ids, labels)
    assert float(ref1) == pytest.approx(float(ref0), abs=1e-4)             # the reference does not move with the adapter
    assert abs(float(pol1) - float(ref1)) > 1e-3
    assert (labels != IGNORE).sum() > 0


class SimpleCfg:
    seed = 3


def test_logprob_covers_only_assistant_tokens():
    torch = pytest.importorskip("torch")
    from verispace_engine.gvpo import episode_tensors, sequence_logprob
    model, tok, dev = tiny_build(SimpleCfg())
    ep = make_episode(2)
    ids, labels = episode_tensors(ep.messages, tok, dev)
    total, per_tok = sequence_logprob(model, ids, labels, grad=False)
    n_assistant_tokens = sum(len(m["content"]) + len("<|im_end|>") for m in ep.messages if m["role"] == "assistant")
    assert per_tok.numel() == n_assistant_tokens
    assert float(total) == pytest.approx(float(per_tok.sum()), rel=1e-5)
    assert float(total) < 0


def test_two_pass_gradient_equals_autograd_of_the_gvpo_loss():
    """The implementation backpropagates -beta * w_i * logp_i episode by episode; that must equal the gradient of
    mean_i 0.5 * (c_i - beta * (d_i - mean d))^2 taken literally."""
    torch = pytest.importorskip("torch")
    from verispace_engine.gvpo import (centred, episode_tensors, gvpo_group_backward, ref_logprob,
                                       sequence_logprob)
    model, tok, dev = tiny_build(SimpleCfg())
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.05)
    eps = [make_episode(i, reward=r) for i, r in zip(range(4), [1.0, 0.0, 0.5, 0.25])]
    beta = 0.1
    params = [p for p in model.parameters() if p.requires_grad]
    # literal loss
    tens = [episode_tensors(e.messages, tok, dev) for e in eps]
    lps = [sequence_logprob(model, i, l, grad=True)[0] for i, l in tens]
    refs = [float(ref_logprob(model, i, l)) for i, l in tens]
    d = torch.stack([lp - r for lp, r in zip(lps, refs)])
    c = torch.tensor(centred([e.final_reward for e in eps]))
    loss = (0.5 * (c - beta * (d - d.mean())) ** 2).mean()
    g_literal = torch.autograd.grad(loss, params)
    # implementation
    model.zero_grad()
    stats = gvpo_group_backward(model, tok, dev, eps, [e.final_reward for e in eps], beta, n_total=len(eps), scaler=None)
    g_impl = [p.grad for p in params]
    assert abs(sum(stats["weights"])) < 1e-6
    for a, b in zip(g_literal, g_impl):
        assert torch.allclose(a, b, rtol=2e-3, atol=1e-6)


def test_grpo_group_backward_runs_and_moves_towards_higher_reward():
    torch = pytest.importorskip("torch")
    from verispace_engine.gvpo import episode_tensors, grpo_group_backward, sequence_logprob
    model, tok, dev = tiny_build(SimpleCfg())
    good, bad = make_episode(1, reward=1.0), make_episode(2, reward=0.0)
    eps, rewards = [good, bad], [1.0, 0.0]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.05)
    gi, gl = episode_tensors(good.messages, tok, dev)
    bi, bl = episode_tensors(bad.messages, tok, dev)
    before = float(sequence_logprob(model, gi, gl, grad=False)[0] - sequence_logprob(model, bi, bl, grad=False)[0])
    for _ in range(5):
        model.zero_grad()
        grpo_group_backward(model, tok, dev, eps, rewards, kl_coef=0.04, n_total=2, scaler=None)
        opt.step()
    after = float(sequence_logprob(model, gi, gl, grad=False)[0] - sequence_logprob(model, bi, bl, grad=False)[0])
    assert after > before


# --- the loop ---------------------------------------------------------------------------------------------
def adapter_weights(path):
    from safetensors.torch import load_file
    return load_file(str(Path(path) / "adapter_model.safetensors"))


def test_loop_runs_checkpoints_and_logs(tmp_path, capsys):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo
    calls = []
    s = train_gvpo(tiny_cfg(tmp_path, "a"), TASKS, fake_sampler(calls), tiny_build)
    assert s["finished"] and s["steps_done"] == 6 and len(calls) == 6 * 2          # one fresh group per task per step
    log = [json.loads(x) for x in (tmp_path / "a" / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4, 5, 6]
    for k in ("mean_reward", "degenerate_groups", "mean_log_ratio", "grad_norm", "lr", "valid_rate"):
        assert k in log[0]
    assert all(math.isfinite(r["grad_norm"]) for r in log)
    assert '"event": "checkpoint", "step": 6, "verified_step_in_state_pt": 6' in capsys.readouterr().out
    assert (tmp_path / "a" / "adapter" / "adapter_model.safetensors").exists()
    assert s["algo"] == "gvpo" and s["reward"] == "total_return" and s["replay"] is False


def test_no_replay_every_update_uses_fresh_samples_only(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo
    seen = []
    train_gvpo(tiny_cfg(tmp_path, "a", steps=4), TASKS, fake_sampler(seen), tiny_build)
    assert len(seen) == len(set(seen)) == 8                  # no (task, seed) is ever sampled twice, nothing is reused
    seeds = [s for _, s in seen]
    assert len(set(seeds)) == len(seeds)


def test_the_policy_changes_and_degenerate_groups_are_logged(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo

    def constant(model, tok, device, task_id, group_size, seed, cfg):
        return [make_episode(i, reward=0.5) for i in range(group_size)]
    train_gvpo(tiny_cfg(tmp_path, "c", steps=2), TASKS, constant, tiny_build)
    log = [json.loads(x) for x in (tmp_path / "c" / "train_log.jsonl").read_text().splitlines()]
    assert all(r["degenerate_groups"] == 2 for r in log)
    assert log[0]["mean_log_ratio"] == pytest.approx(0.0, abs=1e-4)     # step 1: policy == reference


def test_resuming_gives_the_same_run_as_never_stopping(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo
    straight = train_gvpo(tiny_cfg(tmp_path, "straight"), TASKS, fake_sampler(), tiny_build)
    first = train_gvpo(tiny_cfg(tmp_path, "resumed"), TASKS, fake_sampler(), tiny_build, stop_after_steps=3)
    assert first["steps_done"] == 3 and not first["finished"]
    second = train_gvpo(tiny_cfg(tmp_path, "resumed"), TASKS, fake_sampler(), tiny_build)
    assert second["resumed_from"] == 3 and second["finished"] and second["steps_done"] == 6
    a = [json.loads(x) for x in (tmp_path / "straight" / "train_log.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (tmp_path / "resumed" / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in b] == [1, 2, 3, 4, 5, 6]
    for k in ("mean_reward", "mean_log_ratio", "loss_proxy"):
        assert [r[k] for r in a] == pytest.approx([r[k] for r in b], rel=1e-4, abs=1e-6)
    wa, wb = adapter_weights(tmp_path / "straight" / "adapter"), adapter_weights(tmp_path / "resumed" / "adapter")
    import torch
    assert wa.keys() == wb.keys() and all(torch.allclose(wa[k], wb[k], atol=1e-5) for k in wa)


def test_a_checkpoint_from_another_config_is_refused(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo
    train_gvpo(tiny_cfg(tmp_path, "a"), TASKS, fake_sampler(), tiny_build, stop_after_steps=2)
    with pytest.raises(ValueError, match="different training config"):
        train_gvpo(tiny_cfg(tmp_path, "a", beta=0.5), TASKS, fake_sampler(), tiny_build)


def test_grpo_arm_and_outcome_only_reward_run(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo
    s = train_gvpo(tiny_cfg(tmp_path, "g", algo="grpo", reward="final_reward", steps=3), TASKS, fake_sampler(), tiny_build)
    assert s["algo"] == "grpo" and s["reward"] == "final_reward" and s["finished"]


def test_reward_choice_changes_what_is_optimised(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo

    def sample(model, tok, device, task_id, group_size, seed, cfg):
        eps = [make_episode(i, reward=0.5, n_tools=1 + (i % 3)) for i in range(group_size)]   # outcome identical, calls differ
        return eps
    a = train_gvpo(tiny_cfg(tmp_path, "tot", steps=1, reward="total_return"), TASKS, sample, tiny_build)
    b = train_gvpo(tiny_cfg(tmp_path, "fin", steps=1, reward="final_reward"), TASKS, sample, tiny_build)
    la = json.loads((tmp_path / "tot" / "train_log.jsonl").read_text().splitlines()[0])
    lb = json.loads((tmp_path / "fin" / "train_log.jsonl").read_text().splitlines()[0])
    assert la["degenerate_groups"] == 0 and lb["degenerate_groups"] == 2
    assert a["finished"] and b["finished"]


def test_hub_mirror_and_resume_from_the_hub_alone(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo
    h = hub()
    first = train_gvpo(tiny_cfg(tmp_path, "a"), TASKS, fake_sampler(), tiny_build, stop_after_steps=3, hub=h)
    assert first["steps_done"] == 3 and h.latest_meta()["step"] == 3
    assert {"train_summary.json", "train_log.jsonl", "checkpoint/state.pt"} <= set(h.list_files())
    second = train_gvpo(tiny_cfg(tmp_path, "fresh"), TASKS, fake_sampler(), tiny_build, hub=h)   # new runtime, only the Hub
    assert second["resumed_from"] == 3 and second["finished"]
    assert "adapter/adapter_model.safetensors" in h.list_files()
    assert json.loads(h.api.files["train_summary.json"])["finished"] is True


def test_a_failed_hub_verification_stops_the_loop(tmp_path):
    pytest.importorskip("torch")
    from verispace_engine.gvpo import train_gvpo
    from verispace_engine.hub import HubVerifyError
    with pytest.raises(HubVerifyError):
        train_gvpo(tiny_cfg(tmp_path, "a"), TASKS, fake_sampler(), tiny_build, stop_after_steps=2,
                   hub=hub(lose={"checkpoint/state.pt"}))


def test_the_step_is_read_back_from_the_written_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import verispace_engine.gvpo as g
    monkeypatch.setattr(g, "_read_step", lambda d: 999)
    with pytest.raises(RuntimeError, match="reads back step 999"):
        g.train_gvpo(tiny_cfg(tmp_path, "a"), TASKS, fake_sampler(), tiny_build, stop_after_steps=1)


def test_task_order_is_a_pure_function_of_seed_and_step():
    from verispace_engine.gvpo import tasks_for_step
    cfg0, cfg1 = tiny_cfg(Path("x"), "a", seed=0), tiny_cfg(Path("x"), "a", seed=1)
    assert tasks_for_step(TASKS, cfg0, 5) == tasks_for_step(TASKS, cfg0, 5)
    assert tasks_for_step(TASKS, cfg0, 5) != tasks_for_step(TASKS, cfg1, 5)
    seen = [t for s in range(6) for t in tasks_for_step(TASKS, cfg0, s)]          # 6 steps x 2 = one pass over 12 tasks
    assert sorted(seen) == sorted(TASKS)
    assert tasks_for_step(TASKS, cfg0, 0) == tasks_for_step(TASKS, tiny_cfg(Path("y"), "b", seed=0, algo="grpo"), 0)  # arms share order


# --- group rollout equals the harness rollout ----------------------------------------------------------------
def _scripted_replies(task):
    from helioguard.policies import ToolInformedPolicy
    from helioguard.sft_traces import action_text
    from verispace_engine.env_base import StepOutcome
    acts = []
    gen = ToolInformedPolicy().actions(task)
    a = next(gen)
    from helioguard.env import HelioGuardEnv
    env = HelioGuardEnv([task])
    env.reset(task_id=task["id"])
    while True:
        acts.append(action_text(a))
        out = env.step(a)
        if out.done:
            break
        try:
            a = gen.send(StepOutcome(out.observation, out.reward, out.done, out.info))
        except StopIteration:
            break
    return acts


def test_rollout_group_matches_rollout_task_attempt_by_attempt():
    from helioguard.harness_env import ALLOWED_TYPES, EXTRA_ACTIONS, compact_observation, load
    from helioguard.env import HelioGuardEnv
    from verispace_engine.harness import RolloutConfig, ScriptedBackend, rollout_task
    from verispace_engine.harness.group_rollout import rollout_group
    env, ids = load("eval", 3)
    tid = ids[0]
    good = _scripted_replies(env._by_id[tid])
    scripts = [good, ["not json"], good[:-1] + ["x"] * 30, good[:3] + ['{"type":"tool_call","name":"nope","arguments":{}}'] + good[3:]]
    cfg = RolloutConfig(attempts=1, temperature=0.7, seed=0, allowed_types=tuple(ALLOWED_TYPES),
                        extra_actions=tuple(EXTRA_ACTIONS), compactor=compact_observation)
    singles = [rollout_task(env, tid, ScriptedBackend(s), cfg, a) for a, s in enumerate(scripts)]
    pos = [0] * len(scripts)

    def generate_batch(messages_list, max_new_tokens, temperature, seed, indices):
        out = []
        for i in indices:
            out.append(scripts[i][min(pos[i], len(scripts[i]) - 1)])
            pos[i] += 1
        return out
    envs = [HelioGuardEnv(env.tasks) for _ in scripts]
    group = rollout_group(envs, tid, generate_batch, cfg, seed=0)
    assert len(group) == len(scripts)
    for single, g in zip(singles, group):
        a, b = single.__dict__, g.result.__dict__
        assert {k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}
        assert g.messages[0]["role"] == "system" and g.messages[-1]["role"] == "assistant"
    assert group[0].result.valid_submission and not group[1].result.valid_submission
    assert group[1].result.failure.startswith("malformed")
    # the transcript ends with the last model output and contains every model turn in order
    n_assistant = [sum(m["role"] == "assistant" for m in g.messages) for g in group]
    assert n_assistant == [s.turns for s in singles]


# --- precheck ------------------------------------------------------------------------------------------------
def test_precheck_summary_counts_identical_groups_and_step_reward_differences():
    from helioguard.gvpo_precheck import summarize_groups
    groups = [
        {"task_id": "a", "final_reward": [0.5] * 8, "total_return": [0.8] * 8, "valid": [True] * 8},
        {"task_id": "b", "final_reward": [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5], "total_return": [0.8, 0.8, 1.3, 0.8, 0.8, 0.8, 0.8, 0.8],
         "valid": [True] * 8},
        {"task_id": "c", "final_reward": [0.5] * 8, "total_return": [0.8, 0.77, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], "valid": [True] * 7 + [False]},
        {"task_id": "d", "final_reward": [-0.5, 0.2, 0.3, 0.1, 0.9, 0.5, 0.4, 0.0], "total_return": [-0.5, 0.5, 0.6, 0.4, 1.2, 0.8, 0.7, 0.3],
         "valid": [False] + [True] * 7},
    ]
    s = summarize_groups(groups)
    assert s["groups"] == 4 and s["samples"] == 32
    assert s["identical_final_reward"] == 2 and s["fraction_identical_final_reward"] == pytest.approx(0.5)
    assert s["fraction_informative_final_reward"] == pytest.approx(0.5)
    assert s["identical_total_return"] == 1                                  # only group "a"
    assert s["groups_where_only_step_rewards_differ"] == 1                   # group "c": outcome identical, total_return differs
    assert s["valid_submission_rate"] == pytest.approx(30 / 32)
    assert s["gate_G0_pass"] is True                                          # 0.5 >= 0.40
    assert summarize_groups(groups[:1])["gate_G0_pass"] is False


def test_precheck_task_selection_is_fixed_and_train_only():
    from helioguard.gvpo_precheck import select_tasks
    ids = [f"train_{i}" for i in range(500)]
    a = select_tasks(ids, 100, seed=0)
    assert a == select_tasks(ids, 100, seed=0) and len(set(a)) == 100 and set(a) <= set(ids)
    assert a != select_tasks(ids, 100, seed=1)
    with pytest.raises(ValueError, match="train"):
        select_tasks(["eval_storm_1", "train_2"], 1, seed=0)


def test_precheck_is_resumable_and_appends_each_group(tmp_path):
    from helioguard.gvpo_precheck import load_groups, run_precheck
    from verispace_engine.gvpo import Episode

    class Crash(Exception):
        pass
    calls = []

    def sample(task_id, k):
        calls.append(task_id)
        if len(calls) == 3:
            raise Crash
        return [Episode(messages=[], final_reward=float(i), total_return=float(i) + 0.3, valid=True) for i in range(k)]
    ids = ["train_1", "train_2", "train_3", "train_4"]
    with pytest.raises(Crash):
        run_precheck(sample, ids, tmp_path / "pre.jsonl", attempts=8)
    assert [g["task_id"] for g in load_groups(tmp_path / "pre.jsonl")] == ["train_1", "train_2"]
    calls.clear()
    out = run_precheck(sample, ids, tmp_path / "pre.jsonl", attempts=8)
    assert calls[:1] == ["train_3"] and len(calls) == 2                      # finished groups are skipped
    assert out["groups"] == 4 and out["identical_final_reward"] == 0


# --- gates ---------------------------------------------------------------------------------------------------
def _runs(rewards_by_seed, valid=True, optimal_at=None):
    from verispace_engine.harness.rollout import EpisodeResult
    out = {}
    for seed, rs in rewards_by_seed.items():
        out[seed] = [EpisodeResult(f"t{i}", 0, r, r, valid if isinstance(valid, bool) else valid[i], 10, 11, None)
                     for i, r in enumerate(rs)]
    return out


def test_gate_constants_match_the_preregistration_text():
    from helioguard import gvpo_gates as g
    text = (ROOT / "docs" / "phase6_gvpo_gates.md").read_text(encoding="utf-8")
    assert (g.G0_MIN_INFORMATIVE, g.G1_MIN_VALID, g.G3_NEEDED_ABOVE, g.G5_SFT_PERCENT_OPTIMAL) == (0.40, 0.90, 0.743, 0.24)
    for needle in ("at least 40%", "0.90", "+0.743", "24%", "G0 to G5 all hold"):
        assert needle in text, needle


def test_gates_pass_when_every_seed_improves_clearly():
    from helioguard.gvpo_gates import evaluate_gates
    sft = _runs({0: [0.80] * 50})[0]
    arm = _runs({s: [1.0] * 20 + [0.97] * 30 for s in (0, 1, 2)})
    rep = evaluate_gates(0.6, arm, sft)
    assert rep["all_pass"] is True and all(rep[g]["pass"] for g in ("G0", "G1", "G2", "G3", "G4", "G5"))


def test_gates_fail_loudly_and_individually():
    from helioguard.gvpo_gates import evaluate_gates
    sft = _runs({0: [0.80] * 50})[0]
    good = {s: [1.0] * 20 + [0.97] * 30 for s in (0, 1, 2)}
    assert evaluate_gates(0.2, _runs(good), sft)["G0"]["pass"] is False                     # precheck below 40%
    rep = evaluate_gates(0.6, _runs(good, valid=[True] * 44 + [False] * 6), sft)
    assert rep["G1"]["pass"] is False and rep["all_pass"] is False                          # 88% valid < 90%
    flat = evaluate_gates(0.6, _runs({s: [0.80] * 50 for s in (0, 1, 2)}), sft)
    assert flat["G2"]["pass"] is False and flat["all_pass"] is False                         # no improvement
    seeds = evaluate_gates(0.6, _runs({0: [0.97] * 50, 1: [0.97] * 50, 2: [0.70] * 50}), sft)
    assert seeds["G4"]["pass"] is False and seeds["G4"]["per_seed"] == pytest.approx([0.17, 0.17, -0.10])
    low = evaluate_gates(0.6, _runs({s: [0.74] * 50 for s in (0, 1, 2)}), _runs({0: [0.5] * 50})[0])
    assert low["G2"]["pass"] is True and low["G3"]["pass"] is False                          # beats SFT but not +0.743 with CI
    assert low["all_pass"] is False


def test_percent_optimal_gate_uses_the_harness_definition():
    from helioguard.gvpo_gates import evaluate_gates
    sft = _runs({0: [0.80] * 50})[0]
    arm = _runs({s: [1.0] * 10 + [0.99] * 40 for s in (0, 1, 2)})                          # 20% optimal < 24%
    rep = evaluate_gates(0.6, arm, sft)
    assert rep["G5"]["pass"] is False and rep["G5"]["percent_optimal"] == pytest.approx(0.2)


def test_paired_comparison_between_arms():
    from helioguard.gvpo_gates import compare_arms
    a, b = _runs({s: [0.9] * 50 for s in (0, 1)}), _runs({s: [0.7] * 50 for s in (0, 1)})
    r = compare_arms(a, b)
    assert r["mean_difference"] == pytest.approx(0.2) and r["ci_low"] > 0 and r["ci_excludes_zero_above"]


# --- configs and docs -----------------------------------------------------------------------------------------
def test_registered_config_matches_the_preregistration():
    from verispace_engine.gvpo import load_gvpo_config
    cfg = load_gvpo_config(ROOT / "configs" / "gvpo_1p5b.json")
    assert (cfg.group_size, cfg.tasks_per_step, cfg.steps, cfg.lr, cfg.warmup_steps, cfg.beta, cfg.kl_coef) == \
        (8, 2, 30, 5e-5, 3, 0.1, 0.04)
    assert (cfg.temperature, cfg.top_p, cfg.max_new_tokens, cfg.max_grad_norm, cfg.weight_decay) == (0.7, 0.95, 256, 1.0, 0.0)
    assert (cfg.lora_r, cfg.lora_alpha, cfg.replay, cfg.algo, cfg.reward) == (16, 32, False, "gvpo", "total_return")
    assert cfg.model == "Qwen/Qwen2.5-1.5B-Instruct"
    text = (ROOT / "docs" / "phase6_gvpo_gates.md").read_text(encoding="utf-8")
    for needle in ("| Steps | **30**", "learning rate 5e-5", "GVPO beta | 0.1", "KL coefficient 0.04", "| Replay | **off**"):
        assert needle in text, needle


def test_smoke_config_is_small_and_on_the_0p5b_model():
    from verispace_engine.gvpo import load_gvpo_config
    cfg = load_gvpo_config(ROOT / "configs" / "gvpo_smoke_0p5b.json")
    assert "0.5B" in cfg.model and cfg.steps == 10 and cfg.replay is False and cfg.group_size <= 4


def test_preregistration_precedes_the_code_in_git_history():
    import subprocess
    first = lambda path: subprocess.run(["git", "log", "--diff-filter=A", "--format=%ct", "--", path], capture_output=True,
                                        text=True, cwd=ROOT).stdout.split()
    doc, code = first("docs/phase6_gvpo_gates.md"), first("verispace_engine/gvpo.py")
    if not doc or not code:
        pytest.skip("code not yet committed")
    assert int(doc[-1]) <= int(code[-1])


def test_doc_states_which_step_rewards_exist_and_that_none_cover_the_decision():
    text = (ROOT / "docs" / "phase6_gvpo_gates.md").read_text(encoding="utf-8")
    assert "None of these covers the decision step except the final reward" in text
    assert "cannot test whether" in text and "process rewards" in text


# --- slow: 10-step CPU smoke test on the 0.5B model ---------------------------------------------------------------
@pytest.mark.slow
def test_cpu_smoke_run_10_steps_on_the_0p5b_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from helioguard.gvpo_run import run_training
    from verispace_engine.gvpo import load_gvpo_config
    # No SFT adapter: with it, one step took 17 minutes on this CPU (full 10-turn episodes), so a 10-step smoke would take hours.
    # The base 0.5B model mostly fails the strict parser at turn 1, which keeps episodes short; the loop, the update,
    # checkpointing and the log are exercised either way.
    cfg = load_gvpo_config(ROOT / "configs" / "gvpo_smoke_0p5b.json", out_dir=str(tmp_path / "o"),
                           checkpoint_dir=str(tmp_path / "o" / "ckpt"))
    summary = run_training(cfg)
    assert summary["steps_done"] == 10 and summary["finished"]
    log = [json.loads(x) for x in (tmp_path / "o" / "train_log.jsonl").read_text().splitlines()]
    assert len(log) == 10 and all(math.isfinite(r["grad_norm"]) and math.isfinite(r["loss_proxy"]) for r in log)
    assert log[0]["mean_log_ratio"] == pytest.approx(0.0, abs=1e-3)           # step 1: policy == reference
