"""GVPO (Group Variance Policy Optimization) and a GRPO baseline for multi-turn episodes, resumable, domain-neutral.

Per task, a group of k episodes is sampled from the CURRENT policy (replay is off: nothing is stored or reused).
With R_i the episode reward, l_i = log pi_theta(y_i) the summed log-probability of all assistant tokens of the episode
(tool results are context), and r_i = log pi_ref(y_i) the same under the reference policy:

    c_i = R_i - mean(R)                        centred reward, NEVER divided by the group standard deviation
    d_i = l_i - r_i
    w_i = c_i - beta * (d_i - mean(d))         the weights sum to zero in a group (asserted)
    loss = mean over episodes of 0.5 * (c_i - beta * (d_i - mean(d)))^2,   gradient = -(beta/N) * sum_i w_i * grad l_i

so the update is two passes per group: no-grad log-probs under the policy and the reference, then one backward per
episode of -(beta * w_i / N) * l_i. The reference is the model with the (new) LoRA adapter DISABLED. To make that
reference the SFT model, the SFT adapter is merged into the base weights first and a fresh zero-initialised LoRA is
trained on top (see `build_policy`); at step 0 the policy equals the reference.

GRPO baseline: advantage = centred reward / group std, loss = -A_i * mean-token log-prob + kl_coef * mean k3 KL to
the reference. Same sampling, same optimizer, same reference.

Resume: a checkpoint holds the adapter, optimizer, GradScaler, step and RNG states; the task order and the sampling
seeds are pure functions of (seed, step), so resuming gives the same run as never stopping. Every checkpoint is read
back (the step in state.pt is checked and printed) and, with a `HubSync`, mirrored to a private Hub repo.
Torch, transformers and peft are imported inside the functions that need them.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .sft import IGNORE, _read_step, _rng_state, _set_rng_state, build_labels, git_state

__all__ = ["Episode", "GVPOConfig", "assert_zero_sum", "build_policy", "centred", "episode_tensors", "grpo_advantages",
           "grpo_group_backward", "gvpo_group_backward", "gvpo_weights", "load_gvpo_config", "lr_at", "ref_logprob",
           "sequence_logprob", "tasks_for_step", "train_gvpo"]


@dataclass
class Episode:
    messages: list[dict[str, str]]            # the whole chat, ending with the model's last output
    final_reward: float
    total_return: float                       # sum of all step rewards including the final one
    valid: bool = True
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class GVPOConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    init_adapter: str | None = None           # SFT adapter folder, merged into the base: the reference is the SFT model
    algo: str = "gvpo"                        # "gvpo" or "grpo"
    reward: str = "total_return"              # "total_return" (step+final) or "final_reward" (outcome-only)
    out_dir: str = "results/phase6/gvpo"
    checkpoint_dir: str = "results/phase6/gvpo/checkpoints"      # point at Drive on Colab
    group_size: int = 8
    tasks_per_step: int = 2
    steps: int = 30
    lr: float = 5e-5
    warmup_steps: int = 3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta: float = 0.1                         # GVPO
    kl_coef: float = 0.04                     # GRPO
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    max_seq_len: int = 4096                   # longer episodes are left out of the update (counted in the log)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    seed: int = 0
    checkpoint_minutes: float = 10.0
    dtype: str = "auto"                       # auto: float32 on CPU, float16 on GPU
    gradient_checkpointing: bool = True
    replay: bool = False                      # must stay False: updates use only samples from the current policy
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.replay:
            raise ValueError("replay is off: GVPO here trains only on samples from the current policy")
        if self.algo not in ("gvpo", "grpo"):
            raise ValueError(f"algo must be 'gvpo' or 'grpo', got {self.algo!r}")
        if self.reward not in ("total_return", "final_reward"):
            raise ValueError(f"reward must be 'total_return' or 'final_reward', got {self.reward!r}")
        if self.group_size < 2:
            raise ValueError("a group needs at least 2 episodes")


def load_gvpo_config(path: str | Path, **overrides: Any) -> GVPOConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    raw.update(overrides)
    if "target_modules" in raw:
        raw["target_modules"] = tuple(raw["target_modules"])
    return GVPOConfig(**raw)


# --- the maths (pure python) ------------------------------------------------------------------------------------
def centred(rewards: Sequence[float]) -> list[float]:
    """Reward minus the group mean. Not divided by the standard deviation."""
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def assert_zero_sum(weights: Sequence[float], tol: float = 1e-6) -> None:
    if abs(sum(weights)) > tol * max(1.0, sum(abs(w) for w in weights)):
        raise AssertionError(f"group weights must sum to zero, got {sum(weights):.3e}")


def gvpo_weights(rewards: Sequence[float], logp: Sequence[float], ref_logp: Sequence[float], beta: float) -> list[float]:
    """w_i = (R_i - mean R) - beta * (d_i - mean d), d_i = logp_i - ref_logp_i. Sums to zero (asserted)."""
    c = centred(rewards)
    d = [a - b for a, b in zip(logp, ref_logp)]
    d_mean = sum(d) / len(d)
    w = [ci - beta * (di - d_mean) for ci, di in zip(c, d)]
    assert_zero_sum(w)
    return w


def grpo_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """GRPO baseline: centred reward divided by the group standard deviation (zero for identical rewards)."""
    c = centred(rewards)
    std = math.sqrt(sum(x * x for x in c) / len(c))
    return [x / (std + eps) for x in c]


def lr_at(cfg: GVPOConfig, step: int) -> float:
    """Linear warm-up, then constant."""
    return cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))


def _order(n: int, seed: int, epoch: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(idx)
    return idx


def tasks_for_step(task_ids: Sequence[str], cfg: GVPOConfig, step: int) -> list[str]:
    """The tasks of an optimizer step: a pure function of (seed, step), identical for every arm."""
    n = len(task_ids)
    out = []
    for j in range(cfg.tasks_per_step):
        epoch, within = divmod(step * cfg.tasks_per_step + j, n)
        out.append(task_ids[_order(n, cfg.seed, epoch)[within]])
    return out


# --- model-level pieces (torch) ----------------------------------------------------------------------------------
def episode_tensors(messages: Sequence[dict[str, str]], tokenizer: Any, device: str, max_len: int | None = None):
    """(input_ids, labels) as 1 x T tensors; labels are IGNORE except on assistant tokens."""
    import torch
    ids, labels = build_labels(messages, tokenizer)
    if max_len is not None and len(ids) > max_len:
        raise ValueError(f"episode has {len(ids)} tokens, over the limit of {max_len}")
    return torch.tensor([ids], device=device), torch.tensor([labels], device=device)


def sequence_logprob(model: Any, ids: Any, labels: Any, grad: bool):
    """(total, per_token): summed and per-token log-probability of the assistant tokens of one episode."""
    import torch
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=ids).logits[0, :-1]
        target = labels[0, 1:]
        mask = target != IGNORE
        per_token = -torch.nn.functional.cross_entropy(logits[mask].float(), target[mask], reduction="none")
    return per_token.sum(), per_token


def ref_logprob(model: Any, ids: Any, labels: Any, per_token: bool = False):
    """Log-probability under the reference policy: the same model with the LoRA adapter disabled. No gradient."""
    import torch
    with torch.no_grad(), model.disable_adapter():
        total, tokens = sequence_logprob(model, ids, labels, grad=False)
    return tokens if per_token else total


def _backward(loss: Any, scaler: Any) -> None:
    (scaler.scale(loss) if scaler is not None else loss).backward()


def gvpo_group_backward(model: Any, tokenizer: Any, device: str, episodes: Sequence[Episode], rewards: Sequence[float],
                        beta: float, n_total: int, scaler: Any = None) -> dict[str, Any]:
    """Accumulate the GVPO gradient of one group. Pass 1: no-grad log-probs under policy and reference give the weights.
    Pass 2: one backward per episode of -(beta * w_i / n_total) * logp_i (= the gradient of the group's loss term)."""
    tens = [episode_tensors(e.messages, tokenizer, device) for e in episodes]
    lp = [float(sequence_logprob(model, i, l, grad=False)[0]) for i, l in tens]
    ref = [float(ref_logprob(model, i, l)) for i, l in tens]
    w = gvpo_weights(rewards, lp, ref, beta)
    d = [a - b for a, b in zip(lp, ref)]
    d_mean = sum(d) / len(d)
    for (ids, labels), wi in zip(tens, w):
        if abs(wi) < 1e-12:
            continue
        total, _ = sequence_logprob(model, ids, labels, grad=True)
        _backward(-(beta * wi / n_total) * total, scaler)
    c = centred(rewards)
    loss_sum = sum(0.5 * (ci - beta * (di - d_mean)) ** 2 for ci, di in zip(c, d))
    return {"weights": w, "log_ratio": d, "loss_sum": loss_sum}


def grpo_group_backward(model: Any, tokenizer: Any, device: str, episodes: Sequence[Episode], rewards: Sequence[float],
                        kl_coef: float, n_total: int, scaler: Any = None) -> dict[str, Any]:
    """Accumulate the GRPO gradient of one group (one gradient step per batch, so the ratio is 1 and there is no clipping)."""
    import torch
    adv = grpo_advantages(rewards)
    log_ratio, loss_sum = [], 0.0
    for e, a in zip(episodes, adv):
        ids, labels = episode_tensors(e.messages, tokenizer, device)
        total, per = sequence_logprob(model, ids, labels, grad=True)
        ref_tok = ref_logprob(model, ids, labels, per_token=True)
        gap = ref_tok - per
        kl = (torch.exp(gap) - gap - 1.0).mean()
        loss = (-(a * per.mean()) + kl_coef * kl) / n_total
        _backward(loss, scaler)
        log_ratio.append(float(total.detach()) - float(ref_tok.sum()))
        loss_sum += float(loss.detach()) * n_total
    return {"weights": adv, "log_ratio": log_ratio, "loss_sum": loss_sum}


def build_policy(cfg: GVPOConfig):
    """(model, tokenizer, device): base model, SFT adapter merged in (if given), plus a fresh zero-initialised LoRA.
    At step 0 the policy equals the SFT model, and disabling the adapter gives exactly that model back."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    has_gpu = torch.cuda.is_available()
    dtype = {"auto": torch.float16 if has_gpu else torch.float32, "float32": torch.float32,
             "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.dtype]
    device = "cuda" if has_gpu else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model)
    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device)
    if cfg.init_adapter:
        base = PeftModel.from_pretrained(base, cfg.init_adapter).merge_and_unload()
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    torch.manual_seed(cfg.seed)
    model = get_peft_model(base, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                            target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"))
    model.trainer_name = f"{cfg.algo}+peft"
    return model, tok, device


# --- checkpoints ---------------------------------------------------------------------------------------------------
def _cfg_hash(cfg: GVPOConfig) -> str:
    keep = {k: v for k, v in asdict(cfg).items()
            if k not in ("steps", "out_dir", "checkpoint_dir", "checkpoint_minutes", "init_adapter", "extra")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _save_checkpoint(model, optimizer, scaler, step: int, cfg: GVPOConfig, log: list[dict]) -> Path:
    import torch
    root = Path(cfg.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp, latest, old = root / "tmp", root / "latest", root / "old"
    for d in (tmp, old):
        shutil.rmtree(d, ignore_errors=True)
    tmp.mkdir()
    model.save_pretrained(str(tmp / "adapter"))
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "rng": _rng_state(), "cfg_hash": _cfg_hash(cfg), "log": log}, tmp / "state.pt")
    if latest.exists():
        os.replace(latest, old)                 # a crash between these two lines leaves `old` to recover from
    os.replace(tmp, latest)
    shutil.rmtree(old, ignore_errors=True)
    return latest


def _local_checkpoint(cfg: GVPOConfig) -> Path | None:
    root = Path(cfg.checkpoint_dir)
    for d in (root / "latest", root / "old"):
        if (d / "state.pt").exists():
            return d
    return None


def _restore_from_hub_if_newer(cfg: GVPOConfig, hub: Any) -> None:
    remote = hub.latest_meta()
    if remote is None:
        return
    if remote.get("cfg_hash") != _cfg_hash(cfg):
        raise ValueError("Hub checkpoint was written with a different training config; use a new Hub repo")
    local = _local_checkpoint(cfg)
    local_step = _read_step(local) if local else -1
    if int(remote["step"]) > local_step:
        step = hub.download_checkpoint(Path(cfg.checkpoint_dir) / "latest")
        shutil.rmtree(Path(cfg.checkpoint_dir) / "old", ignore_errors=True)
        print(f"resume: downloaded step {step} from Hub {hub.repo_id} (local was {local_step if local else 'empty'})", flush=True)
    else:
        print(f"resume: local checkpoint step {local_step} is at least as new as Hub step {remote['step']}", flush=True)


def _load_checkpoint(model, optimizer, scaler, cfg: GVPOConfig) -> tuple[int, list[dict]]:
    import torch
    latest = _local_checkpoint(cfg)
    if latest is None:
        return 0, []
    state = torch.load(latest / "state.pt", weights_only=False, map_location="cpu")
    if state["cfg_hash"] != _cfg_hash(cfg):
        raise ValueError("checkpoint was written with a different training config; use a new checkpoint_dir")
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model, load_file(str(latest / "adapter" / "adapter_model.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    _set_rng_state(state["rng"])
    return int(state["step"]), list(state["log"])


# --- the loop --------------------------------------------------------------------------------------------------------
SampleFn = Callable[[Any, Any, str, str, int, int, GVPOConfig], "list[Episode]"]


def train_gvpo(cfg: GVPOConfig, task_ids: Sequence[str], sample_fn: SampleFn,
               build: Callable[[GVPOConfig], tuple[Any, Any, str]] | None = None, hub: Any = None,
               stop_after_steps: int | None = None) -> dict[str, Any]:
    """Run (or resume) GVPO/GRPO. `sample_fn(model, tokenizer, device, task_id, group_size, seed, cfg)` returns the
    group's episodes sampled from the current policy. `build(cfg)` lets tests supply a tiny model.
    `stop_after_steps` ends the run early after checkpointing (to test resuming). Returns a summary."""
    import torch
    git_commit, git_dirty = git_state()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model, tok, device = (build or build_policy)(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_scaler = device == "cuda" and any(p.dtype == torch.float16 for p in model.parameters())
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    if hub is not None:
        _restore_from_hub_if_newer(cfg, hub)
    step, log = _load_checkpoint(model, optimizer, scaler, cfg)
    resumed_from = step
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    if step:                                    # rebuild the log file from the checkpoint (it may be missing or stale)
        log_path.write_text("".join(json.dumps(r) + "\n" for r in log), encoding="utf-8")
    elif log_path.exists():
        log_path.unlink()
    started, last_ckpt = time.time(), time.time()

    def write_summary() -> dict[str, Any]:
        summary = {"model": cfg.model, "init_adapter": cfg.init_adapter, "algo": cfg.algo, "reward": cfg.reward,
                   "trainer": getattr(model, "trainer_name", "custom"), "training_seed": cfg.seed, "replay": cfg.replay,
                   "git_commit": git_commit, "git_dirty": git_dirty, "steps_done": step, "steps_total": cfg.steps,
                   "resumed_from": resumed_from, "first_mean_reward": log[0]["mean_reward"] if log else None,
                   "last_mean_reward": log[-1]["mean_reward"] if log else None, "config": asdict(cfg),
                   "finished": step == cfg.steps}
        (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    while step < cfg.steps:
        for g in optimizer.param_groups:
            g["lr"] = lr_at(cfg, step)
        groups = []
        for j, tid in enumerate(tasks_for_step(task_ids, cfg, step)):
            groups.append(sample_fn(model, tok, device, tid, cfg.group_size, cfg.seed * 1_000_003 + step * 101 + j, cfg))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        kept, skipped_long = [], 0
        for eps in groups:                       # episodes over the context limit are left out, never truncated
            ok = []
            for e in eps:
                try:
                    episode_tensors(e.messages, tok, device, cfg.max_seq_len)
                    ok.append(e)
                except ValueError:
                    skipped_long += 1
            kept.append(ok)
        n_total = sum(len(g) for g in kept)
        loss_sum, log_ratios, degenerate, used_groups = 0.0, [], 0, 0
        for eps in kept:
            if len(eps) < 2:
                continue
            rewards = [float(getattr(e, cfg.reward)) for e in eps]
            used_groups += 1
            degenerate += max(rewards) - min(rewards) < 1e-9
            stats = (gvpo_group_backward(model, tok, device, eps, rewards, cfg.beta, n_total, scaler) if cfg.algo == "gvpo"
                     else grpo_group_backward(model, tok, device, eps, rewards, cfg.kl_coef, n_total, scaler))
            loss_sum += stats["loss_sum"]
            log_ratios += stats["log_ratio"]
        if scaler:
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        flat = [e for g in groups for e in g]
        rec = {"step": step, "mean_reward": sum(float(getattr(e, cfg.reward)) for e in flat) / len(flat),
               "mean_final_reward": sum(e.final_reward for e in flat) / len(flat),
               "mean_total_return": sum(e.total_return for e in flat) / len(flat),
               "valid_rate": sum(e.valid for e in flat) / len(flat), "episodes": len(flat), "groups": used_groups,
               "degenerate_groups": degenerate, "skipped_too_long": skipped_long,
               "mean_log_ratio": sum(log_ratios) / len(log_ratios) if log_ratios else 0.0,
               "loss_proxy": loss_sum / max(1, n_total), "grad_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"],
               "elapsed_s": round(time.time() - started, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        stopping = stop_after_steps is not None and step - resumed_from >= stop_after_steps
        if stopping or step == cfg.steps or (time.time() - last_ckpt) / 60 >= cfg.checkpoint_minutes:
            if step == cfg.steps:                # final adapter and summary first, so the last upload carries them
                model.save_pretrained(str(out_dir / "adapter"))
            latest = _save_checkpoint(model, optimizer, scaler, step, cfg, log)
            last_ckpt = time.time()
            write_summary()
            read_back = _read_step(latest)       # prove the file on disk loads, and say which step it holds
            if read_back != step:
                raise RuntimeError(f"checkpoint at {latest} reads back step {read_back}, expected {step}")
            print(json.dumps({"event": "checkpoint", "step": step, "verified_step_in_state_pt": read_back,
                              "path": str(latest)}), flush=True)
            if hub is not None:
                extra = {"train_summary.json": out_dir / "train_summary.json", "train_log.jsonl": log_path}
                if step == cfg.steps:
                    extra.update({f"adapter/{f.name}": f for f in (out_dir / "adapter").glob("*") if f.is_file()})
                hub.upload_checkpoint(latest, step, _cfg_hash(cfg), extra)
        if stopping:
            break
    if step == cfg.steps and resumed_from == cfg.steps:      # resumed a finished run: still publish the outputs
        model.save_pretrained(str(out_dir / "adapter"))
        write_summary()
        if hub is not None and (latest := _local_checkpoint(cfg)) is not None:
            extra = {"train_summary.json": out_dir / "train_summary.json", "train_log.jsonl": log_path,
                     **{f"adapter/{f.name}": f for f in (out_dir / "adapter").glob("*") if f.is_file()}}
            hub.upload_checkpoint(latest, step, _cfg_hash(cfg), extra)
    return write_summary()
