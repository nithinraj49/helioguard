"""Supervised fine-tuning of a chat model on episode traces (LoRA), resumable, domain-neutral.

build_labels: tokenise a chat so that only assistant tokens carry loss (system, user and tool
outputs are context). train: a plain PyTorch loop with LoRA through PEFT, or Unsloth when it is
selected (config field `trainer`) and a GPU is present. Torch, transformers, peft and unsloth are imported inside the
functions that need them, so the engine imports without them (optional "sft" extra).

Resume: a checkpoint holds the adapter, optimizer, GradScaler, step and every RNG state. The data
order is a pure function of (seed, step), so resuming gives the same run as never stopping.
"""
from __future__ import annotations

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

IGNORE = -100
__all__ = ["IGNORE", "SFTConfig", "build_labels", "load_config", "tokenize_rows", "train"]


@dataclass
class SFTConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data: str = "data/sft/train_traces.jsonl"
    out_dir: str = "results/phase6/sft_0p5b"
    checkpoint_dir: str = "results/phase6/sft_0p5b/checkpoints"      # point at Drive on Colab
    max_seq_len: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs: float = 1.0
    grad_accum: int = 8                       # examples per optimizer step (micro-batch is one example)
    warmup_steps: int = 5
    max_steps: int | None = None              # overrides epochs when set (smoke tests)
    max_grad_norm: float = 1.0
    seed: int = 0
    checkpoint_minutes: float = 10.0
    log_every: int = 1
    dtype: str = "auto"                       # auto: float32 on CPU, float16 on GPU
    trainer: str = "peft"                     # "peft" (transformers + PEFT) or "unsloth" (GPU only)
    gradient_checkpointing: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path, **overrides: Any) -> SFTConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    raw.update(overrides)
    if "target_modules" in raw:
        raw["target_modules"] = tuple(raw["target_modules"])
    return SFTConfig(**raw)


# --- loss mask ---------------------------------------------------------------------------------
def build_labels(messages: Sequence[dict[str, str]], tokenizer: Any) -> tuple[list[int], list[int]]:
    """(input_ids, labels) for a whole chat; labels are IGNORE except on assistant turns.

    Each assistant span runs from the end of its generation prompt to the end of its turn (the
    end-of-turn token included, the newline after it excluded). Raises if rendering a chat prefix does
    not extend the previous one, which would make spans ambiguous."""
    messages = list(messages)

    def ids_of(msgs, gen_prompt):
        out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen_prompt, return_dict=True)
        return list(out["input_ids"])

    full = ids_of(messages, False)
    labels = [IGNORE] * len(full)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(ids_of(messages[:i], True))
        end = len(ids_of(messages[: i + 1], False))
        if full[:start] != ids_of(messages[:i], True) or end > len(full) or full[:end] != ids_of(messages[: i + 1], False):
            raise ValueError("chat template prefix is not stable; cannot place the loss mask")
        if end > start and tokenizer.decode(full[end - 1: end]).strip() == "":
            end -= 1                                        # drop the newline after the end-of-turn token
        labels[start:end] = full[start:end]
    return full, labels


def tokenize_rows(rows: Sequence[dict], tokenizer: Any, max_len: int) -> tuple[list[dict[str, list[int]]], int]:
    """Tokenise trace rows; rows longer than max_len are skipped (never truncated). Returns (examples, skipped)."""
    examples, skipped = [], 0
    for r in rows:
        ids, labels = build_labels(r["messages"], tokenizer)
        if len(ids) > max_len:
            skipped += 1
            continue
        examples.append({"input_ids": ids, "labels": labels})
    return examples, skipped


# --- training ----------------------------------------------------------------------------------
def _cfg_hash(cfg: SFTConfig) -> str:
    keep = {k: v for k, v in asdict(cfg).items()
            if k not in ("max_steps", "checkpoint_dir", "out_dir", "checkpoint_minutes", "log_every", "extra")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _lr_at(cfg: SFTConfig, step: int, total: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    return cfg.lr * max(0.0, (total - step) / max(1, total - cfg.warmup_steps))


def _order(n: int, seed: int, epoch: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(idx)
    return idx


def _example_for(step: int, micro: int, cfg: SFTConfig, n: int, steps_per_epoch: int) -> int:
    epoch, within = divmod(step, steps_per_epoch)
    return _order(n, cfg.seed, epoch)[within * cfg.grad_accum + micro]


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(model, optimizer, scaler, step: int, cfg: SFTConfig, log: list[dict]) -> Path:
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


def _load_checkpoint(model, optimizer, scaler, cfg: SFTConfig) -> tuple[int, list[dict]]:
    import torch
    root = Path(cfg.checkpoint_dir)
    latest = root / "latest" if (root / "latest" / "state.pt").exists() else root / "old"
    if not (latest / "state.pt").exists():
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


def _build_model(cfg: SFTConfig, tokenizer_hint: Any = None):
    """(model, tokenizer, device) with a LoRA adapter attached. Unsloth on GPU when available."""
    import torch
    has_gpu = torch.cuda.is_available()
    if cfg.trainer not in ("peft", "unsloth"):
        raise ValueError(f"trainer must be 'peft' or 'unsloth', got {cfg.trainer!r}")
    if cfg.trainer == "unsloth":
        if not has_gpu:
            raise RuntimeError("trainer 'unsloth' needs a GPU; use trainer 'peft' on CPU")
        try:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(cfg.model, max_seq_length=cfg.max_seq_len,
                                                           dtype=torch.float16, load_in_4bit=False)
            model = FastLanguageModel.get_peft_model(
                model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.target_modules), random_state=cfg.seed,
                use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False)
            model.trainer_name = "unsloth"
            return model, tok, "cuda"
        except ImportError as exc:                 # no silent fallback: the registered trainer is the one that runs
            raise RuntimeError("trainer 'unsloth' is selected but unsloth is not installed; "
                               "install it or set trainer to 'peft'") from exc
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"auto": torch.float16 if has_gpu else torch.float32, "float32": torch.float32,
             "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.dtype]
    device = "cuda" if has_gpu else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    torch.manual_seed(cfg.seed)
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"))
    model.trainer_name = "transformers+peft"
    return model, tok, device


def train(cfg: SFTConfig, rows: Sequence[dict] | None = None,
          build: Callable[[SFTConfig], tuple[Any, Any, str]] | None = None,
          stop_after_steps: int | None = None) -> dict[str, Any]:
    """Run (or resume) SFT. `build(cfg) -> (peft_model, tokenizer, device)` lets tests supply a tiny model.
    `stop_after_steps` ends the run early after checkpointing (to test resuming). Returns a summary."""
    import torch
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model, tok, device = (build or _build_model)(cfg)
    if rows is None:
        rows = [json.loads(line) for line in Path(cfg.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    examples, skipped = tokenize_rows(rows, tok, cfg.max_seq_len)
    if len(examples) < cfg.grad_accum:
        raise ValueError(f"only {len(examples)} usable examples for grad_accum={cfg.grad_accum}")
    steps_per_epoch = len(examples) // cfg.grad_accum
    total = cfg.max_steps if cfg.max_steps else max(1, math.ceil(cfg.epochs * steps_per_epoch))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_scaler = device == "cuda" and any(p.dtype == torch.float16 for p in model.parameters())
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    step, log = _load_checkpoint(model, optimizer, scaler, cfg)
    resumed_from = step
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    last_ckpt, started = time.time(), time.time()
    model.train()

    while step < total:
        for g in optimizer.param_groups:
            g["lr"] = _lr_at(cfg, step, total)
        group = [examples[_example_for(step, m, cfg, len(examples), steps_per_epoch)] for m in range(cfg.grad_accum)]
        n_tokens = sum(sum(1 for t in e["labels"][1:] if t != IGNORE) for e in group)
        loss_sum = 0.0
        for e in group:
            ids = torch.tensor([e["input_ids"]], device=device)
            labels = torch.tensor([e["labels"]], device=device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, labels[0, 1:], ignore_index=IGNORE, reduction="sum") / n_tokens
            (scaler.scale(loss) if scaler else loss).backward()
            loss_sum += float(loss)
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
        rec = {"step": step, "loss": loss_sum, "grad_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"],
               "label_tokens": n_tokens, "elapsed_s": round(time.time() - started, 1)}
        log.append(rec)
        if step % cfg.log_every == 0:
            print(json.dumps(rec), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        stopping = stop_after_steps is not None and step - resumed_from >= stop_after_steps
        if stopping or step == total or (time.time() - last_ckpt) / 60 >= cfg.checkpoint_minutes:
            _save_checkpoint(model, optimizer, scaler, step, cfg, log)
            last_ckpt = time.time()
        if stopping:
            break
    if step == total:
        model.save_pretrained(str(out_dir / "adapter"))
    summary = {"model": cfg.model, "trainer": getattr(model, "trainer_name", "custom"), "training_seed": cfg.seed,
               "steps_done": step, "steps_total": total, "resumed_from": resumed_from,
               "examples": len(examples), "skipped_too_long": skipped, "steps_per_epoch": steps_per_epoch,
               "final_loss": log[-1]["loss"] if log else None, "first_loss": log[0]["loss"] if log else None,
               "config": asdict(cfg), "finished": step == total}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
