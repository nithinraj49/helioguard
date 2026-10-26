"""One command for evaluating a model (or a scripted baseline) on a task split.

    python -m verispace_engine.harness.eval --domain <module> --model <hf model id> --split eval --n 50
    python -m verispace_engine.harness.eval --domain <module> --model baseline:<name> --split eval --n 50
    python -m verispace_engine.harness.eval --domain <module> --backend openai --model <name> --base-url <url>

The domain plug-in (a module named by --domain; a project may wrap main() with a default) provides load(split, n), ALLOWED_TYPES,
EXTRA_ACTIONS and BASELINES. Writes <out>/<model>_<split>.jsonl (every turn) and a summary JSON.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

from .backends import HFBackend, OpenAICompatibleBackend
from .rollout import JsonlLog, RolloutConfig, run_rollouts, summarize


def main(argv: list[str] | None = None, default_domain: str | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id, endpoint model name, or baseline:<name>")
    p.add_argument("--backend", choices=("hf", "openai"), default="hf")
    p.add_argument("--base-url", default="https://api.openai.com/v1")
    p.add_argument("--key-var", default="OPENAI_API_KEY", help="name of the variable holding the key (env or .env)")
    p.add_argument("--domain", default=default_domain, required=default_domain is None,
                   help="module providing load(), ALLOWED_TYPES, EXTRA_ACTIONS, BASELINES")
    p.add_argument("--split", default="eval")
    p.add_argument("--n", type=int, default=50, help="number of tasks")
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--adapter", default=None, help="LoRA adapter folder to apply to a local --model")
    p.add_argument("--out", default="results/phase5")
    a = p.parse_args(argv)

    domain = importlib.import_module(a.domain)
    env, ids = domain.load(a.split, a.n)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", a.model)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{slug}_{a.split}.jsonl"
    log_path.write_text("", encoding="utf-8")
    scripted = a.model.startswith("baseline:")

    if scripted:
        name, log = a.model.split(":", 1)[1], JsonlLog(log_path)
        results = [domain.baseline_result(env, name, t) for t in ids]
        for r in results:
            log.write({"model": a.model, "event": "episode", **r.__dict__})
    else:
        cfg = RolloutConfig(attempts=a.attempts, max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                            seed=a.seed, allowed_types=tuple(domain.ALLOWED_TYPES),
                            extra_actions=tuple(domain.EXTRA_ACTIONS),
                            compactor=getattr(domain, "compact_observation", None))
        backend = (HFBackend(a.model, device=a.device, adapter=a.adapter) if a.backend == "hf"
                   else OpenAICompatibleBackend(a.model, base_url=a.base_url, key_var=a.key_var))
        results = run_rollouts(env, ids, backend, cfg, log_path)
    summary = {"model": a.model, "split": a.split, "tasks": len(ids),
               "attempts": 1 if scripted else a.attempts, "temperature": None if scripted else a.temperature,
               **summarize(results)}
    (out_dir / f"{slug}_{a.split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
