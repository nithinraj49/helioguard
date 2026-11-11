"""GVPO / GRPO training from the SFT model. Resumable: run the same command again to continue.

    python scripts/train_gvpo.py configs/gvpo_smoke_0p5b.json                      # CPU smoke test
    python scripts/train_gvpo.py configs/gvpo_1p5b.json --seed 0                   # arm A, seed 0 (Colab T4)
    python scripts/train_gvpo.py configs/gvpo_1p5b.json --algo grpo --seed 0       # arm B
    python scripts/train_gvpo.py configs/gvpo_1p5b.json --reward final_reward      # arm C
    ... --hub-repo user/helioguard-gvpo-A-s0   (private; token from HF_TOKEN; use one repo per run)

The steps, group size, learning rate and beta are pre-registered (docs/phase6_gvpo_gates.md); do not change them.
"""
import argparse
import json

from helioguard.gvpo_run import run_training
from verispace_engine.gvpo import load_gvpo_config

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--algo", choices=["gvpo", "grpo"])
    p.add_argument("--reward", choices=["total_return", "final_reward"])
    p.add_argument("--seed", type=int)
    p.add_argument("--init-adapter", help="SFT adapter folder (merged into the base; it is the reference policy)")
    p.add_argument("--checkpoint-dir")
    p.add_argument("--out-dir")
    p.add_argument("--steps", type=int, help="only for smoke tests; the registered run uses the config value")
    p.add_argument("--hub-repo", help="private Hub repo: mirror checkpoints, resume from it (token from HF_TOKEN)")
    a = p.parse_args()
    over = {k: v for k, v in (("algo", a.algo), ("reward", a.reward), ("seed", a.seed), ("init_adapter", a.init_adapter),
                              ("checkpoint_dir", a.checkpoint_dir), ("out_dir", a.out_dir), ("steps", a.steps)) if v is not None}
    hub = None
    if a.hub_repo:
        from verispace_engine.hub import HubSync
        hub = HubSync(a.hub_repo)                 # fails here, not 25 minutes in, if the token or repo is wrong
    s = run_training(load_gvpo_config(a.config, **over), hub=hub)
    print(json.dumps({k: s[k] for k in ("algo", "reward", "training_seed", "steps_done", "steps_total", "resumed_from",
                                        "first_mean_reward", "last_mean_reward", "finished")}, indent=2))
