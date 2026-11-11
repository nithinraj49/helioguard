"""Precheck: how often does a group of 8 sampled attempts have identical rewards? (run on Colab, GPU)

    python scripts/gvpo_precheck.py --adapter /content/drive/MyDrive/helioguard/sft_1p5b/adapter \
        --out /content/drive/MyDrive/helioguard/gvpo_precheck --hub-repo user/helioguard-gvpo-precheck

100 fixed train tasks x 8 attempts, temperature 0.7, SFT model. Resumable. Gate G0 (docs/phase6_gvpo_gates.md, section 4):
at least 40% of groups must have non-identical final rewards.
"""
import argparse
import json

from helioguard.gvpo_precheck import run_precheck, select_tasks
from helioguard.gvpo_run import make_sampler
from helioguard.sft_traces import train_tasks
from verispace_engine.gvpo import GVPOConfig, build_policy

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", required=True, help="the SFT adapter folder")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hub-repo")
    p.add_argument("--upload-every", type=int, default=10)
    a = p.parse_args()
    hub = None
    if a.hub_repo:
        from verispace_engine.hub import HubSync
        hub = HubSync(a.hub_repo)
    cfg = GVPOConfig(model=a.model, init_adapter=a.adapter, group_size=a.attempts, gradient_checkpointing=False,
                     seed=a.seed)
    model, tok, device = build_policy(cfg)           # fresh zero LoRA on the merged SFT model == the SFT policy
    model.eval()
    sampler = make_sampler(cfg)
    ids = select_tasks([t["id"] for t in train_tasks()], a.n, seed=a.seed)

    def sample(task_id, k):
        return sampler(model, tok, device, task_id, k, a.seed * 1000 + ids.index(task_id), cfg)
    print(json.dumps(run_precheck(sample, ids, f"{a.out}/precheck_groups.jsonl", a.attempts, hub, a.upload_every), indent=2))
