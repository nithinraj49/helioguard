"""SFT (LoRA) on the expert traces. Resumable: run the same command again to continue.

    python scripts/train_sft.py configs/sft_smoke_0p5b.json          # CPU smoke test
    python scripts/train_sft.py configs/sft_1p5b.json                # the real run (Colab T4)
    python scripts/train_sft.py configs/sft_1p5b.json --checkpoint-dir /content/drive/MyDrive/helioguard/ckpt

The config file sets the model, LoRA rank, lr, epochs and seed (see verispace_engine.sft.SFTConfig).
"""
import argparse
import json

from verispace_engine.sft import load_config, train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--checkpoint-dir")
    p.add_argument("--out-dir")
    p.add_argument("--data")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--hub-repo", help="private Hub repo (user/name): mirror checkpoints there and resume from it; "
                                      "token from the HF_TOKEN environment variable")
    a = p.parse_args()
    over = {k: v for k, v in (("checkpoint_dir", a.checkpoint_dir), ("out_dir", a.out_dir), ("data", a.data),
                              ("max_steps", a.max_steps)) if v is not None}
    hub = None
    if a.hub_repo:
        from verispace_engine.hub import HubSync
        hub = HubSync(a.hub_repo)                 # fails here, not 25 minutes in, if the token or repo is wrong
    summary = train(load_config(a.config, **over), hub=hub)
    print(json.dumps({k: summary[k] for k in ("steps_done", "steps_total", "resumed_from", "examples",
                                              "skipped_too_long", "first_loss", "final_loss", "finished")}, indent=2))
