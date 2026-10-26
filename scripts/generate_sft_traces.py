"""Generate SFT traces from the tool-informed expert on TRAIN tasks only (CPU, deterministic).

    python scripts/generate_sft_traces.py [--threshold 0.9] [--seed 0] [--workers 4]

Writes data/sft/train_traces.jsonl (git-ignored, regenerated in minutes) and data/sft/data_card.json
(committed). Messages are exactly what the harness shows a model; see helioguard/sft_traces.py.
"""
import argparse
import json
from pathlib import Path

from helioguard.sft_traces import THRESHOLD, data_card, generate_traces, train_tasks, write_jsonl

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--seed", type=int, default=0, help="recorded in the data card; generation is deterministic")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default="data/sft")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct", help="for token counts in the card; '' skips")
    a = p.parse_args()

    tasks = train_tasks()
    kept, everything = generate_traces(tasks, a.threshold, a.workers)
    counts = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        counts = [len(tok.apply_chat_template(t["messages"], tokenize=True, return_dict=True)["input_ids"]) for t in kept]
    out = Path(a.out)
    write_jsonl(out / "train_traces.jsonl", kept)
    card = data_card(kept, everything, a.threshold, a.seed, counts)
    (out / "data_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))
