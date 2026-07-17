"""Evaluate a checkpoint and write metrics to results/<name>.json.

    uv run scripts/eval_checkpoint.py --ckpt checkpoints/cda_reg \
        --dev data/quora-dev.csv --name cda_reg

Use --name baseline | cda | cda_reg so the dashboard can match runs to the
report's Table 2 rows.

The package is installed into the venv by `uv sync`, so no sys.path juggling.
"""

import argparse
import json
import os

import torch

from fairness_gpt2.data import load_qqp
from fairness_gpt2.evaluate import evaluate
from fairness_gpt2.model import GPT2ParaphraseClassifier, build_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2ParaphraseClassifier.load(args.ckpt, device=device)
    tokenizer = build_tokenizer()
    pairs = load_qqp(args.dev)

    metrics = evaluate(model, tokenizer, pairs, device, batch_size=args.batch_size)
    print(json.dumps(metrics, indent=2))

    os.makedirs("results/reproduced", exist_ok=True)
    out = os.path.join("results", "reproduced", f"{args.name}.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
