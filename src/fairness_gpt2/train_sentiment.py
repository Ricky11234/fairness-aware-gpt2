"""Sentiment classification training for SST (5-class) and CFIMDB (binary).

Section 5.1 of the report. These tasks carry no fairness component — they are
the CS224N framework's general fine-tuning checks — so there is no CDA and no
consistency regularizer here, just cross-entropy.

    uv run fairness-sentiment --task sst --train data/ids-sst-train.csv \
        --dev data/ids-sst-dev.csv --out checkpoints/sst
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SentimentCollator, SentimentDataset, load_sentiment
from .model import NUM_LABELS, GPT2Classifier, build_tokenizer
from .train import SEED, set_seed


@torch.no_grad()
def evaluate_sentiment(model, tokenizer, examples, device, batch_size=32, max_length=128):
    dl = DataLoader(
        SentimentDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SentimentCollator(tokenizer, max_length),
    )
    model.eval()
    correct = total = 0
    for batch in dl:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds = logits.argmax(-1).cpu()
        correct += (preds == batch["labels"]).sum().item()
        total += len(preds)
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sst", "cfimdb"], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--save-half", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    train_ex = load_sentiment(args.train)
    dev_ex = load_sentiment(args.dev)
    n_labels = NUM_LABELS[args.task]
    print(f"task={args.task}  labels={n_labels}  train={len(train_ex)}  dev={len(dev_ex)}")

    tokenizer = build_tokenizer()
    model = GPT2Classifier(num_labels=n_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    collate = SentimentCollator(tokenizer, args.max_length)

    os.makedirs(args.out, exist_ok=True)
    history, best = [], 0.0

    for epoch in range(args.epochs):
        dl = DataLoader(
            SentimentDataset(train_ex),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, steps = 0.0, 0

        for i, batch in enumerate(tqdm(dl, desc=f"{args.task} epoch {epoch}")):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = F.cross_entropy(logits, batch["labels"].to(device))
            scaler.scale(loss / args.grad_accum).backward()
            running += loss.item()
            steps += 1

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        acc = evaluate_sentiment(model, tokenizer, dev_ex, device, args.batch_size * 2)
        row = {"epoch": epoch, "loss": round(running / max(steps, 1), 4), "dev_acc": round(acc, 4)}
        history.append(row)
        print(json.dumps(row))

        # Keep the best epoch, not the last — these sets are small and noisy.
        if acc > best:
            best = acc
            model.save(args.out, half=args.save_half)

        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(
                {"task": args.task, "seed": SEED, "best_dev_acc": best, "history": history},
                f,
                indent=2,
            )

    os.makedirs("results/reproduced", exist_ok=True)
    with open(f"results/reproduced/{args.task}.json", "w") as f:
        json.dump({"task": args.task, "dev_accuracy": best, "n_dev": len(dev_ex)}, f, indent=2)

    tokenizer.save_pretrained(args.out)
    print(f"best dev acc {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
