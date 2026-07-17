"""Shakespearean sonnet generation (Section 5.1, 5.4).

Fine-tunes GPT-2's language-modelling head on a sonnet corpus, then completes
held-out sonnets from their first three lines and scores the completion against
the reference with CHRF (character n-gram F-score).

    uv run fairness-sonnet --train data/sonnets.txt --out checkpoints/sonnet
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import GPT2SonnetLM, build_tokenizer
from .train import SEED, set_seed

PROMPT_LINES = 3  # lines given to the model; it must produce the rest


def parse_sonnets(path: str) -> list[list[str]]:
    """Split a sonnet corpus into a list of sonnets, each a list of lines.

    Handles the common layout where each sonnet is preceded by a numeral
    (roman or arabic) on its own line, and sonnets are blank-line separated.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    blocks = re.split(r"\n\s*\n", raw)
    sonnets: list[list[str]] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        # Drop a leading numeral header if present.
        if lines and re.fullmatch(r"[IVXLCDM\d]+\.?", lines[0], flags=re.IGNORECASE):
            lines = lines[1:]
        if len(lines) >= PROMPT_LINES + 1:
            sonnets.append(lines)
    return sonnets


class SonnetDataset(Dataset):
    def __init__(self, sonnets: list[list[str]], tokenizer, max_length: int = 256):
        self.texts = ["\n".join(s) for s in sonnets]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i]


class SonnetCollator:
    def __init__(self, tokenizer, max_length: int = 256):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        enc = self.tok(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        # Don't train on padding.
        labels[enc["attention_mask"] == 0] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def chrf_score(hypotheses: list[str], references: list[str]) -> float:
    from sacrebleu.metrics import CHRF

    return CHRF().corpus_score(hypotheses, [references]).score


@torch.no_grad()
def evaluate_chrf(model, tokenizer, held_out: list[list[str]], device: str) -> tuple[float, list]:
    hyps, refs, samples = [], [], []
    for sonnet in tqdm(held_out, desc="generating"):
        prompt = "\n".join(sonnet[:PROMPT_LINES]) + "\n"
        reference = "\n".join(sonnet[PROMPT_LINES:])
        full = model.generate(tokenizer, prompt, max_new_tokens=160, device=device)
        completion = full[len(prompt) :].strip()
        hyps.append(completion)
        refs.append(reference)
        if len(samples) < 3:
            samples.append({"prompt": prompt, "generated": completion})
    return chrf_score(hyps, refs), samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Sonnet corpus text file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--held-out", type=int, default=14, help="Sonnets reserved for CHRF")
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sonnets = parse_sonnets(args.train)
    if len(sonnets) <= args.held_out:
        raise SystemExit(f"Only parsed {len(sonnets)} sonnets from {args.train}; need more.")
    train_s, dev_s = sonnets[: -args.held_out], sonnets[-args.held_out :]
    print(f"sonnets: {len(sonnets)} total -> {len(train_s)} train / {len(dev_s)} held out")

    tokenizer = build_tokenizer()
    model = GPT2SonnetLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    collate = SonnetCollator(tokenizer, args.max_length)

    os.makedirs(args.out, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        dl = DataLoader(
            SonnetDataset(train_s, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        model.train()
        running, steps = 0.0, 0
        for batch in tqdm(dl, desc=f"sonnet epoch {epoch}"):
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += out.loss.item()
            steps += 1

        row = {"epoch": epoch, "lm_loss": round(running / max(steps, 1), 4)}
        history.append(row)
        print(json.dumps(row))

    model.save(args.out)
    tokenizer.save_pretrained(args.out)

    chrf, samples = evaluate_chrf(model, tokenizer, dev_s, device)
    print(f"CHRF: {chrf:.3f}")

    os.makedirs("results/reproduced", exist_ok=True)
    with open("results/reproduced/sonnet.json", "w") as f:
        json.dump(
            {
                "task": "sonnet",
                "chrf": chrf,
                "n_held_out": len(dev_s),
                "seed": SEED,
                "history": history,
                "samples": samples,
            },
            f,
            indent=2,
        )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
