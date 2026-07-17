"""Evaluation: accuracy, subgroup accuracy gap, prediction flip rate (Section 5.2)."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import torch
from torch.utils.data import DataLoader

from .data import Collator, Pair, QQPDataset
from .identity import contains_identity, subgroup_of, swap_identity

MIN_SUBGROUP_N = 10  # "Only subgroups containing at least ten examples are considered."


@torch.no_grad()
def predict(
    model,
    tokenizer,
    pairs: list[Pair],
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 128,
) -> torch.Tensor:
    """Return predicted label ids for every pair, in order."""
    ds = QQPDataset(pairs)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer, max_length)
    )
    model.eval()
    preds = []
    for batch in dl:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds)


def subgroup_accuracy_gap(pairs: list[Pair], preds: torch.Tensor) -> dict:
    """max_{i,j} |Acc(g_i) - Acc(g_j)| over subgroups with >= 10 examples."""
    buckets = defaultdict(lambda: {"correct": 0, "n": 0})
    for p, yhat in zip(pairs, preds.tolist(), strict=True):
        g = subgroup_of(p.s1, p.s2)
        if g == "mixed":
            continue  # both male and female cues; not one of the four reported groups
        buckets[g]["n"] += 1
        buckets[g]["correct"] += int(yhat == p.label)

    accs = {g: v["correct"] / v["n"] for g, v in buckets.items() if v["n"] >= MIN_SUBGROUP_N}
    gap = max((abs(a - b) for a, b in combinations(accs.values(), 2)), default=0.0)
    return {
        "gap": gap,
        "per_subgroup": {g: {"acc": accs.get(g), "n": buckets[g]["n"]} for g in buckets},
    }


@torch.no_grad()
def prediction_flip_rate(
    model,
    tokenizer,
    pairs: list[Pair],
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 128,
) -> dict:
    """Fraction of identity-bearing examples whose label changes under a
    deterministic identity swap."""
    idty = [p for p in pairs if contains_identity(p.s1) or contains_identity(p.s2)]
    if not idty:
        return {"flip_rate": 0.0, "flips": 0, "n_identity": 0}

    swapped = [Pair(swap_identity(p.s1), swap_identity(p.s2), p.label, p.pid) for p in idty]
    orig_preds = predict(model, tokenizer, idty, device, batch_size, max_length)
    cf_preds = predict(model, tokenizer, swapped, device, batch_size, max_length)

    flips = int((orig_preds != cf_preds).sum().item())
    return {
        "flip_rate": flips / len(idty),
        "flips": flips,
        "n_identity": len(idty),
    }


def evaluate(
    model,
    tokenizer,
    pairs: list[Pair],
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 128,
) -> dict:
    preds = predict(model, tokenizer, pairs, device, batch_size, max_length)
    labels = torch.tensor([p.label for p in pairs])
    acc = (preds == labels).float().mean().item()

    sub = subgroup_accuracy_gap(pairs, preds)
    flip = prediction_flip_rate(model, tokenizer, pairs, device, batch_size, max_length)

    return {
        "accuracy": acc,
        "subgroup_gap": sub["gap"],
        "per_subgroup": sub["per_subgroup"],
        "flip_rate": flip["flip_rate"],
        "flips": flip["flips"],
        "n_identity": flip["n_identity"],
        "n": len(pairs),
    }
