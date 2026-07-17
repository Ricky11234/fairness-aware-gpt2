"""QQP dataset, counterfactual augmentation, and collation (Sections 4.4, 5.1)."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .identity import contains_identity, swap_identity
from .model import PROMPT_TEMPLATE, encode_single


@dataclass
class Pair:
    s1: str
    s2: str
    label: int
    pid: str = ""


def load_qqp(path: str) -> list[Pair]:
    """Read a CS224N-style QQP TSV/CSV.

    Accepts the starter-code format (columns: id, sentence1, sentence2,
    is_duplicate) and the raw Quora release (question1, question2,
    is_duplicate). Test files with no label column get label -1.
    """
    rows: list[Pair] = []
    delim = "\t" if path.endswith((".tsv", ".txt")) else ","
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        def pick(*names: str) -> str | None:
            for n in names:
                if n in fields:
                    return fields[n]
            return None

        c1 = pick("sentence1", "question1", "q1")
        c2 = pick("sentence2", "question2", "q2")
        cl = pick("is_duplicate", "label", "gold_label")
        cid = pick("id", "qid", "pair_id")
        if not (c1 and c2):
            raise ValueError(f"Could not find question columns in {path}: {reader.fieldnames}")

        for i, row in enumerate(reader):
            s1, s2 = (row[c1] or "").strip(), (row[c2] or "").strip()
            if not s1 or not s2:
                continue
            label = int(float(row[cl])) if cl and row.get(cl) not in (None, "") else -1
            rows.append(Pair(s1, s2, label, row[cid] if cid else str(i)))
    return rows


class QQPDataset(Dataset):
    """Paraphrase pairs, optionally with stochastic CDA applied on the fly.

    When ``cda_prob > 0`` each identity token is swapped with that probability
    (Section 4.4: "identity substitutions are applied with probability 0.5").
    When ``return_counterfactual`` is set, each item also carries the
    deterministic identity-swapped view needed for the fairness loss.
    """

    def __init__(
        self,
        pairs: list[Pair],
        cda_prob: float = 0.0,
        return_counterfactual: bool = False,
        seed: int = 11711,
    ):
        self.pairs = pairs
        self.cda_prob = cda_prob
        self.return_counterfactual = return_counterfactual
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        p = self.pairs[i]
        s1, s2 = p.s1, p.s2
        if self.cda_prob > 0:
            s1 = swap_identity(s1, p=self.cda_prob, rng=self.rng)
            s2 = swap_identity(s2, p=self.cda_prob, rng=self.rng)

        item = {"s1": s1, "s2": s2, "label": p.label}
        if self.return_counterfactual:
            item["cf_s1"] = swap_identity(s1, p=1.0)
            item["cf_s2"] = swap_identity(s2, p=1.0)
        return item


def build_counterfactual_pool(pairs: list[Pair], n: int = 1000, seed: int = 11711) -> list[Pair]:
    """Sample ``n`` identity-bearing pairs and return their swapped versions.

    Section 5.3: "counterfactual data augmentation with 1,000 generated
    counterfactual pairs per epoch".
    """
    rng = random.Random(seed)
    candidates = [p for p in pairs if contains_identity(p.s1) or contains_identity(p.s2)]
    if not candidates:
        return []
    chosen = rng.sample(candidates, min(n, len(candidates)))
    return [Pair(swap_identity(p.s1), swap_identity(p.s2), p.label, f"cf-{p.pid}") for p in chosen]


class Collator:
    def __init__(self, tokenizer, max_length: int = 128, with_cf: bool = False):
        self.tok = tokenizer
        self.max_length = max_length
        self.with_cf = with_cf

    def _encode(self, s1s, s2s):
        prompts = [PROMPT_TEMPLATE.format(s1=a, s2=b) for a, b in zip(s1s, s2s, strict=True)]
        return self.tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, batch):
        enc = self._encode([b["s1"] for b in batch], [b["s2"] for b in batch])
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }
        if self.with_cf:
            cf = self._encode([b["cf_s1"] for b in batch], [b["cf_s2"] for b in batch])
            out["cf_input_ids"] = cf["input_ids"]
            out["cf_attention_mask"] = cf["attention_mask"]
        return out


# --------------------------------------------------------------------------
# Sentiment tasks: SST (5-class) and CFIMDB (binary). Section 5.1.
# Single sentences, no identity swapping — these carry no fairness component
# in the report; they exist to check general fine-tuning behaviour.
# --------------------------------------------------------------------------
@dataclass
class SentimentExample:
    text: str
    label: int
    sid: str = ""


def load_sentiment(path: str) -> list[SentimentExample]:
    """Read a single-sentence sentiment CSV/TSV (columns: id, sentence, label)."""
    rows: list[SentimentExample] = []
    delim = "\t" if path.endswith((".tsv", ".txt")) else ","
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        def pick(*names: str) -> str | None:
            for n in names:
                if n in fields:
                    return fields[n]
            return None

        ct = pick("sentence", "text", "review")
        cl = pick("label", "sentiment", "gold_label")
        cid = pick("id", "idx", "sid")
        if not ct:
            raise ValueError(f"No text column in {path}: {reader.fieldnames}")

        for i, row in enumerate(reader):
            text = (row[ct] or "").strip()
            if not text:
                continue
            label = int(float(row[cl])) if cl and row.get(cl) not in (None, "") else -1
            rows.append(SentimentExample(text, label, row[cid] if cid else str(i)))
    return rows


class SentimentDataset(Dataset):
    def __init__(self, examples: list[SentimentExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        e = self.examples[i]
        return {"text": e.text, "label": e.label}


class SentimentCollator:
    def __init__(self, tokenizer, max_length: int = 128):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        enc = encode_single(self.tok, [b["text"] for b in batch], self.max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }
