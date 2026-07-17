"""Download every dataset the report uses and write them into data/.

    uv sync --group train
    uv run scripts/download_data.py              # all four
    uv run scripts/download_data.py --only qqp   # just one

Sources (the CS224N starter splits aren't public, so these are the closest
public equivalents):

    qqp     nyu-mll/glue "qqp"                      -> quora-{train,dev,test-student}.csv
    sst     SetFit/sst5                             -> ids-sst-{train,dev,test}.csv
    cfimdb  tasksource/counterfactually-augmented-imdb -> ids-cfimdb-{train,dev,test}.csv
    sonnet  Shakespeare's sonnets (public domain)   -> sonnets.txt

    uv sync --group train
    uv run scripts/download_data.py

GLUE's QQP splits:
    train       363,846 pairs (labelled)
    validation   40,430 pairs (labelled)  <- the report's dev set size, exactly
    test        390,965 pairs (labels withheld, all -1)

The report used a 283,011-pair train split from the CS224N starter code, which
is a subset of GLUE's train. Pass --train-size 283011 to match that count; note
this samples randomly rather than reproducing the exact starter-code split, so
train-set membership will differ even though the size matches. The dev split
needs no such treatment — it lines up exactly at 40,430.
"""

from __future__ import annotations

import argparse
import csv
import os
import random

from datasets import load_dataset

SEED = 11711


def write_split(rows, path: str, labelled: bool = True) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence1", "sentence2", "is_duplicate"])
        for r in rows:
            s1 = (r["question1"] or "").strip().replace("\n", " ")
            s2 = (r["question2"] or "").strip().replace("\n", " ")
            if not s1 or not s2:
                continue
            label = r["label"] if labelled else -1
            w.writerow([r["idx"], s1, s2, label])
            n += 1
    return n


def write_sentiment(rows, path: str, text_key: str, label_key: str | None) -> int:
    """Write a single-sentence sentiment CSV (id, sentence, label)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "label"])
        for i, r in enumerate(rows):
            text = (r.get(text_key) or "").strip().replace("\n", " ")
            if not text:
                continue
            label = r.get(label_key, -1) if label_key else -1
            w.writerow([i, text, label])
            n += 1
    return n


def fetch_sst(out_dir: str) -> None:
    """SST-5: fine-grained sentiment, labels 0-4 (very neg .. very pos)."""
    print("\nDownloading SST-5 (SetFit/sst5)...")
    ds = load_dataset("SetFit/sst5")
    for split, fname in [
        ("train", "ids-sst-train.csv"),
        ("validation", "ids-sst-dev.csv"),
        ("test", "ids-sst-test.csv"),
    ]:
        n = write_sentiment(ds[split], os.path.join(out_dir, fname), "text", "label")
        print(f"  {fname}  {n:,} sentences")


def fetch_cfimdb(out_dir: str) -> None:
    """Counterfactual IMDB: binary sentiment over human-revised reviews.

    Note: Kaushik et al.'s revisions flip SENTIMENT, not demographic identity.
    The report's Section 5.1 describes this dataset incorrectly.
    """
    print("\nDownloading CFIMDB (tasksource/counterfactually-augmented-imdb)...")
    ds = load_dataset("tasksource/counterfactually-augmented-imdb")
    text_key = "text" if "text" in ds["train"].column_names else "Text"
    label_key = "label" if "label" in ds["train"].column_names else "Sentiment"
    splits = {"train": "ids-cfimdb-train.csv"}
    if "validation" in ds:
        splits["validation"] = "ids-cfimdb-dev.csv"
    if "test" in ds:
        splits["test"] = "ids-cfimdb-test.csv"
    for split, fname in splits.items():
        rows = [dict(r) for r in ds[split]]
        for r in rows:  # normalise string labels if present
            v = r.get(label_key)
            if isinstance(v, str):
                r[label_key] = 1 if v.strip().lower() in ("positive", "pos", "1") else 0
        n = write_sentiment(rows, os.path.join(out_dir, fname), text_key, label_key)
        print(f"  {fname}  {n:,} reviews")


# Shakespeare's sonnets are public domain. Project Gutenberg blocks unknown
# user-agents (urllib gets a block page, not the book), so the GITenberg mirror
# on GitHub is the primary source — same Project Gutenberg etext #1041, no
# blocking. Gutenberg itself is kept as a fallback, with a browser user-agent.
SONNET_SOURCES = [
    "https://raw.githubusercontent.com/GITenberg/Shakespeare-s-Sonnets_1041/master/1041.txt",
    "https://www.gutenberg.org/cache/epub/1041/pg1041.txt",
    "https://www.gutenberg.org/files/1041/1041.txt",
]
EXPECTED_SONNETS = 154
_UA = "Mozilla/5.0 (compatible; fairness-gpt2/0.2; +https://github.com/)"


def _strip_gutenberg_boilerplate(raw: str) -> str:
    """Drop the PG header and licence footer, whichever marker style is used."""
    for marker in ("*** START OF", "***START OF"):
        i = raw.find(marker)
        if i != -1:
            raw = raw[raw.find("\n", i) + 1 :]
            break
    for marker in ("*** END OF", "***END OF"):
        i = raw.rfind(marker)
        if i != -1:
            raw = raw[:i]
            break
    return raw.strip()


def fetch_sonnets(out_dir: str) -> None:
    """Download Shakespeare's sonnets and verify we actually got the poems.

    Writing whatever the server returned without checking is how you end up with
    a 0-sonnet corpus and a confusing training crash three steps later.
    """
    import urllib.error
    import urllib.request

    from fairness_gpt2.train_sonnet import parse_sonnets

    print("\nDownloading Shakespeare's sonnets...")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sonnets.txt")
    tmp = path + ".tmp"

    for url in SONNET_SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  {url.split('/')[2]}: unreachable ({e})")
            continue

        text = _strip_gutenberg_boilerplate(raw)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        n = len(parse_sonnets(tmp))

        if n >= EXPECTED_SONNETS - 5:
            os.replace(tmp, path)
            print(f"  sonnets.txt  {n} sonnets parsed  (from {url.split('/')[2]})")
            return

        print(
            f"  {url.split('/')[2]}: got {n} sonnets, expected ~{EXPECTED_SONNETS} — "
            f"response starts {text[:60]!r}"
        )

    if os.path.exists(tmp):
        os.remove(tmp)
    raise SystemExit(
        "Could not download a usable sonnet corpus from any source.\n"
        "Download https://www.gutenberg.org/cache/epub/1041/pg1041.txt in a browser,\n"
        f"save it as {path}, and re-run with --only qqp --only sst --only cfimdb to skip this step."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        choices=["qqp", "sst", "cfimdb", "sonnet"],
        action="append",
        help="Fetch only these datasets (repeatable). Default: all four.",
    )
    ap.add_argument("--out-dir", default="data")
    ap.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="Randomly subsample the train split to N pairs (0 = keep all 363,846). "
        "Use 283011 to match the pair count reported in the paper.",
    )
    ap.add_argument("--skip-test", action="store_true", help="Skip the unlabelled test split.")
    args = ap.parse_args()

    wanted = set(args.only) if args.only else {"qqp", "sst", "cfimdb", "sonnet"}

    if "sst" in wanted:
        fetch_sst(args.out_dir)
    if "cfimdb" in wanted:
        fetch_cfimdb(args.out_dir)
    if "sonnet" in wanted:
        fetch_sonnets(args.out_dir)
    if "qqp" not in wanted:
        print("\nDone.")
        return

    print("\nDownloading GLUE QQP (~40MB, cached after the first run)...")
    ds = load_dataset("nyu-mll/glue", "qqp")

    train = list(ds["train"])
    if args.train_size and args.train_size < len(train):
        random.Random(SEED).shuffle(train)
        train = train[: args.train_size]
        print(f"Subsampled train to {len(train):,} pairs (seed {SEED})")

    n_train = write_split(train, os.path.join(args.out_dir, "quora-train.csv"))
    n_dev = write_split(ds["validation"], os.path.join(args.out_dir, "quora-dev.csv"))
    print(f"quora-train.csv  {n_train:,} pairs")
    print(f"quora-dev.csv    {n_dev:,} pairs")

    if not args.skip_test:
        n_test = write_split(
            ds["test"], os.path.join(args.out_dir, "quora-test-student.csv"), labelled=False
        )
        print(f"quora-test-student.csv  {n_test:,} pairs (labels withheld)")

    if n_dev != 40430:
        print(f"\nHeads up: dev is {n_dev:,}, not the expected 40,430 — blank rows were dropped.")


if __name__ == "__main__":
    main()
