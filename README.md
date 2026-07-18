# Twin Test

**Does your question-matching model give the same answer when only the name changes?**

Duplicate-question detection routes support tickets, merges help-centre articles,
and surfaces existing answers in community Q&A. When that model's decision
depends on whether the question says *James* or *Mary*, the same question gets a
different outcome for different people — and aggregate accuracy won't show it.

Twin Test measures it. Paste a pair or upload a CSV; it swaps every name and
gendered word for its counterfactual, re-runs the model, and reports how many
decisions flipped.

```
"Why was he charged twice?"   → duplicate      (p = 0.81)
"Why was she charged twice?"  → not duplicate  (p = 0.43)   ← flip
```

The task is unchanged by the swap: if two questions are duplicates about James,
they're duplicates about Mary — the name appears on both sides and cancels. So a
flip means the identity token moved the decision, not the meaning.

## What's under it

A GPT-2 paraphrase classifier fine-tuned on Quora Question Pairs with two
fairness interventions — counterfactual data augmentation and a symmetric-KL
consistency penalty (λ = 0.5) — plus an audit harness of 102 substitution pairs
(60 gendered names, 20 ethnicity-associated names, 22 pronoun/term swaps).

The classifier isn't the point; 89% on QQP isn't state of the art. **The audit
harness is the point.**

## Replication

The method is replicated from a Stanford CS224N report (cited at the bottom).
Reproducing it from the text alone means reconstructing what the text omits, and
two of those reconstructions became findings:

**1. The paper's subgroup definition is recoverable from its own arithmetic.**
Table 5 reports per-subgroup counts but never defines the subgroups. The counts
define them anyway — they sum to 40,996 against a 40,430-pair dev set:

```
|male ∩ female|             = 40,996 − 40,430      = 566
|male ∪ female|             = 1833 + 1751 − 566    = 3,018
union + name-only + neutral = 3,018 + 734 + 36,678 = 40,430   ← the dev set, exactly
implied n_identity          = 3,018 + 734          = 3,752    ← paper reports 3,751
```

Two independent checks land exact. Male/female are decided by gendered *terms*,
not names; a pair with both counts in both groups; *name-only* means a name with
no gendered term. The intuitive reading (`James` → male) collapses name-only from
734 examples to 14, and the subgroup gap then measures noise on 14 rows.

**2. The flip-rate metric has a grammatical confound.** Table 1 lists `his ↔ hers`
as a flat pair, but English overloads both words:

```
literal:            "improve his credit score"  →  "improve hers credit score"
                    "raise her credit score"    →  "raise him credit score"
resolved by role:   "improve his credit score"  →  "improve her credit score"
```

Ungrammatical text is out-of-distribution for GPT-2, so it may flip because the
sentence broke rather than because the name changed. `swap_identity` resolves
`his`/`her` by syntactic role; `--literal-pronouns` on `eval_checkpoint.py`
reproduces the original mapping so the effect can be measured.

### Results

| Metric | Paper (5 ep) | This replication |
|---|---|---|
| Dev accuracy | 0.8856 | 0.8893 |
| Flip rate | 0.0296 | 0.0159 |
| Identity-bearing dev examples | 3,751 | 3,710 (98.9%) |

Accuracy reproduces. The flip rate lands lower — a ~1% lexicon-coverage
difference can't explain that, so the confound above is the leading candidate.

**Scope: QQP only.** The source report also covers SST, CFIMDB and sonnet
generation as course requirements; those carry no fairness component and are out
of scope. The leaderboard test accuracy (0.876) isn't reproducible — it needs a
submission to a leaderboard that holds the labels.

## Layout

```
pyproject.toml              Project metadata, deps, dependency groups, tooling
uv.lock                     Locked resolution — COMMITTED; Cloud installs from it
.python-version             Pins Python 3.12
streamlit_app.py            Demo: live counterfactual probe + results dashboard
src/fairness_gpt2/
  identity.py               Substitution lexicons, swap logic, subgroup assignment
  model.py                  GPT-2 + linear head on the final non-pad token
  data.py                   QQP loading, CDA, counterfactual pool, collation
  train.py                  Paraphrase training — baseline / cda / cda_reg
  train_sentiment.py        SST (5-class) and CFIMDB (binary)
  train_sonnet.py           Sonnet LM fine-tuning + CHRF
  evaluate.py               Accuracy, subgroup gap, flip rate
  results.py                Reported-vs-reproduced results loading
notebooks/train_qqp_colab.ipynb   Colab notebook — QQP training end to end
tests/                      Lexicon, metric, and data-pipeline tests (no weights needed)
scripts/download_data.py    Fetch QQP from GLUE, write the CSVs the loader wants
scripts/eval_checkpoint.py  Evaluate a checkpoint, dump metrics JSON
scripts/push_to_hub.py      Upload weights to Hugging Face for the deployed app
results/reported.json       Numbers transcribed from the report's tables
results/reproduced/         Written by training/eval runs; the dashboard reads both
.github/workflows/ci.yml    uv-based lint + test on every push
```

Standard src-layout: `uv sync` installs `fairness_gpt2` into the venv, so imports
resolve no matter which directory you run from — including on Streamlit Cloud.

## Setup

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS / Linux
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```bash
git clone https://github.com/<you>/fairness-gpt2-qqp && cd fairness-gpt2-qqp
uv sync                 # runtime deps only — creates .venv, installs the package
uv run streamlit run streamlit_app.py
```

`uv sync` reads `.python-version` and downloads Python 3.12 if needed. No
`pip install`, no `activate`, no `requirements.txt`.

### Dependency groups

`[tool.uv] default-groups = []`, so a bare `uv sync` installs only what the
deployed app needs. Add groups explicitly:

```bash
uv sync --group dev              # + pytest, ruff
uv sync --group train            # + datasets, numpy, scikit-learn, tqdm
uv sync --group dev --group train
```

That's deliberate. Streamlit Cloud runs a bare `uv sync`, and the free tier has
~1GB of RAM — anything in the default groups gets installed into the deployed app.

### Common commands

```bash
uv run pytest                    # tests
uv run ruff check .              # lint
uv run ruff format .             # format
uv add <package>                 # add a dep (updates pyproject + uv.lock)
uv add --group train <package>   # add a training-only dep
uv lock --upgrade                # refresh the lock
uv tree                          # inspect the resolved dependency tree
```

Commit `uv.lock` on every change. It's the file Streamlit Cloud installs from.

### Why torch is pinned to the CPU index

PyPI's Linux `torch` wheel bundles the CUDA runtime — several GB of `nvidia-*`
packages that will blow up a Streamlit Cloud build. So `pyproject.toml` pins
torch to PyTorch's CPU index:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

`explicit = true` scopes that index to torch alone; everything else still comes
from PyPI. CPU is the right default because the deployed app only does
inference. For GPU training, swap the wheel after syncing — uv detects the
driver for you:

```bash
uv sync --group train
uv pip install torch --torch-backend=auto
```

## Replication scope

The report covers four tasks. Only paraphrase detection carries the fairness
contribution; the rest are CS224N framework checks.

| Component | Reported | Reproducible here |
|---|---|---|
| QQP paraphrase + fairness | Table 2 | Yes — but see the lexicon caveat below |
| SST (5-class) | 50.9% | Yes |
| CFIMDB (binary) | 98.4% | Yes |
| Sonnet generation | CHRF 41.294 | Yes |
| QQP leaderboard test acc | 0.876 | **No** — needs a CS224N submission |

**The lexicon caveat.** The report gives substitution *counts* (60 / 20 / 22) and
three examples, not the lists themselves. `identity.py` reconstructs plausible
lists from the cited sources (Bertrand & Mullainathan for the ethnicity-associated
names). A different lexicon means different counterfactuals, so flip rates will
not match the report exactly. If you still have the originals, replace the three
lists in `identity.py` and the numbers should converge.

The dashboard never passes reported numbers off as reproduced ones — it shows
both columns and leaves yours blank until you train.

## Data

If you have the CS224N starter-code splits, drop them in `data/`. Otherwise fetch
public equivalents:

```bash
uv sync --group train
uv run scripts/download_data.py            # all four datasets
uv run scripts/download_data.py --only qqp # or just one
```

| Task | Source | Splits |
|---|---|---|
| QQP | `nyu-mll/glue` (qqp) | 363,846 / 40,430 / 390,965 |
| SST-5 | `SetFit/sst5` | 8,544 / 1,101 / 2,210 |
| CFIMDB | `tasksource/counterfactually-augmented-imdb` | 1,707 / 245 / 488 |
| Sonnets | GITenberg mirror of PG etext #1041 (public domain) | 154 sonnets |

Sonnets come from the GITenberg GitHub mirror rather than gutenberg.org
directly: PG blocks unknown user-agents and serves a block page, which the
downloader would otherwise save as a 0-sonnet corpus. `fetch_sonnets` now
validates that it parsed ~154 sonnets before writing, and falls back to
gutenberg.org with a browser user-agent.

GLUE's QQP validation split is 40,430 pairs — exactly the dev-set size in the
report — so fairness metrics are directly comparable. GLUE's train split is
363,846 against the report's 283,011; pass `--train-size 283011` to match the
count, though it samples randomly rather than reproducing the starter-code split.

The loader accepts both the starter-code column names (`sentence1`, `sentence2`,
`is_duplicate`) and the raw Quora release (`question1`, `question2`). Test files
without labels get `-1`. `data/` is gitignored.

## Training

Each run is roughly 2–3 hours on one GPU. Colab with a T4 is enough; Streamlit
Cloud is not — no GPU, ~1GB RAM, so it only serves inference.

```bash
# Baseline: plain cross-entropy
uv run fairness-train --mode baseline --train data/quora-train.csv \
  --dev data/quora-dev.csv --out checkpoints/baseline --epochs 10

# CDA only
uv run fairness-train --mode cda --train data/quora-train.csv \
  --dev data/quora-dev.csv --out checkpoints/cda --epochs 10

# CDA + fairness regularization (the deployed model)
uv run fairness-train --mode cda_reg --train data/quora-train.csv \
  --dev data/quora-dev.csv --out checkpoints/cda_reg --epochs 10 \
  --lambda-fair 0.5 --save-half
```

Defaults match the paper: AdamW (β = 0.9/0.999, ε = 1e-6), lr 1e-5, batch 16 × 4
accumulation steps = effective 64, seed 11711, AMP on GPU.

Smoke-test the whole loop in a couple of minutes before committing a real run:

```bash
uv run fairness-train --mode cda_reg --train data/quora-train.csv \
  --dev data/quora-dev.csv --out /tmp/smoke --epochs 1 \
  --train-subset 2000 --eval-subset 1000
```

Then evaluate and record metrics:

```bash
uv run scripts/eval_checkpoint.py --ckpt checkpoints/cda_reg \
  --dev data/quora-dev.csv --name cda_reg
```

`--save-half` stores fp16 weights (~250MB instead of ~500MB), which matters for
the memory ceiling on Streamlit Cloud. Weights are cast back to fp32 on load.

### The other three tasks

```bash
# SST — 5-class sentiment
uv run fairness-sentiment --task sst --train data/ids-sst-train.csv \
  --dev data/ids-sst-dev.csv --out checkpoints/sst --epochs 10

# CFIMDB — binary sentiment
uv run fairness-sentiment --task cfimdb --train data/ids-cfimdb-train.csv \
  --dev data/ids-cfimdb-dev.csv --out checkpoints/cfimdb --epochs 10

# Shakespearean sonnets — LM fine-tuning, scored with CHRF
uv run fairness-sonnet --train data/sonnets.txt --out checkpoints/sonnet --epochs 10
```

Each writes `results/reproduced/<task>.json`, which the dashboard picks up
automatically. SST and CFIMDB are small — minutes, not hours.

### On Colab

```python
!pip install uv
!git clone https://github.com/<you>/fairness-gpt2-qqp
%cd fairness-gpt2-qqp
!uv sync --group train && uv pip install torch --torch-backend=auto
!uv run fairness-train --mode cda_reg --train data/quora-train.csv \
    --dev data/quora-dev.csv --out checkpoints/cda_reg --epochs 10 --save-half
```

## Deploying to Streamlit Cloud

Checkpoints are far too large for git, so the app pulls them from the Hugging
Face Hub at startup.

1. **Push the weights.**
   ```bash
   uv run huggingface-cli login
   uv run scripts/push_to_hub.py --ckpt checkpoints/cda_reg --repo <you>/fairness-gpt2-qqp
   ```

2. **Push the code** to a public GitHub repo. Make sure `uv.lock` is committed —
   `.gitignore` excludes `checkpoints/` and `data/` but keeps the lock.

3. **Create the app** at [share.streamlit.io](https://share.streamlit.io) → New
   app → repo, branch `main`, main file `streamlit_app.py`. Under *Advanced
   settings*, **set Python to 3.12** to match `.python-version` and
   `requires-python`.

4. **Add the secret.** Advanced settings → Secrets (or Settings → Secrets after
   deploying):
   ```toml
   MODEL_REPO = "<you>/fairness-gpt2-qqp"
   ```
   See `.streamlit/secrets.toml.example`. Without it the dashboard and swap
   preview still work; only live predictions are disabled.

Cloud detects `uv.lock` and runs `uv sync` — no `requirements.txt` needed. It
uses the **first** dependency file it finds, in this order: `uv.lock`, `Pipfile`,
`environment.yml`, `requirements.txt`, `pyproject.toml` (Poetry). Don't add a
second one; `uv.lock` would win anyway, but you'd get a warning and the two could
drift.

To run locally against a local checkpoint, skip the Hub — the app finds
`checkpoints/cda_reg/` on its own.

### `No module named 'torchvision'` spam on startup

Cosmetic. Streamlit's file watcher walks `sys.modules` and calls
`hasattr(m, "__path__")` on each entry, which trips `transformers`' lazy loader
into importing its vision models — none of which this project uses, and all of
which want torchvision. `.streamlit/config.toml` sets `fileWatcherType = "none"`
to stop it, at the cost of hot-reload (press **R** in the browser to rerun).
Installing torchvision would also silence it, and would undo the install-size
work described above.

### If the app runs out of memory

The free tier is tight for a 124M-parameter model. In order of effectiveness:
train with `--save-half`; confirm the CPU torch index is being used (`uv tree`
should show no `nvidia-*` packages); trim anything unused from `dependencies`. If
it still won't fit, ship the dashboard alone and host inference behind a Hugging
Face Space.

## Known limitations

From the report, plus two the code makes visible:

- Subgroup assignment relies on fixed lexicons and misses implicit cues.
- Swaps assume binary gender and a limited name list.
- **Fixed, and a deviation from the report:** Table 1 lists `his ↔ hers` as a flat
  pair, but `his` and `her` are each two words (possessive determiner vs.
  pronoun/object). The literal mapping yields "hers book" and "him credit score" —
  ungrammatical counterfactuals that confound the flip rate, since the model can
  flip because the syntax broke rather than because the identity changed.
  `swap_identity` now resolves them by syntactic role. Pass
  `contextual_pronouns=False` to reproduce the report's literal mapping.
- Pairs carrying both male and female cues fall into a `mixed` bucket that the
  subgroup gap excludes; the report's four groups don't cover them.
- ~90% of the dev set is identity-free, so headline accuracy is driven by
  examples the interventions never touch.

## Citation

Method replicated from:

> Owens, D. *Fairness-Aware Fine-Tuning of GPT-2 for Paraphrase Detection.*
> Stanford CS224N Default Project.

Also drawing on: Radford et al. 2019 (GPT-2) · Maudslay et al. 2019 (name-based
counterfactual data substitution) · Bertrand & Mullainathan 2004 (audit-study
name lists) · Zhao et al. 2018 (WinoBias) · Dixon et al. 2018 · Zhang et al. 2018
(adversarial debiasing) · Black et al. 2020 (FlipTest) · Kaushik et al. 2020
(counterfactually-augmented data) · Loshchilov & Hutter 2017 (AdamW).
