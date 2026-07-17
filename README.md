# Fairness-Aware GPT-2 for Paraphrase Detection

Implementation of the CS224N project: does fairness-aware training reduce
identity-based prediction instability in GPT-2 without costing task accuracy?

Two interventions on top of a GPT-2 paraphrase classifier fine-tuned on Quora
Question Pairs:

- **CDA** — counterfactual data augmentation over 60 gendered name pairs, 20
  ethnicity-associated name pairs, and 22 pronoun/gendered-term swaps.
- **Consistency regularization** — a symmetric-KL penalty (λ = 0.5) between the
  prediction on an example and on its identity-swapped twin.

Reported result: CDA cuts the subgroup accuracy gap from 3.65% → 2.90%;
regularization cuts the prediction flip rate from 4.56% → 2.80% while holding
dev accuracy at 89.56%.

Managed with [uv](https://docs.astral.sh/uv/). Deployed on Streamlit Community
Cloud, which reads `uv.lock` natively.

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
| Sonnets | Project Gutenberg (public domain) | 154 sonnets |

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
- `his → hers` is wrong in possessive-determiner position ("his book" → "hers
  book"). The lexicon has no syntax; fixing it needs POS tagging. There's a test
  pinning this behaviour so it can't change silently.
- Pairs carrying both male and female cues fall into a `mixed` bucket that the
  subgroup gap excludes; the report's four groups don't cover them.
- ~90% of the dev set is identity-free, so headline accuracy is driven by
  examples the interventions never touch.

## References

Radford et al. 2019 (GPT-2) · Maudslay et al. 2019 (name-based CDS) · Zhao et al.
2018 (WinoBias) · Dixon et al. 2018 · Zhang et al. 2018 · Loshchilov & Hutter
2017 (AdamW).
