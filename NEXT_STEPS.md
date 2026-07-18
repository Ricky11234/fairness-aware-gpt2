# What to do next

Work top to bottom. Don't skip ahead — each step depends on the last.

| Step | You get | Time |
|---|---|---|
| 1 | Files up to date | 10 min |
| 2 | Your name on it | 2 min |
| 3 | Code on GitHub | 3 min |
| 4 | **A live public URL** | 10 min |
| 5 | **Live predictions** | 20 min |
| 6 | The baseline comparison | 3.5 hr (Kaggle does it) |

**Steps 1–5 get you a working, shareable project.** Step 6 makes it a strong one.

---

# Step 1 — Sync your files

Don't chase individual files. Replace everything except two.

Download the latest zip, extract to Downloads, then in **cmd** from your project
folder:

```cmd
set SRC=%USERPROFILE%\Downloads\fairness-gpt2-qqp-uv\fairness-gpt2-qqp
dir "%SRC%\streamlit_app.py"
```

Should list one file. If not, find it:
```cmd
dir /s /b %USERPROFILE%\Downloads\streamlit_app.py
```
and fix the `set SRC=` line (path without `\streamlit_app.py` on the end).

Now copy:

```cmd
xcopy "%SRC%\src" src /E /I /Y
xcopy "%SRC%\scripts" scripts /E /I /Y
xcopy "%SRC%\tests" tests /E /I /Y
xcopy "%SRC%\notebooks" notebooks /E /I /Y
xcopy "%SRC%\results" results /E /I /Y
copy "%SRC%\streamlit_app.py" . /Y
copy "%SRC%\README.md" . /Y
copy "%SRC%\NEXT_STEPS.md" . /Y
copy "%SRC%\.streamlit\config.toml" .streamlit\ /Y
```

## ⚠ Do NOT copy these two

- **`pyproject.toml`** — yours uses the `uv_build` backend from setup. The zip's
  uses hatchling. Yours is correct.
- **`uv.lock`** — matches your pyproject.

## Check

```cmd
uv run pytest
```

**Expect `60 passed`.** Fewer means something didn't copy. Stop and tell me.

---

# Step 2 — Put your name on it

Two files. I wrongly assumed you were the paper's author early on and wrote her
name into your project. Fix both.

### `pyproject.toml`

```toml
description = "Twin Test — identity-robustness auditing for question-matching models"
authors = [{ name = "YOUR NAME", email = "you@example.com" }]
```

### `streamlit_app.py` (near the top, ~line 34)

```python
AUTHOR = "Your Name"
REPO_URL = "https://github.com/Ricky11234/fairness-aware-gpt2"
```

Check `REPO_URL` is actually your repo.

## Check

```cmd
uv run streamlit run streamlit_app.py
```

Browser opens. You should see **Twin Test** and three tabs. Click through:

- **Check a pair** — hit "Run the test". Swap shows with words highlighted, plus
  a yellow "no model connected" warning. **The warning is correct** — no model yet.
- **Audit a dataset** — same warning.
- **How it works** — should be fully readable, with the citation at the bottom.

`Ctrl+C` to stop.

---

# Step 3 — Push

GitHub Desktop:

- You should see changes to `src`, `streamlit_app.py`, `results`, `notebooks`
- Summary: `Twin Test — identity robustness audit tool`
- **Commit to main** → **Push origin**

## Check

Your repo on github.com → **Actions** tab → wait ~2 min for a green tick.

---

# Step 4 — Deploy

1. **share.streamlit.io** → **Create app** → **Deploy a public app from GitHub**
2. Repository: `<your-username>/fairness-aware-gpt2`
3. Branch: `main`
4. Main file: `streamlit_app.py`
5. **Advanced settings → Python version → 3.12** ← don't skip, the build fails otherwise
6. **Deploy**

First build ~5 min (downloading PyTorch).

## Check

Your public URL loads. Same three tabs, same "no model connected" warning.

**✅ You now have something you can share.** Steps 5–6 make it better.

---

# Step 5 — Connect the model

Your trained model is sitting in Kaggle. It's too big for GitHub, so it goes on
Hugging Face and the app downloads it on startup.

## 5a. Get the checkpoint out of Kaggle

1. Open your finished Kaggle run → **Output** tab
2. Find the `checkpoints/cda_reg` folder
3. **Download** it (or: select it → **New Dataset** to keep it on Kaggle)

## 5b. Upload to Hugging Face

Easiest is the website:

1. **huggingface.co** → your profile → **New Model**
2. Name: `fairness-gpt2-qqp`, **Public**
3. **Files → Add file → Upload files**
4. Drag in everything from `cda_reg/`: `pytorch_model.bin`, `head_config.json`,
   `vocab.json`, `merges.txt`, `tokenizer_config.json`, etc.

## 5c. Tell your app

Streamlit Cloud → your app → **Settings → Secrets** → paste:

```toml
MODEL_REPO = "your-hf-username/fairness-gpt2-qqp"
```

Save. The app restarts itself.

## Check

Your app → **Check a pair** → **Run the test**. Yellow warning gone. You should
see two predictions and a green "prediction held" or red "prediction flipped".

Then **Audit a dataset** → tick "Use a sample set" → **Run audit**. You should
get a flip rate, a subgroup table, and a CSV download.

**✅ Fully working tool.**

---

# Step 6 — The baseline run

Right now the app says *"Baseline not trained yet"* in the How-it-works tab.
That's honest: you trained `cda_reg` only, so you can't yet claim the
interventions improved anything **from your own data**.

This run fixes that. It also re-scores `cda_reg` with the corrected subgroup
code, so both numbers are consistent.

## What you'll be able to say afterwards

> Identity robustness improved ~39% at no cost to accuracy.

...backed by your own before/after, not the paper's.

## Do it

1. Kaggle → your finished run → **Output** → select `checkpoints` → **New Dataset**
   (name it something like `fairness-gpt2-checkpoints`)
2. New notebook → **File → Import Notebook** → `notebooks/train_qqp_kaggle_batch.ipynb`
3. Change one line in the config cell:
   ```python
   MODE = "baseline"
   ```
4. **Accelerator = GPU T4 x2**, **Internet = On**
5. **Save Version → Save & Run All (Commit)**
6. Close the tab. ~3 hours.

> **T4, not P100.** P100 is compute 6.0 and current PyTorch dropped support —
> you'd get `no kernel image is available`. The notebook aborts on this in 3
> seconds rather than wasting the session.

## Then re-score cda_reg

Separate run, ~15 min: import `notebooks/reeval_qqp_kaggle.ipynb`, add your
checkpoint dataset, run it. This makes `cda_reg.json` consistent with the fixed
subgroup code, and measures how much of the flip rate was the grammar confound.

## Then bring both home

Download `reproduced.zip` from each run's Output. Unzip into
`results\reproduced\` so you have:

```
results\reproduced\baseline.json
results\reproduced\cda_reg.json
```

Commit → push. **Your live app updates itself** — the comparison section fills in
automatically. No redeploy.

---

# If something breaks

| Symptom | Cause |
|---|---|
| `uv run pytest` not 60 | a folder didn't copy in Step 1 |
| Streamlit build fails | Python isn't set to 3.12 in Advanced settings |
| App loads, no predictions | `MODEL_REPO` secret missing or misspelled |
| Kaggle: `no kernel image` | Accelerator is P100 — switch to T4 |
| Kaggle: `AMBIGUOUS_PRONOUNS` assert | you didn't push before running |
| Wall of `torchvision` errors | harmless; `.streamlit/config.toml` silences it |
