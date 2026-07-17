# Setup Guide — Windows

Follow top to bottom. Every step says what to run and what you should see.
If something doesn't match, stop there rather than pushing on.

**Where you're headed:**

| Part | You'll have | Time |
|---|---|---|
| A | The project running on your laptop | 20 min |
| B | Code on GitHub | 5 min |
| C | A live public URL | 10 min |
| D | QQP data + a trained model | ~3 hrs |
| E | Live predictions in the app | 15 min |

Parts A–C need no data and no model. You get a working public site first, then
add the model later. That way you find out about problems early.

---

# Part A — Get it running locally

## A1. Install uv

uv is the tool that manages Python and packages for this project.

```cmd
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Now close this window and open a new cmd window.** The installer changes your
PATH and the old window won't see it.

```cmd
uv --version
```

Should print something like `uv 0.11.7`.

> If it says `'uv' is not recognized`, open another new window and try again.

## A2. Download the project files

Download the zip I gave you and extract it (right-click → Extract All).
Leave it in Downloads. You'll copy files *out* of it in step A5.

Don't work inside this folder. It's just a parts bin.

## A3. Create the project folder

Go to where you keep projects, then:

```cmd
cd %USERPROFILE%\Documents
uv init --package --name fairness-gpt2 --python 3.12 fairness-aware-gpt2
cd fairness-aware-gpt2
```

Should print:
```
Initialized project `fairness-gpt2` at `...\fairness-aware-gpt2`
```

This makes an empty skeleton and sets up git for you. It does **not** make the
project — you'll fill it in next.

Why the flags:
- `--package` gives the folder layout Streamlit Cloud needs
- `--name fairness-gpt2` keeps the code's import name short
- `--python 3.12` — **don't skip.** Newer Python versions break this project.

## A4. Replace pyproject.toml

`pyproject.toml` is the file that lists what your project needs. uv wrote a
blank placeholder. Open it in Notepad:

```cmd
notepad pyproject.toml
```

**Delete everything in it**, paste the block below, save, close.

```toml
[project]
name = "fairness-gpt2"
version = "0.1.0"
description = "Fairness-Aware GPT-2 for Paraphrase Detection"
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = { text = "MIT" }
authors = [{ name = "Deonna Owens", email = "deonnao@stanford.edu" }]

dependencies = [
    "streamlit>=1.36",
    "torch>=2.3,<3",
    "transformers>=4.41",
    "huggingface-hub>=0.23",
    "pandas>=2.0",
    "altair>=5.0",
]

[project.scripts]
fairness-train = "fairness_gpt2.train:main"

[dependency-groups]
train = ["datasets>=2.19", "numpy>=1.26", "scikit-learn>=1.4", "tqdm>=4.66"]
dev = ["pytest>=8.0", "ruff>=0.6"]

[build-system]
requires = ["uv_build>=0.11.7,<0.12.0"]
build-backend = "uv_build"

[tool.uv]
default-groups = []

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]
```

## A5. Copy in the code

Point cmd at your extracted download:

```cmd
set SRC=%USERPROFILE%\Downloads\fairness-gpt2-qqp-uv\fairness-gpt2-qqp
dir "%SRC%\streamlit_app.py"
```

Should list one file.

> **File Not Found?** Run this to find it, then fix the `set SRC=` line:
> ```cmd
> dir /s /b %USERPROFILE%\Downloads\streamlit_app.py
> ```
> Use the folder path *without* `\streamlit_app.py` on the end.

Now copy everything across:

```cmd
xcopy "%SRC%\src" src /E /I /Y
xcopy "%SRC%\tests" tests /E /I /Y
xcopy "%SRC%\scripts" scripts /E /I /Y
xcopy "%SRC%\results" results /E /I /Y
xcopy "%SRC%\.streamlit" .streamlit /E /I /Y
xcopy "%SRC%\.github" .github /E /I /Y
copy "%SRC%\streamlit_app.py" . /Y
copy "%SRC%\README.md" . /Y
copy "%SRC%\.gitignore" . /Y
mkdir data
mkdir checkpoints
copy NUL data\.gitkeep
copy NUL checkpoints\.gitkeep
```

**Do not copy pyproject.toml.** The one you pasted in A4 is the correct one.

## A6. Install everything

```cmd
uv lock
```

Should print `Resolved ~90 packages in Xs`. Takes a minute the first time.

```cmd
uv sync --group dev
```

Should print a long list of `+ package==version` lines.

```cmd
uv tree | findstr nvidia
```

**Should print nothing at all.** Nothing is what you want here — it means you
got the small CPU version of PyTorch (~200MB) instead of the huge GPU one
(several GB), which would be too big for Streamlit Cloud.

> If you *do* see `nvidia-...` lines, stop. Something's wrong with
> pyproject.toml.

## A7. Run the tests

```cmd
uv run pytest
```

Should print `31 passed`.

## A8. See it work

```cmd
uv run streamlit run streamlit_app.py
```

A browser opens. Check both:

- **Results** tab — should be full of your paper's numbers and charts.
- **Counterfactual probe** tab — click "Run the probe". You'll see the
  identity-swapped copy with changed words highlighted, plus a yellow warning
  saying no model is configured. **The warning is correct** — there's no
  trained model yet. That comes in Part D.

Press `Ctrl+C` in cmd to stop it.

✅ **Part A done.**

---

# Part B — Put it on GitHub

## B1. Check the lock file exists

```cmd
if exist uv.lock echo FOUND
```

Should print `FOUND`. This file is how Streamlit Cloud installs your project.
Without it, deploying fails.

## B2. Open in GitHub Desktop

1. Open **GitHub Desktop**
2. **File → Add local repository**
3. Choose your `fairness-aware-gpt2` folder
4. Click **Add repository**

It works straight away because `uv init` already set up git in A3.

## B3. Check what's being added

Look at the file list on the left. You should see about **22 files**.

**Must be there:**
- `uv.lock`
- `pyproject.toml`
- `streamlit_app.py`
- the `src\fairness_gpt2\` files

**Must NOT be there:**
- `.venv` (anything at all)
- anything ending `.bin` or `.safetensors`

> If you see `.venv` in the list, stop — `.gitignore` didn't copy properly in
> A5. Re-run the `copy "%SRC%\.gitignore" . /Y` line.

## B4. Commit

At the bottom left:
- **Summary:** `Fairness-Aware GPT-2 for Paraphrase Detection`
- Click **Commit to main**

## B5. Publish

1. Click **Publish repository** at the top
2. **Name:** `fairness-aware-gpt2`
3. **Untick "Keep this code private"** — it must be public for free Streamlit Cloud
4. Click **Publish repository**

## B6. Check it worked

Open your repo on github.com.

- Confirm `uv.lock` is in the file list
- Click the **Actions** tab

A check should be running. Wait ~2 minutes for a green tick. This runs your
tests automatically and proves the lock file is valid — which means Streamlit
Cloud will be able to install it.

✅ **Part B done.**

---

# Part C — Deploy

## C1. Create the app

1. Go to **share.streamlit.io**
2. Click **Create app** → **Deploy a public app from GitHub**
3. Fill in:
   - **Repository:** `<your-username>/fairness-aware-gpt2`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** pick anything free

## C2. Set the Python version — important

Click **Advanced settings** and set **Python version** to **3.12**.

If you skip this, Streamlit may use 3.13 and the build will fail with a
confusing error. Your project needs 3.12.

## C3. Deploy

Click **Deploy**. First build takes ~5 minutes — it's downloading PyTorch.

You'll see logs scrolling. Look for it finding `uv.lock` and running `uv sync`.

## C4. Check it

Your app should load with a public URL you can share.

Same as A8: **Results** tab full of numbers, **probe** tab showing the swap
with a yellow "no model" warning.

✅ **Part C done. You now have a live site.**

---

# Part D — Data and training

Training needs a GPU. Your laptop and Streamlit Cloud don't have one, so use
Google Colab (free).

## D1. Get the data

You can do this on your laptop:

```cmd
uv sync --group train
uv run scripts\download_data.py
```

Downloads all four datasets your paper uses into `data\`:

| Task | Files |
|---|---|
| QQP (paraphrase) | `quora-train.csv`, `quora-dev.csv`, `quora-test-student.csv` |
| SST (sentiment, 5 classes) | `ids-sst-train.csv`, `ids-sst-dev.csv`, `ids-sst-test.csv` |
| CFIMDB (sentiment, 2 classes) | `ids-cfimdb-train.csv`, `ids-cfimdb-dev.csv` |
| Sonnets | `sonnets.txt` |

The QQP dev split is 40,430 — exactly the number in your paper, so those
results will be comparable.

> `data\` is ignored by git on purpose. Don't try to push it.

## D2. Train on Colab

Go to **colab.research.google.com**, new notebook, then
**Runtime → Change runtime type → T4 GPU**.

Paste into a cell and run:

```python
!pip install uv
!git clone https://github.com/<your-username>/fairness-aware-gpt2
%cd fairness-aware-gpt2
!uv sync --group train
!uv pip install torch --torch-backend=auto
!uv run scripts/download_data.py
```

The `--torch-backend=auto` line swaps the CPU PyTorch for the GPU one. Colab
has a GPU; your laptop doesn't.

Then a quick test run first — about 2 minutes:

```python
!uv run fairness-train --mode cda_reg --train data/quora-train.csv \
    --dev data/quora-dev.csv --out /tmp/smoke --epochs 1 \
    --train-subset 2000 --eval-subset 1000
```

If that finishes without errors, run the real thing.

### The three paraphrase models (~3 hours each)

This is your paper's main contribution. `cda_reg` is the one the app uses.

```python
!uv run fairness-train --mode cda_reg --train data/quora-train.csv \
    --dev data/quora-dev.csv --out checkpoints/cda_reg --epochs 10 \
    --lambda-fair 0.5 --save-half
```

For the full Table 2 you need all three. Run them one at a time:

```python
!uv run fairness-train --mode baseline --train data/quora-train.csv \
    --dev data/quora-dev.csv --out checkpoints/baseline --epochs 10
!uv run fairness-train --mode cda --train data/quora-train.csv \
    --dev data/quora-dev.csv --out checkpoints/cda --epochs 10
```

> Keep the Colab tab open — it disconnects when idle. Free Colab may cut you
> off before three runs finish. Do `cda_reg` first so you have something to
> deploy.

`--save-half` halves the model size (~250MB), which matters for Streamlit
Cloud's memory limit.

### The other three tasks (minutes, not hours)

```python
!uv run fairness-sentiment --task sst --train data/ids-sst-train.csv \
    --dev data/ids-sst-dev.csv --out checkpoints/sst --epochs 10
!uv run fairness-sentiment --task cfimdb --train data/ids-cfimdb-train.csv \
    --dev data/ids-cfimdb-dev.csv --out checkpoints/cfimdb --epochs 10
!uv run fairness-sonnet --train data/sonnets.txt --out checkpoints/sonnet --epochs 10
```

### Get your results back

Every run writes a small JSON file into `results/reproduced/`. Those are what
make your dashboard show **your** numbers instead of the paper's. Download them
from Colab:

```python
from google.colab import files
!zip -r reproduced.zip results/reproduced
files.download("reproduced.zip")
```

Unzip into your project's `results\reproduced\` folder, then commit and push
in GitHub Desktop. Your live app will show a reproduced column next to the
paper's numbers.

✅ **Part D done.**

---

# Part E — Connect the model

The model is too big for GitHub, so it goes on Hugging Face and the app
downloads it when it starts.

## E1. Upload it (in Colab, same notebook)

```python
!uv run huggingface-cli login
```

Paste an access token from **huggingface.co/settings/tokens** (make a new one
with **Write** access).

```python
!uv run scripts/push_to_hub.py --ckpt checkpoints/cda_reg \
    --repo <your-hf-username>/fairness-gpt2-qqp
```

## E2. Tell your app where it is

1. Go to **share.streamlit.io** and open your app
2. **Settings → Secrets**
3. Paste this, with your username:

```toml
MODEL_REPO = "<your-hf-username>/fairness-gpt2-qqp"
```

4. Save. The app restarts on its own.

## E3. Check it

Open your app, go to **Counterfactual probe**, click **Run the probe**.

The yellow warning should be gone. You should now see two real predictions
side by side — the original and the identity-swapped version — and a green
"prediction held" or red "prediction flipped" verdict.

✅ **Done.** Your project is live.

---

# Quick reference

Run these from your project folder.

| What you want | Command |
|---|---|
| Run the app | `uv run streamlit run streamlit_app.py` |
| Run tests | `uv run pytest` |
| Check code style | `uv run ruff check .` |
| Add a package | `uv add <name>` |
| Get the data | `uv run scripts\download_data.py` |
| Check PyTorch is the small one | `uv tree \| findstr nvidia` (want no output) |

After any change: commit and push in GitHub Desktop. Streamlit Cloud
redeploys automatically.

**If you change pyproject.toml, run `uv lock` and commit the new `uv.lock`**,
or the deploy will fail.
