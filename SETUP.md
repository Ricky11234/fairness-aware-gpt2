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

> ### Seeing a wall of `No module named 'torchvision'` tracebacks?
>
> **That's noise, not an error — your app is fine.** Streamlit's hot-reload
> watcher pokes at every loaded module, which makes `transformers` try to
> lazily import ~40 vision models it ships with. Those need `torchvision`,
> which this project doesn't install because it doesn't do images and the
> install has to stay small enough for Streamlit Cloud.
>
> `.streamlit\config.toml` already turns the watcher off (`fileWatcherType =
> "none"`), which silences it. If you copied an older config, add this:
>
> ```toml
> [server]
> fileWatcherType = "none"
> ```
>
> One consequence: the app no longer reloads automatically when you edit a
> file. Press **R** in the browser to rerun.
>
> **Don't "fix" it by installing torchvision.** It's a large dependency you
> don't need, and it puts your Streamlit Cloud build back over the memory
> limit.

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

# Part D — Data and training (QQP only)

**Scope: the Quora paraphrase task.** That's where your paper's fairness
contribution lives. SST, CFIMDB and the sonnets are CS224N framework
requirements with no fairness component — skipping them costs you nothing.

Training needs a GPU. Your laptop and Streamlit Cloud don't have one, so use
Google Colab (free).

## D1. Open the notebook

`notebooks/train_qqp_colab.ipynb` is in your repo. Either:

- Go to **colab.research.google.com** → **GitHub** tab → paste your repo URL, or
- **File → Upload notebook** and pick the file from your project folder.

Then **Runtime → Change runtime type → T4 GPU**. The first cell checks this — if
it says no GPU, stop and fix it. CPU training would take weeks.

## D2. Edit two lines

In the notebook:

- Cell 2: `REPO = "https://github.com/YOUR-USERNAME/fairness-aware-gpt2"`
- Cell 9: `HF_USER = "YOUR-HF-USERNAME"`

## D3. Run the cells in order

| Step | What | Time |
|---|---|---|
| Setup + data | Clone, install, download QQP | ~10 min |
| Smoke test | 1 epoch on 2,000 pairs | ~2 min |
| Train `cda_reg` | The model your app uses | ~2 hr |
| Download results | `reproduced.zip` | instant |

**Don't skip the smoke test.** It catches every setup problem in two minutes
instead of two hours.

**Keep the Colab tab open and visible.** Free Colab kills idle sessions.

## D4. Why 5 epochs, not 10

Your paper's Table 4 reports 5-epoch numbers, so this is a comparison against a
figure you actually published:

| Mode | Dev acc | Subgroup gap | Flip rate |
|---|---|---|---|
| CDA | 0.8835 | 0.0433 | 0.0392 |
| CDA + Reg. | 0.8856 | 0.0512 | 0.0296 |

Ten epochs is double the time and free Colab will probably disconnect first. Do
5 now; you can always run 10 later.

## D5. Bring the results home

The notebook downloads `reproduced.zip`. Unzip it into your project so you have:

```
results\reproduced\cda_reg.json
```

Then commit and push in GitHub Desktop. **Your live app will now show your
numbers next to the paper's.**

## D6. Optional — the other two models

`cda_reg` alone is enough to deploy. Run `baseline` and `cda` only if you want
the full Table 2 comparison. One at a time, re-downloading results after each.

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
