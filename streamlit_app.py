"""Blind Match — identity-robustness auditing for text-similarity / plagiarism detection.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Blind Match · plagiarism-checker fairness audit",
    page_icon="⧉",
    layout="wide",
)

ROOT = Path(__file__).parent

from fairness_gpt2.results import (  # noqa: E402
    intervention_effect,
    load_reported,
    load_reproduced,
)

REPORTED = load_reported()
REPRODUCED = load_reproduced()
EFFECT = intervention_effect(REPRODUCED)

# ---- EDIT THESE ----------------------------------------------------------
AUTHOR = "Abhinav Barman"
REPO_URL = "https://github.com/Ricky11234/fairness-aware-gpt2"
# --------------------------------------------------------------------------

MAX_BATCH = 200


def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)


# cda_reg is the shipped model; baseline is optional (enables side-by-side).
MODEL_REPO = _secret("MODEL_REPO")
BASELINE_REPO = _secret("BASELINE_REPO")
LOCAL_CKPT = ROOT / "checkpoints" / "cda_reg"
LOCAL_BASELINE = ROOT / "checkpoints" / "baseline"

# --------------------------------------------------------------------------
# Design system — aviation ops console.
# Palette: deep slate "night ramp", signage amber, boarding-pass paper,
# a go/hold pair (jade / signal-red) for the verdict. Type: a condensed
# grotesque feel via system stacks, monospace for all machine output.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Barlow+Semi+Condensed:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

      :root {
        --ground:#e9ece4; --panel:#f3f5ef; --line:#cdd4c6;
        --forest:#1f4d3a; --forest-deep:#143528; --forest-soft:#2f6b52;
        --brass:#b08d3f; --brass-dim:#8f7132;
        --paper:#f6f4ec; --ink:#20291f;
        --go:#2f6b52; --go-bg:#e4efe7; --go-line:#bcd6c6;
        --hold:#a23a2e; --hold-bg:#f6e7e3; --hold-line:#e3c1b8;
        --mute:#6a7566;
      }
      .stApp {background:var(--ground);}
      .block-container {padding-top:1.4rem; max-width:1180px;}

      h1, h2, h3 {font-family:'Barlow Semi Condensed',sans-serif !important;
                  letter-spacing:.01em;}
      .stApp {font-family:'Inter',sans-serif;}

      /* Departure-board header */
      .board {
        background:linear-gradient(160deg,var(--forest) 0%,var(--forest-deep) 100%);
        border:1px solid var(--line); border-radius:10px;
        padding:1.4rem 1.6rem; margin-bottom:.5rem;
        box-shadow:0 10px 34px rgba(20,53,40,.22);
      }
      .board .eyebrow {
        font-family:'IBM Plex Mono',monospace; font-size:.72rem;
        letter-spacing:.28em; text-transform:uppercase; color:var(--brass);
        margin-bottom:.5rem;
      }
      .board h1 {
        color:var(--paper); font-size:2.5rem; line-height:1.02;
        margin:0 0 .5rem 0; font-weight:700;
      }
      .board .tag {color:#c3d1c4; font-size:1.02rem; max-width:70ch; margin:0;}
      .board .flip-strip {
        font-family:'IBM Plex Mono',monospace; color:#d9b968;
        font-size:.8rem; margin-top:.9rem; letter-spacing:.04em;
        border-top:1px dashed rgba(214,201,168,.25); padding-top:.8rem;
      }

      /* Mission card — the "what this is / isn't" box */
      .mission {
        background:var(--paper); color:var(--ink);
        border-left:4px solid var(--forest-soft); border-radius:6px;
        padding:1.1rem 1.3rem; margin:.4rem 0 .2rem;
      }
      .mission b {color:var(--forest);}

      /* Boarding-pass pair */
      .pass {
        font-family:'IBM Plex Mono',monospace;
        background:var(--paper); color:var(--ink);
        border:1px solid #d9d3c4; border-radius:8px;
        padding:.7rem .95rem; margin:.3rem 0;
        position:relative;
      }
      .pass .rt {font-size:.62rem; letter-spacing:.2em; text-transform:uppercase;
                 color:#9a8f78; display:block; margin-bottom:.2rem;}
      .swapmark {background:var(--brass); color:#2a2008; padding:0 .2rem;
                 border-radius:3px; font-weight:600;}

      /* Verdict */
      .verdict {font-family:'Barlow Semi Condensed',sans-serif;
                font-size:1.15rem; font-weight:700; padding:.7rem 1rem;
                border-radius:6px; margin-top:.6rem; letter-spacing:.02em;}
      .hold {background:var(--hold-bg); color:var(--hold); border:1px solid var(--hold-line);}
      .go   {background:var(--go-bg);   color:var(--go);   border:1px solid var(--go-line);}

      /* Glossary chips */
      .gl {border-bottom:1px dotted var(--brass-dim); cursor:help;}

      /* Model column headers */
      .modelhdr {font-family:'IBM Plex Mono',monospace; font-size:.72rem;
                 letter-spacing:.16em; text-transform:uppercase;
                 padding:.25rem .5rem; border-radius:4px; display:inline-block;}
      .m-base {background:#dfe4d9; color:#55604f; border:1px solid #c8d0be;}
      .m-reg  {background:var(--forest-deep); color:#d9b968;}


      .stButton>button[kind="primary"] {
        background:var(--forest) !important; border:1px solid var(--forest-deep) !important;
        color:var(--paper) !important; font-family:'Barlow Semi Condensed',sans-serif;
        letter-spacing:.02em; font-weight:600;
      }
      .stButton>button[kind="primary"]:hover {background:var(--forest-soft) !important;}
      section[data-testid="stSidebar"] {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Models. cda_reg always; baseline too when configured -> side-by-side.
# --------------------------------------------------------------------------
def _resolve(repo: str, local: Path):
    if repo:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=repo)
    if local.exists():
        return str(local)
    return None


@st.cache_resource(show_spinner="Loading the fairness-tuned model…")
def load_primary():
    import torch

    from fairness_gpt2.model import GPT2ParaphraseClassifier, build_tokenizer

    d = _resolve(MODEL_REPO, LOCAL_CKPT)
    if not d:
        return None, None
    torch.set_num_threads(2)
    return GPT2ParaphraseClassifier.load(d, device="cpu"), build_tokenizer()


@st.cache_resource(show_spinner="Loading the baseline model…")
def load_baseline():
    import torch

    from fairness_gpt2.model import GPT2ParaphraseClassifier, build_tokenizer

    d = _resolve(BASELINE_REPO, LOCAL_BASELINE)
    if not d:
        return None, None
    torch.set_num_threads(2)
    return GPT2ParaphraseClassifier.load(d, device="cpu"), build_tokenizer()


def dup_prob(model, tokenizer, s1_list, s2_list, batch_size: int = 8):
    import torch

    from fairness_gpt2.model import encode_pairs

    out = []
    for i in range(0, len(s1_list), batch_size):
        enc = encode_pairs(tokenizer, s1_list[i : i + batch_size], s2_list[i : i + batch_size])
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
        out.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return out


def highlight_swaps(original: str, swapped: str) -> str:
    return " ".join(
        f'<span class="swapmark">{b}</span>' if a != b else b
        for a, b in zip(original.split(), swapped.split(), strict=True)
    )


def glossary(term: str, definition: str) -> str:
    return f'<span class="gl" title="{definition}">{term}</span>'


def model_warning():
    st.warning(
        "**No model connected.** Set `MODEL_REPO` in the app's secrets to your "
        "Hugging Face repo id (and `BASELINE_REPO` for the side-by-side "
        "comparison), or drop checkpoints in `checkpoints/`. The identity swap "
        "runs without a model."
    )


# --------------------------------------------------------------------------
# Header — departure board
# --------------------------------------------------------------------------
_flip_line = ""
if EFFECT:
    f = EFFECT["flip_rate"]
    _flip_line = (
        f"FAIRNESS TUNING ▸ IDENTITY-DRIVEN VERDICT FLIPS  {f['baseline']:.1%} → "
        f"{f['cda_reg']:.1%}   ({f['pct']:+.0%})"
    )
else:
    _flip_line = "FAIRNESS TUNING ▸ CONNECT A MODEL TO SEE LIVE RESULTS"

st.markdown(
    f"""
    <div class="board">
      <div class="eyebrow">⧉ Academic Integrity · Text-Similarity Matching · Fairness Audit</div>
      <h1>Blind&nbsp;Match</h1>
      <p class="tag">Plagiarism checkers like Turnitin and Copyscape rest on one
      operation: deciding whether two passages say the same thing. This audits that
      operation for a single failure — does the similarity verdict stay the same when
      only the author's name or gender changes? An identical passage should be judged
      identical whether it is signed <i>James</i> or <i>Mary</i>.</p>
      <div class="flip-strip">{_flip_line}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mission">
      <b>What this tool is for.</b> It does <b>not</b> try to catch plagiarism, and it
      makes no claim about whether two passages are "really" a match. It checks one
      narrower, more important thing: that the similarity verdict does not move when an
      author named <i>James</i> becomes <i>Mary</i>, or <i>he</i> becomes <i>she</i>.
      The same passage should draw the same flag no matter whose name is on it. A
      detector can look accurate on average and still, quietly, flag one student and
      clear an identical submission from another. This surfaces that.
    </div>
    """,
    unsafe_allow_html=True,
)

# Glossary strip — plain definitions, always visible on hover.
g1 = glossary(
    "flip",
    "The similarity verdict changed after only the author's name or gender was swapped. One flip = one student flagged (or cleared) for who they are, not what they wrote.",
)
g2 = glossary(
    "flip rate",
    "Of all passages carrying a name or gendered word, the share whose verdict flipped under the swap. Lower is fairer; 0% = perfectly identity-blind.",
)
g3 = glossary(
    "p",
    "The model's confidence, from 0 to 1, that two passages say the same thing. Above 0.50 = flagged as a match. This is a probability, not a statistical p-value.",
)
g4 = glossary(
    "subgroup",
    "Which identity a passage carries: male / female (from pronouns or gendered words), name-only (a name, no gendered word), or neutral.",
)
st.markdown(
    f"<div style='font-size:.86rem;color:#5a6673;margin:.5rem 0 0;'>"
    f"Hover any term for a plain-English definition: {g1} · {g2} · {g3} · {g4}"
    f"</div>",
    unsafe_allow_html=True,
)

st.write("")
check_tab, audit_tab, how_tab = st.tabs(
    ["  Check one message  ", "  Audit a batch  ", "  How it works  "]
)


# --------------------------------------------------------------------------
# Tab 1 — single message, both models side by side
# --------------------------------------------------------------------------
with check_tab:
    from fairness_gpt2.identity import contains_identity, subgroup_of, swap_identity

    st.markdown(
        "Enter two passages — say, a student submission and a source. The tool swaps "
        "the author's identity in both, then asks each model — the plain **baseline** "
        "and our **fairness-tuned** model — whether they match, before and after the "
        "swap. A "
        f"{glossary('flip', 'verdict changed after only the identity changed')} is a "
        "passage judged differently for a different author.",
        unsafe_allow_html=True,
    )

    examples = {
        "Essay vs. source (paraphrased)": (
            "In his essay, James argues the revolution was driven mainly by economic grievance.",
            "James claims economic hardship was the primary force behind the uprising.",
        ),
        "Two student submissions": (
            "Mary explains that photosynthesis converts light energy into chemical energy in plants.",
            "According to Mary, plants turn light into chemical energy through photosynthesis.",
        ),
        "Lab report similarity": (
            "Connor concludes his results confirm the hypothesis within the margin of error.",
            "Connor states the findings support the hypothesis, allowing for experimental error.",
        ),
    }
    choice = st.selectbox("Start from a real integrity-check scenario", list(examples) + ["Blank"])
    d1, d2 = examples.get(choice, ("", ""))

    c1, c2 = st.columns(2)
    q1 = c1.text_area("Passage 1", value=d1, height=90)
    q2 = c2.text_area("Passage 2", value=d2, height=90)

    if st.button("Run the fairness check", type="primary"):
        if not q1.strip() or not q2.strip():
            st.warning("Enter both passages.")
        else:
            cf1, cf2 = swap_identity(q1), swap_identity(q2)

            st.markdown("###### Original vs identity-swapped")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(
                    f'<div class="pass"><span class="rt">As written</span>{q1}</div>'
                    f'<div class="pass"><span class="rt">As written</span>{q2}</div>',
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    f'<div class="pass"><span class="rt">Identity swapped</span>{highlight_swaps(q1, cf1)}</div>'
                    f'<div class="pass"><span class="rt">Identity swapped</span>{highlight_swaps(q2, cf2)}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem;"
                f"color:#5a6673;margin:.3rem 0 .8rem;'>subgroup: "
                f"<b>{subgroup_of(q1, q2)}</b> → <b>{subgroup_of(cf1, cf2)}</b></div>",
                unsafe_allow_html=True,
            )

            if not contains_identity(q1 + " " + q2):
                st.info(
                    "No name or gendered word here, so the swap changes nothing and "
                    "nothing can flip. Try a message with a name or a pronoun."
                )

            reg_model, reg_tok = load_primary()
            base_model, base_tok = load_baseline()

            if reg_model is None:
                model_warning()
            else:

                def verdict_block(label, css, model, tok):
                    po, pc = dup_prob(model, tok, [q1, cf1], [q2, cf2])
                    yo, yc = int(po > 0.5), int(pc > 0.5)
                    flipped = yo != yc
                    st.markdown(
                        f'<span class="modelhdr {css}">{label}</span>',
                        unsafe_allow_html=True,
                    )
                    m1, m2 = st.columns(2)
                    m1.metric("As written", "Duplicate" if yo else "Not duplicate", f"p = {po:.3f}")
                    m2.metric("After swap", "Duplicate" if yc else "Not duplicate", f"p = {pc:.3f}")
                    st.markdown(
                        f'<div class="verdict {"hold" if flipped else "go"}">'
                        + (
                            "⚠ FLIPPED — same request, different verdict after the identity changed."
                            if flipped
                            else f"✓ HELD — verdict unchanged (confidence moved {abs(po - pc):.3f})."
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    return flipped

                if base_model is not None:
                    colA, colB = st.columns(2)
                    with colA:
                        fb = verdict_block(
                            "Baseline · no fairness tuning", "m-base", base_model, base_tok
                        )
                    with colB:
                        fr = verdict_block(
                            "Fairness-tuned · CDA + regularization", "m-reg", reg_model, reg_tok
                        )
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    if fb and not fr:
                        st.success(
                            "**This is the whole point.** The baseline flipped its verdict on "
                            "the identity swap; the fairness-tuned model held. Same text, "
                            "and only our model judges it consistently regardless of author."
                        )
                    elif fb and fr:
                        st.info(
                            "Both models flipped here — a hard case even for the tuned model. "
                            "Across the full dev set the tuned model flips far less often "
                            "(see *How it works*)."
                        )
                    elif not fb and not fr:
                        st.info(
                            "Both models held on this one. Try the *Audit a batch* tab to see "
                            "the difference at scale, where the baseline's flips add up."
                        )
                else:
                    verdict_block(
                        "Fairness-tuned · CDA + regularization", "m-reg", reg_model, reg_tok
                    )
                    st.caption(
                        "Showing the fairness-tuned model only. Set `BASELINE_REPO` in secrets "
                        "to compare it side by side with the untuned baseline."
                    )


with audit_tab:
    from fairness_gpt2.identity import contains_identity, subgroup_of, swap_identity

    st.write(
        "Upload a batch of passage pairs and get a flip rate for the set: how often the "
        "decision changes under identity substitution. Rows without a name or "
        "gendered word are reported but can't flip, so they're excluded from the rate."
    )
    st.caption(
        f"CSV with columns `question1`, `question2` (or `sentence1`, `sentence2`). "
        f"Capped at {MAX_BATCH} rows — this runs on a free CPU instance."
    )

    SAMPLE = pd.DataFrame(
        [
            (
                "In his thesis, James argues the war was caused chiefly by economic strain.",
                "James contends economic pressure was the main cause of the war.",
            ),
            (
                "Mary writes that supply and demand set prices in a free market.",
                "According to Mary, prices in a free market are set by supply and demand.",
            ),
            (
                "Connor's report states the reaction is exothermic and releases heat.",
                "Connor notes the reaction gives off heat, making it exothermic.",
            ),
            (
                "She concludes that the sample size was too small to be significant.",
                "Her conclusion is that too few samples were used for significance.",
            ),
            (
                "The paper defines recursion as a function that calls itself.",
                "Recursion is described as a function invoking itself within the text.",
            ),
            (
                "James summarises that the poem's central theme is loss.",
                "In James's reading, loss is the poem's main theme.",
            ),
            (
                "Photosynthesis converts sunlight into chemical energy in the cell.",
                "In the cell, sunlight is turned into chemical energy by photosynthesis.",
            ),
            (
                "He asserts the experiment confirmed the original hypothesis.",
                "According to him, the results backed up the initial hypothesis.",
            ),
        ],
        columns=["question1", "question2"],
    )

    up = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use a sample set instead", value=up is None)

    df = None
    if up is not None and not use_sample:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
    elif use_sample:
        df = SAMPLE.copy()

    if df is not None:
        cols = {c.lower().strip(): c for c in df.columns}
        c1 = cols.get("question1") or cols.get("sentence1") or cols.get("q1")
        c2 = cols.get("question2") or cols.get("sentence2") or cols.get("q2")

        if not (c1 and c2):
            st.error(f"Need question1/question2 columns. Found: {list(df.columns)}")
        else:
            if len(df) > MAX_BATCH:
                st.info(f"{len(df):,} rows uploaded — auditing the first {MAX_BATCH}.")
                df = df.head(MAX_BATCH)
            st.dataframe(df[[c1, c2]].head(5), use_container_width=True, hide_index=True)

            if st.button("Run audit", type="primary"):
                model, tokenizer = load_primary()
                if model is None:
                    model_warning()
                else:
                    s1 = df[c1].astype(str).tolist()
                    s2 = df[c2].astype(str).tolist()
                    cf1 = [swap_identity(s) for s in s1]
                    cf2 = [swap_identity(s) for s in s2]

                    with st.spinner(f"Scoring {len(s1) * 2} pairs on CPU…"):
                        p_orig = dup_prob(model, tokenizer, s1, s2)
                        p_cf = dup_prob(model, tokenizer, cf1, cf2)

                    out = pd.DataFrame(
                        {
                            "question1": s1,
                            "question2": s2,
                            "has_identity": [
                                contains_identity(a + " " + b) for a, b in zip(s1, s2, strict=True)
                            ],
                            "subgroup": [subgroup_of(a, b) for a, b in zip(s1, s2, strict=True)],
                            "p_duplicate": p_orig,
                            "p_duplicate_swapped": p_cf,
                        }
                    )
                    out["prediction"] = (out.p_duplicate > 0.5).map({True: "dup", False: "not dup"})
                    out["prediction_swapped"] = (out.p_duplicate_swapped > 0.5).map(
                        {True: "dup", False: "not dup"}
                    )
                    out["flipped"] = out.prediction != out.prediction_swapped
                    out["prob_shift"] = (out.p_duplicate - out.p_duplicate_swapped).abs()

                    testable = out[out.has_identity]
                    n_flip = int(testable.flipped.sum())
                    rate = n_flip / len(testable) if len(testable) else 0.0

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Rows audited", f"{len(out):,}")
                    k2.metric(
                        "Testable (contain identity)",
                        f"{len(testable):,}",
                        f"{len(out) - len(testable)} can't flip",
                        delta_color="off",
                    )
                    k3.metric(
                        "Flip rate", f"{rate:.1%}", f"{n_flip} flipped", delta_color="inverse"
                    )

                    if len(testable) == 0:
                        st.info(
                            "No rows contain a name or gendered word, so nothing is testable. "
                            "This tool can only measure what it can substitute."
                        )
                    else:
                        if n_flip:
                            st.error(
                                f"**{n_flip} of {len(testable)} verdicts depend on the identity "
                                "token.** Those passages would be judged differently for different "
                                "authors of the same text."
                            )
                            st.markdown("##### Failing rows")
                            st.dataframe(
                                testable[testable.flipped][
                                    [
                                        "question1",
                                        "question2",
                                        "subgroup",
                                        "prediction",
                                        "prediction_swapped",
                                        "prob_shift",
                                    ]
                                ].style.format({"prob_shift": "{:.3f}"}),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.success(
                                f"**No flips across {len(testable)} testable rows.** No decision "
                                "changed under identity substitution."
                            )

                        st.markdown("##### Where the instability sits")
                        by_group = (
                            testable.groupby("subgroup")
                            .agg(
                                rows=("flipped", "size"),
                                flips=("flipped", "sum"),
                                mean_prob_shift=("prob_shift", "mean"),
                            )
                            .reset_index()
                        )
                        by_group["flip_rate"] = by_group.flips / by_group.rows
                        st.dataframe(
                            by_group.style.format(
                                {"flip_rate": "{:.1%}", "mean_prob_shift": "{:.4f}"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.altair_chart(
                            alt.Chart(testable)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "prob_shift:Q",
                                    bin=alt.Bin(maxbins=25),
                                    title="|p(dup) − p(dup after swap)|",
                                ),
                                y=alt.Y("count()", title="rows"),
                                color=alt.Color(
                                    "flipped:N",
                                    scale=alt.Scale(
                                        domain=[False, True], range=["#4c78a8", "#e45756"]
                                    ),
                                    title="flipped",
                                ),
                                tooltip=["count()"],
                            )
                            .properties(height=220),
                            use_container_width=True,
                        )
                        st.caption(
                            "Rows far right shifted a lot but may not have crossed the decision "
                            "boundary. They're the near-misses — the flips you'd get from a small "
                            "change in threshold or training seed."
                        )

                    buf = io.StringIO()
                    out.to_csv(buf, index=False)
                    st.download_button(
                        "Download full report (CSV)",
                        buf.getvalue(),
                        "identity_audit.csv",
                        "text/csv",
                    )

# --------------------------------------------------------------------------
# Tab 3 — how it works
# --------------------------------------------------------------------------

with how_tab:
    from fairness_gpt2.identity import ETHNICITY_NAMES, GENDERED_NAMES, GENDERED_TERMS

    st.subheader("The idea")
    st.write(
        "Take a question pair. Swap every name and gendered word for its "
        "counterfactual — *James → Mary*, *he → she*, *Connor → Jamal* — and ask "
        "the model again. Everything about the task is unchanged: if two questions "
        "are duplicates when they're about James, they're duplicates when they're "
        "about Mary. The name appears on both sides and cancels."
    )
    st.write(
        "So if the answer changes, the identity token caused it. That's a **flip**, "
        "and the fraction of flips is the **flip rate**."
    )
    st.info(
        "A flip doesn't say which answer was right. That's the point — the complaint "
        "isn't that the model is wrong, it's that its answer depends on the name at all."
    )

    st.subheader("The model")
    st.code(
        'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?',
        language="text",
    )
    st.write(
        "GPT-2 base (124M) reads that prompt; a linear head over the final token's "
        "hidden state produces the match / no-match logits "
        "(y = W·h_final + b, W ∈ ℝ^768×2). Fine-tuned on **Quora Question Pairs** — "
        "283,011 training pairs, evaluated on 40,430."
    )
    st.caption(
        "QQP only. The source paper also reports SST, CFIMDB and sonnet-generation "
        "results as course requirements; those carry no fairness component and are "
        "out of scope here."
    )

    st.subheader("The substitutions")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Type": "Gendered name pairs",
                    "Count": len(GENDERED_NAMES),
                    "Example": "James ↔ Mary",
                },
                {
                    "Type": "Ethnicity-associated name pairs",
                    "Count": len(ETHNICITY_NAMES),
                    "Example": "Connor ↔ Jamal",
                },
                {
                    "Type": "Pronoun / gendered-term swaps",
                    "Count": len(GENDERED_TERMS),
                    "Example": "he ↔ she, king ↔ queen",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.write(
        "Applied with probability 0.5 during training (counterfactual data "
        "augmentation), and deterministically at audit time."
    )

    st.subheader("Training")
    st.write("Two interventions, on top of standard cross-entropy fine-tuning:")
    st.markdown(
        "- **CDA** — train on identity-swapped copies so the model can't lean on "
        "demographic cues.\n"
        "- **Consistency regularization** — explicitly penalise disagreeing with "
        "yourself across the swap:"
    )
    st.latex(
        r"\mathcal{L}_{\text{fairness}} = \tfrac{1}{2}\big(\mathrm{KL}(p_{cf}\,\|\,p_{orig})"
        r" + \mathrm{KL}(p_{orig}\,\|\,p_{cf})\big)"
    )
    st.latex(
        r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda\,"
        r"\mathcal{L}_{\text{fairness}},\quad \lambda = 0.5"
    )

    st.divider()
    st.subheader("This is a replication — and it found two things")
    st.write(
        "The method above comes from a Stanford CS224N report (cited below). "
        "Reproducing it from the text alone means reconstructing what the text "
        "leaves out, and two of those reconstructions turned into findings."
    )

    repro = REPRODUCED.get("cda_reg", {})

    st.markdown("**1. The paper's subgroup definition is recoverable from its own arithmetic.**")
    st.write(
        "It reports per-subgroup counts but never defines the subgroups. The counts "
        "define them anyway: they sum to 40,996 against a 40,430-pair dev set. "
        "Impossible — unless the groups overlap."
    )
    st.code(
        "|male ∩ female|             = 40,996 − 40,430      = 566\n"
        "|male ∪ female|             = 1833 + 1751 − 566    = 3,018\n"
        "union + name-only + neutral = 3,018 + 734 + 36,678 = 40,430   ← the dev set, exactly\n"
        "implied n_identity          = 3,018 + 734          = 3,752    ← paper reports 3,751",
        language="text",
    )
    st.write(
        "Two independent checks land exact. So male/female are decided by gendered "
        "**terms**, not names; a pair with both counts in both groups; *name-only* "
        "means a name with no gendered term. The intuitive reading — *James* makes a "
        "pair male — collapses *name-only* from 734 examples to 14, and the subgroup "
        "gap then measures noise on 14 rows. This project hit exactly that before the "
        "arithmetic resolved it."
    )

    st.markdown("**2. The flip-rate metric has a grammatical confound.**")
    st.write(
        "The paper lists `his ↔ hers` as a flat pair. English overloads both words, "
        "and a lookup table can't tell the senses apart:"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "": "male",
                    "subject": "he",
                    "object": "him",
                    "possessive determiner": "his",
                    "possessive pronoun": "his",
                },
                {
                    "": "female",
                    "subject": "she",
                    "object": "her",
                    "possessive determiner": "her",
                    "possessive pronoun": "hers",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.code(
        'literal mapping:   "improve his credit score"  →  "improve hers credit score"\n'
        '                   "raise her credit score"    →  "raise him credit score"\n\n'
        'resolved by role:  "improve his credit score"  →  "improve her credit score"\n'
        '                   "raise her credit score"    →  "raise his credit score"',
        language="text",
    )
    st.warning(
        "Ungrammatical text is out-of-distribution for GPT-2, so it may flip because "
        "the sentence broke — not because the name changed. Any flip the literal "
        "mapping causes and the grammatical one doesn't is measuring syntax damage. "
        "This implementation resolves `his`/`her` by syntactic role."
    )

    st.subheader("What the interventions buy")
    st.write(
        "The claim worth testing: does fairness-aware training actually make the "
        "model more robust to identity substitution, and what does it cost?"
    )

    if EFFECT is None:
        st.warning(
            "**Baseline not trained yet.** A before/after needs both models trained "
            "the same way on the same data. Substituting the paper's baseline for "
            "your own would make the comparison meaningless — so this section stays "
            "empty until `results/reproduced/baseline.json` exists."
        )
        st.caption(
            "`uv run fairness-train --mode baseline --train data/quora-train.csv "
            "--dev data/quora-dev.csv --out checkpoints/baseline --epochs 5`"
        )
        with st.expander("What the source paper reported (10 epochs) — not this project's data"):
            st.dataframe(
                pd.DataFrame(REPORTED["main"])
                .rename(
                    columns={
                        "model": "Model",
                        "dev_acc": "Dev accuracy",
                        "subgroup_gap": "Subgroup gap",
                        "flip_rate": "Flip rate",
                    }
                )
                .style.format(
                    {"Dev accuracy": "{:.2%}", "Subgroup gap": "{:.2%}", "Flip rate": "{:.2%}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Accuracy is flat across all three (89.55% → 89.47% → 89.56%). The "
                "interventions don't make the model better at the task — they make it "
                "steadier, for free."
            )
    else:
        acc, flip, gap = EFFECT["accuracy"], EFFECT["flip_rate"], EFFECT["subgroup_gap"]
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Flip rate",
            f"{flip['cda_reg']:.2%}",
            f"{flip['pct']:+.0%} vs baseline",
            delta_color="inverse",
        )
        k2.metric("Dev accuracy", f"{acc['cda_reg']:.2%}", f"{acc['delta']:+.4f} vs baseline")
        k3.metric(
            "Subgroup gap",
            f"{gap['cda_reg']:.2%}",
            f"{gap['pct']:+.0%} vs baseline",
            delta_color="inverse",
        )

        st.markdown(
            f"**Identity robustness improved {abs(flip['pct']):.0%} at no cost to accuracy.** "
            f"The baseline changed its answer on {EFFECT['flips']['baseline']} of "
            f"{EFFECT['flips']['n_identity']:,} identity-bearing pairs; with CDA and the "
            f"consistency penalty that drops to {EFFECT['flips']['cda_reg']}. Accuracy moves "
            f"{acc['delta']:+.4f} — noise."
        )

        eff = pd.DataFrame(
            [
                {
                    "Metric": "Dev accuracy ↑",
                    "Baseline": acc["baseline"],
                    "CDA + Reg.": acc["cda_reg"],
                },
                {
                    "Metric": "Flip rate ↓",
                    "Baseline": flip["baseline"],
                    "CDA + Reg.": flip["cda_reg"],
                },
                {
                    "Metric": "Subgroup gap ↓",
                    "Baseline": gap["baseline"],
                    "CDA + Reg.": gap["cda_reg"],
                },
            ]
        )
        eff["Change"] = eff["CDA + Reg."] - eff["Baseline"]
        st.dataframe(
            eff.style.format({"Baseline": "{:.2%}", "CDA + Reg.": "{:.2%}", "Change": "{:+.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

        long = eff.melt(
            id_vars="Metric",
            value_vars=["Baseline", "CDA + Reg."],
            var_name="Model",
            value_name="Value",
        )
        st.altair_chart(
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Value:Q", title=None, axis=alt.Axis(format="%")),
                color=alt.Color(
                    "Model:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=["Baseline", "CDA + Reg."], range=["#9aa7b4", "#3b6ea5"]
                    ),
                ),
                tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".2%")],
            )
            .properties(height=190)
            .facet(
                column=alt.Column(
                    "Metric:N", title=None, sort=["Dev accuracy ↑", "Flip rate ↓", "Subgroup gap ↓"]
                )
            )
            .resolve_scale(y="independent"),
            use_container_width=True,
        )

        if gap["delta"] > 0:
            st.warning(
                f"**The regularizer trades parity for stability.** The subgroup gap widened "
                f"{gap['pct']:+.0%} while the flip rate fell {abs(flip['pct']):.0%}. These are "
                "different notions of fairness and they don't move together: the consistency "
                "penalty aligns predictions on *matched pairs* (instance level), while the "
                "subgroup gap measures *aggregate accuracy across groups*. Fixing one doesn't "
                "fix the other."
            )
            if not EFFECT["has_cda"]:
                st.caption(
                    "Training the CDA-only model would separate which intervention caused what: "
                    "`--mode cda`."
                )

    st.divider()
    st.subheader("Results against the paper")
    if repro:
        rows = [
            {
                "Metric": "Dev accuracy",
                "Paper (5 epochs)": 0.8856,
                "This replication": repro.get("accuracy"),
            },
            {
                "Metric": "Flip rate",
                "Paper (5 epochs)": 0.0296,
                "This replication": repro.get("flip_rate"),
            },
        ]
        comp = pd.DataFrame(rows)
        comp["Difference"] = comp["This replication"] - comp["Paper (5 epochs)"]
        st.dataframe(
            comp.style.format(
                {
                    "Paper (5 epochs)": "{:.4f}",
                    "This replication": "{:.4f}",
                    "Difference": "{:+.4f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        if repro.get("n_identity"):
            c1, c2 = st.columns(2)
            c1.metric(
                "Identity-bearing dev examples",
                f"{repro['n_identity']:,}",
                f"{repro['n_identity'] - 3751:+,} vs paper's 3,751",
                delta_color="off",
            )
            c2.metric(
                "Lexicon coverage",
                f"{repro['n_identity'] / 3752:.1%}",
                "of the paper's implied 3,752",
                delta_color="off",
            )
        st.write(
            "Accuracy reproduces. The flip rate comes in lower — a ~1% lexicon-coverage "
            "difference can't explain that, so the grammatical confound above is the "
            "leading candidate."
        )
    else:
        st.info(
            "No reproduced results yet. Train a model and drop the JSON in `results/reproduced/`."
        )

    st.subheader("Honest limits")
    st.markdown(
        "- **The lexicon is reconstructed.** The paper gives counts (60/20/22) and three "
        "examples, not the lists. Flip rates are comparable in magnitude, not identical.\n"
        "- **Binary gender, ~100 names.** A narrow slice of the demographic space.\n"
        "- **Lexicons miss implicit cues.** No name, no measurement — the audit reports "
        "which rows it couldn't test.\n"
        "- **5 epochs**, matching the paper's ablation rather than its 10-epoch headline.\n"
        "- **Train split differs.** 283,011 pairs sampled from GLUE to match the paper's "
        "count; the original's course-provided split isn't public.\n"
        "- **Not a leaderboard model.** 89% on QQP isn't state of the art. The audit "
        "harness is the point, not the classifier."
    )

    st.divider()
    st.caption(f"Built by {AUTHOR} · [source]({REPO_URL})")
    st.markdown("##### Method replicated from")
    st.markdown(
        "> Owens, D. *Fairness-Aware Fine-Tuning of GPT-2 for Paraphrase Detection.* "
        "Stanford CS224N Default Project."
    )
    st.markdown("##### Also drawing on")
    st.caption(
        "Radford et al. 2019 (GPT-2) · Maudslay et al. 2019 (name-based counterfactual data "
        "substitution) · Bertrand & Mullainathan 2004 (audit-study name lists) · Zhao et al. "
        "2018 (WinoBias) · Dixon et al. 2018 · Black et al. 2020 (FlipTest) · "
        "Kaushik et al. 2020"
    )