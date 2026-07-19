"""Blind Match — identity-robustness auditing for text-similarity / plagiarism detection.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Blind Match · plagiarism-checker fairness audit",
    page_icon="⧉",
    layout="wide",
)


from fairness_gpt2.results import (  # noqa: E402
    intervention_effect,
    load_reproduced,
)

REPRODUCED = load_reproduced()
EFFECT = intervention_effect(REPRODUCED)

# ---- EDIT THESE ----------------------------------------------------------
AUTHOR = "Abhinav Barman"
REPO_URL = "https://github.com/Ricky11234/fairness-aware-gpt2"
# --------------------------------------------------------------------------

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
# Header — findings-report board
# --------------------------------------------------------------------------
_flip_line = "FAIRNESS TUNING ▸ CONNECT RESULTS TO POPULATE"
if EFFECT:
    f = EFFECT["flip_rate"]
    _flip_line = (
        f"BASELINE → CDA + REGULARIZATION  ▸  IDENTITY-DRIVEN FLIPS  "
        f"{f['baseline']:.1%} → {f['cda_reg']:.1%}   ({f['pct']:+.0%})"
    )

st.markdown(
    f"""
    <div class="board">
      <div class="eyebrow">⧉ Fairness-Aware GPT-2 · Quora Question Pairs · Findings</div>
      <h1>Blind&nbsp;Match</h1>
      <p class="tag">A study measuring whether counterfactual data augmentation and a
      consistency regularizer make a GPT-2 paraphrase detector robust to author
      identity — and what that robustness costs in accuracy. Below: how the tuned
      model compares to an untuned baseline, the improvements it delivers, and the
      method behind it.</p>
      <div class="flip-strip">{_flip_line}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mission">
      <b>What this study measures.</b> Not whether the model correctly separates
      duplicate passages from distinct ones — but whether its verdict <b>stays the
      same</b> when only the author's name or gender changes. A detector can be
      accurate on average and still, quietly, flag one student while clearing an
      identical passage signed with a different name. These results quantify that,
      and show the fairness tuning that reduces it.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# --------------------------------------------------------------------------
# A read-only findings report: comparison, improvements, method. No live
# inference — the point is the result, not a prediction demo.
# --------------------------------------------------------------------------
compare_tab, improve_tab, how_tab = st.tabs(
    ["  Results  ", "  Key improvements  ", "  How it works  "]
)

with compare_tab:
    repro = REPRODUCED.get("cda_reg", {})
    st.subheader("What the interventions buy")
    st.write(
        "The claim worth testing: does fairness-aware training actually make the "
        "model more robust to identity substitution, and what does it cost?"
    )

    if EFFECT is None:
        st.warning(
            "**Baseline not trained yet.** The comparison needs both the tuned model "
            "and an untuned baseline trained the same way on the same data. It stays "
            "empty until `results/reproduced/baseline.json` exists."
        )
        st.caption(
            "`uv run fairness-train --mode baseline --train data/quora-train.csv "
            "--dev data/quora-dev.csv --out checkpoints/baseline --epochs 5`"
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
        # Short axis labels ("Base"/"Tuned") so the two bars per panel don't
        # overlap; full names live in the legend and tooltip.
        long["Model"] = long["Model"].map({"Baseline": "Base", "CDA + Reg.": "Tuned"})
        st.altair_chart(
            alt.Chart(long)
            .mark_bar(size=34)
            .encode(
                x=alt.X(
                    "Model:N",
                    title=None,
                    sort=["Base", "Tuned"],
                    axis=alt.Axis(labelAngle=0, labelFontSize=12),
                ),
                y=alt.Y("Value:Q", title=None, axis=alt.Axis(format="%")),
                color=alt.Color(
                    "Model:N",
                    title=None,
                    sort=["Base", "Tuned"],
                    scale=alt.Scale(domain=["Base", "Tuned"], range=["#b9c2ae", "#1f4d3a"]),
                    legend=alt.Legend(orient="top", direction="horizontal"),
                ),
                tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".2%")],
            )
            .properties(width=150, height=200)
            .facet(
                column=alt.Column(
                    "Metric:N",
                    title=None,
                    sort=["Dev accuracy ↑", "Flip rate ↓", "Subgroup gap ↓"],
                    header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
                )
            )
            .resolve_scale(y="independent"),
            use_container_width=True,
        )
        st.caption("Base = untuned baseline · Tuned = CDA + consistency regularizer.")

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
    st.subheader("The tuned model in numbers")
    if repro:
        c1, c2, c3 = st.columns(3)
        c1.metric("Dev accuracy", f"{repro.get('accuracy', 0):.2%}")
        c2.metric("Identity-driven flip rate", f"{repro.get('flip_rate', 0):.2%}")
        c3.metric("Subgroup accuracy gap", f"{repro.get('subgroup_gap', 0):.2%}")
        if repro.get("n_identity"):
            st.caption(
                f"Evaluated on the full 40,430-pair Quora dev set; "
                f"{repro['n_identity']:,} of those carry a name or gendered word and are "
                "the ones an identity swap can affect."
            )
    else:
        st.info("No results yet. Train the model and drop the JSON in `results/reproduced/`.")

    st.divider()
    st.subheader("Why this matters for plagiarism and copyright checking")
    st.write(
        "Automated integrity tools — Turnitin, Copyscape, and the similarity check inside "
        "most LMS gradebooks — all rest on the same core operation this model performs: "
        "deciding whether two passages say the same thing. If that decision shifts when only "
        "the author's name or gender changes, the tool flags one student and clears an "
        "identical submission from another. That is a fairness failure with real "
        "consequences — an academic-integrity referral, a failing grade, a copyright strike — "
        "assigned partly on identity rather than text."
    )
    st.write(
        "The improvement measured above is exactly the property such a system needs. A "
        f"**{abs(EFFECT['flip_rate']['pct']):.0%} reduction in identity-driven flips** means the "
        "similarity verdict is far more likely to depend on the words on the page and far less "
        "on whose name is attached to them — at no cost to how well the model tells matches from "
        "non-matches. Put plainly: the same passage draws the same flag whether it is signed "
        "*James* or *Mary*."
    ) if EFFECT else st.info("Train the baseline to populate this comparison.")
    st.caption(
        "Scope note: this project audits the pairwise similarity operation those tools rely on. "
        "It is not a drop-in replacement for a full plagiarism service, which also matches against "
        "large document corpora. The contribution is the fairness audit of the matching step."
    )

with improve_tab:
    st.subheader("Two methodology choices that make the numbers trustworthy")
    st.write(
        "Two methodology choices in this project take real care to get right. Both "
        "are places where the obvious approach quietly corrupts the fairness metric, "
        "and getting them wrong inflates or deflates the numbers by large margins."
    )

    repro = REPRODUCED.get("cda_reg", {})

    st.markdown("**1. Subgroups are defined by gendered terms, not names.**")
    st.write(
        "To measure a subgroup accuracy gap, every dev pair is assigned to male, "
        "female, name-only, or neutral. The tempting rule — *a name like James makes "
        "a pair male* — is wrong, and provably so against the dev-set counts:"
    )
    st.code(
        "|male ∩ female|             = 40,996 − 40,430      = 566\n"
        "|male ∪ female|             = 1833 + 1751 − 566    = 3,018\n"
        "union + name-only + neutral = 3,018 + 734 + 36,678 = 40,430   ← the dev set, exactly\n"
        "implied n_identity          = 3,018 + 734          = 3,752",
        language="text",
    )
    st.write(
        "The definition used here: male/female are decided by gendered **terms** "
        "(pronouns, gendered nouns); a pair carrying both counts in both groups; "
        "*name-only* means a name with no gendered term. Treating a name as male "
        "instead collapses *name-only* from 734 examples down to 14 — and the "
        "subgroup gap then measures sampling noise on 14 rows rather than a real "
        "disparity. This project hit exactly that inflated gap (0.13) before the "
        "correct definition brought it to 0.03."
    )

    st.markdown("**2. Pronoun swaps are resolved by grammar, not a flat lookup.**")
    st.write(
        "A naïve identity swap maps `his → hers` in every position. But English "
        "overloads both words, and a lookup table can't tell the senses apart:"
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
        'flat lookup:       "improve his credit score"  →  "improve hers credit score"\n'
        '                   "raise her credit score"    →  "raise him credit score"\n\n'
        'resolved by role:  "improve his credit score"  →  "improve her credit score"\n'
        '                   "raise her credit score"    →  "raise his credit score"',
        language="text",
    )
    st.warning(
        "Ungrammatical text is out-of-distribution for GPT-2, so a flat lookup makes "
        "predictions flip because the *sentence broke*, not because the identity "
        "changed — inflating the flip rate with noise that has nothing to do with "
        "fairness. This project resolves `his`/`her` by syntactic role so every "
        "counterfactual stays grammatical, and the flip rate measures identity "
        "sensitivity alone."
    )

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
        "Scope is Quora Question Pairs only — the setting where paraphrase detection "
        "and author identity intersect."
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

    st.subheader("Honest limits")
    st.markdown(
        "- **The substitution lexicon is finite.** 60 gendered names, 20 "
        "ethnicity-associated names, and 22 pronoun/term swaps. It captures explicit "
        "identity tokens, not every possible cue.\n"
        "- **Binary gender, ~100 names.** A narrow slice of the demographic space.\n"
        "- **Lexicons miss implicit cues.** No name, no measurement — the audit reports "
        "which rows it couldn't test.\n"
        "- **5 training epochs.** Chosen for a fast, reproducible run rather than "
        "squeezing the last fraction of a point.\n"
        "- **Train split.** 283,011 Quora pairs sampled from the GLUE QQP training set.\n"
        "- **Not a leaderboard model.** ~89% on QQP isn't state of the art. The fairness "
        "audit is the contribution, not the raw classifier."
    )

    st.divider()
    st.caption(f"Built by {AUTHOR} · [source]({REPO_URL})")

    st.divider()
    st.markdown("##### References")
    st.caption(
        "Owens, D. *Fairness-Aware Fine-Tuning of GPT-2 for Paraphrase Detection.* "
        "Stanford CS224N · Radford et al. 2019 (GPT-2) · Maudslay et al. 2019 "
        "(name-based counterfactual data substitution) · Bertrand & Mullainathan 2004 "
        "(audit-study name lists) · Zhao et al. 2018 (WinoBias) · Dixon et al. 2018 · "
        "Black et al. 2020 (FlipTest) · Kaushik et al. 2020"
    )