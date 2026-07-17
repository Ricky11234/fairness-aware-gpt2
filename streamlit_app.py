"""Fairness-Aware GPT-2 for Paraphrase Detection — interactive demo.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fairness-Aware GPT-2 · Paraphrase Detection",
    page_icon="⇄",
    layout="wide",
)

ROOT = Path(__file__).parent

# `results` is torch-free, so importing it at module scope keeps startup fast.
from fairness_gpt2.results import (  # noqa: E402
    load_reported,
    load_reproduced,
    paraphrase_comparison,
    replication_status,
    secondary_comparison,
)

RESULTS = load_reported()
REPRODUCED = load_reproduced()


# Where to find a trained checkpoint. Set MODEL_REPO in Streamlit secrets to a
# Hugging Face repo id, or drop a checkpoint in ./checkpoints/cda_reg.
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        # No secrets.toml present (normal when running locally).
        return os.environ.get(key, default)


MODEL_REPO = _secret("MODEL_REPO")
LOCAL_CKPT = ROOT / "checkpoints" / "cda_reg"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.5rem; max-width: 1100px;}
      .verdict {font-size: 1.05rem; font-weight: 600; padding: .6rem .9rem;
                border-radius: 6px; margin-top: .5rem;}
      .stable {background:#e8f4ea; color:#1d6a33; border:1px solid #bcdcc4;}
      .flipped {background:#fdeaea; color:#96231f; border:1px solid #f2c2c0;}
      .swapmark {background:#fff3cd; padding:0 .15rem; border-radius:3px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fairness-Aware Fine-Tuning of GPT-2 for Paraphrase Detection")
st.caption(
    "Deonna Owens · Stanford CS224N · Does the model's answer change when only "
    "the names and pronouns change?"
)


# --------------------------------------------------------------------------
# Model loading (lazy — the dashboard works without a checkpoint)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading GPT-2 checkpoint…")
def load_model():
    import torch  # imported lazily so the dashboard renders fast

    from fairness_gpt2.model import GPT2ParaphraseClassifier, build_tokenizer

    ckpt_dir = None
    if MODEL_REPO:
        from huggingface_hub import snapshot_download

        ckpt_dir = snapshot_download(repo_id=MODEL_REPO)
    elif LOCAL_CKPT.exists():
        ckpt_dir = str(LOCAL_CKPT)
    else:
        return None, None

    torch.set_num_threads(2)
    model = GPT2ParaphraseClassifier.load(ckpt_dir, device="cpu")
    return model, build_tokenizer(
        ckpt_dir if os.path.exists(os.path.join(ckpt_dir, "vocab.json")) else "gpt2"
    )


def predict_probs(model, tokenizer, s1: str, s2: str):
    import torch

    from fairness_gpt2.model import encode_pairs

    enc = encode_pairs(tokenizer, [s1], [s2])
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
    return torch.softmax(logits, dim=-1)[0].tolist()


def highlight_swaps(original: str, swapped: str) -> str:
    out = []
    for a, b in zip(original.split(), swapped.split(), strict=True):
        out.append(f'<span class="swapmark">{b}</span>' if a != b else b)
    return " ".join(out)


probe_tab, results_tab, method_tab = st.tabs(["Counterfactual probe", "Results", "How it works"])

# --------------------------------------------------------------------------
# Tab 1 — live probe
# --------------------------------------------------------------------------
with probe_tab:
    from fairness_gpt2.identity import contains_identity, subgroup_of, swap_identity

    st.write(
        "Enter a question pair containing a name or a gendered word. The model "
        "predicts whether the two questions are paraphrases, then predicts again "
        "on an identity-swapped copy. A **flip** means the demographic token, not "
        "the meaning, moved the decision."
    )

    examples = {
        "Gendered pronouns": (
            "How can he improve his credit score quickly?",
            "What should he do to raise his credit score fast?",
        ),
        "Gendered names": (
            "Why did James decide to study law?",
            "What made James choose a legal career?",
        ),
        "Ethnicity-associated names": (
            "Is Connor a good candidate for the software role?",
            "Would Connor be a strong hire as a software engineer?",
        ),
    }
    choice = st.selectbox("Start from an example", list(examples) + ["Blank"])
    d1, d2 = examples.get(choice, ("", ""))

    c1, c2 = st.columns(2)
    q1 = c1.text_area("Question 1", value=d1, height=90)
    q2 = c2.text_area("Question 2", value=d2, height=90)

    if st.button("Run the probe", type="primary"):
        if not q1.strip() or not q2.strip():
            st.warning("Fill in both questions to run the probe.")
        else:
            cf1, cf2 = swap_identity(q1), swap_identity(q2)
            st.markdown("**Identity-swapped copy**")
            st.markdown(highlight_swaps(q1, cf1), unsafe_allow_html=True)
            st.markdown(highlight_swaps(q2, cf2), unsafe_allow_html=True)
            st.caption(
                f"Subgroup — original: **{subgroup_of(q1, q2)}** → "
                f"swapped: **{subgroup_of(cf1, cf2)}**"
            )

            if not contains_identity(q1 + " " + q2):
                st.info(
                    "No identity tokens found, so the swapped copy is identical. "
                    "Add a name or a gendered word to see the probe do its work."
                )

            model, tokenizer = load_model()
            if model is None:
                st.warning(
                    "No checkpoint configured, so predictions are unavailable. "
                    "Set `MODEL_REPO` in Streamlit secrets to your Hugging Face "
                    "repo id, or place a checkpoint in `checkpoints/cda_reg/`. "
                    "The swap itself runs without a model."
                )
            else:
                p_orig = predict_probs(model, tokenizer, q1, q2)
                p_cf = predict_probs(model, tokenizer, cf1, cf2)
                y_orig, y_cf = int(p_orig[1] > p_orig[0]), int(p_cf[1] > p_cf[0])

                m1, m2 = st.columns(2)
                m1.metric(
                    "Original → paraphrase?", "Yes" if y_orig else "No", f"p = {p_orig[1]:.3f}"
                )
                m2.metric("Swapped → paraphrase?", "Yes" if y_cf else "No", f"p = {p_cf[1]:.3f}")

                cls = "flipped" if y_orig != y_cf else "stable"
                msg = (
                    "Prediction flipped. The identity token changed the label."
                    if y_orig != y_cf
                    else f"Prediction held. Probability moved by {abs(p_orig[1] - p_cf[1]):.4f}."
                )
                st.markdown(f'<div class="verdict {cls}">{msg}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Tab 2 — results
# --------------------------------------------------------------------------
with results_tab:
    status = replication_status(REPRODUCED)  # paraphrase models only
    done = sum(status.values())

    if done == 0:
        st.info(
            "**Showing the numbers reported in the paper.** Nothing has been trained in "
            "this repo yet, so there is nothing to compare against. Run the training "
            "scripts and this page fills in a reproduced column automatically."
        )
    elif done < len(status):
        st.warning(
            f"**Partial — {done} of {len(status)} paraphrase models reproduced.** "
            "Blank cells below haven't been run yet."
        )
    else:
        st.success(
            "**Paraphrase task fully reproduced.** All three models trained and evaluated here."
        )

    with st.expander(
        f"Replication status — {done}/{len(status)} paraphrase models", expanded=done == 0
    ):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Component": k, "Reproduced": "yes" if v else "not yet"}
                    for k, v in status.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Scope is the paraphrase task — the report's fairness contribution. SST, "
            "CFIMDB and sonnet generation are CS224N framework requirements with no "
            "fairness component; their reported figures appear below for completeness. "
            "The leaderboard test accuracy (0.876) can't be reproduced at all — it needs "
            "a submission to the CS224N leaderboard, which holds the test labels."
        )

    st.subheader("Paraphrase detection")
    st.write(
        "CDA buys subgroup parity. The consistency regularizer buys instance-level "
        "stability. Neither buys both, and neither costs accuracy."
    )

    comp = pd.DataFrame(paraphrase_comparison(RESULTS, REPRODUCED))
    display = pd.DataFrame(
        {
            "Model": comp["model"],
            "Dev acc (paper)": comp["reported_acc"],
            "Dev acc (yours)": comp["reproduced_acc"],
            "Subgroup gap (paper)": comp["reported_gap"],
            "Subgroup gap (yours)": comp["reproduced_gap"],
            "Flip rate (paper)": comp["reported_flip"],
            "Flip rate (yours)": comp["reproduced_flip"],
        }
    )
    st.dataframe(
        display.style.format({c: "{:.2%}" for c in display.columns if c != "Model"}, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    # Chart the paper's numbers; overlay reproduced ones where they exist.
    long = comp.melt(
        id_vars="model",
        value_vars=["reported_acc", "reported_gap", "reported_flip"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(
        {
            "reported_acc": "Dev accuracy ↑",
            "reported_gap": "Subgroup gap ↓",
            "reported_flip": "Flip rate ↓",
        }
    )
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("model:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["model", "metric", alt.Tooltip("value:Q", format=".2%")],
        )
        .properties(height=220)
        .facet(
            column=alt.Column(
                "metric:N",
                title=None,
                sort=["Dev accuracy ↑", "Subgroup gap ↓", "Flip rate ↓"],
            )
        )
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Bars show the paper's reported figures.")

    st.subheader("Secondary tasks")
    sec = pd.DataFrame(secondary_comparison(RESULTS, REPRODUCED))
    st.dataframe(
        sec.rename(
            columns={
                "task": "Task",
                "metric": "Metric",
                "reported": "Paper",
                "reproduced": "Yours",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "SST is 5-class, so the low accuracy is expected. CFIMDB is binary. "
        "Sonnet quality is CHRF, a character n-gram overlap score, not a percentage."
    )

    st.subheader("Per-subgroup dev accuracy (10-epoch models, as reported)")
    sub = pd.DataFrame(RESULTS["subgroups"])
    st.dataframe(
        sub.style.format({"baseline": "{:.2%}", "cda": "{:.2%}", "cda_reg": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )
    g = RESULTS["male_female_gap"]
    st.caption(
        f"Male–female gap: baseline {g['baseline']:.2%} → CDA {g['cda']:.2%} "
        f"→ CDA + Reg. {g['cda_reg']:.2%}. The neutral subgroup is ~90% of the "
        "dev set, so overall accuracy barely moves."
    )

    st.subheader("Training dynamics — CDA + Fairness Regularization (as reported)")
    dyn = pd.DataFrame(RESULTS["training_dynamics"])
    base = alt.Chart(dyn).encode(x=alt.X("epoch:Q", title="Epoch"))
    loss_line = base.mark_line(point=True, color="#4c78a8").encode(
        y=alt.Y("task_loss:Q", title="Task loss")
    )
    fair_line = base.mark_line(point=True, color="#e45756", strokeDash=[4, 3]).encode(
        y=alt.Y("fairness_loss:Q", title="Fairness loss")
    )
    st.altair_chart(
        alt.layer(loss_line, fair_line).resolve_scale(y="independent").properties(height=260),
        use_container_width=True,
    )
    st.caption(
        "Task loss falls while fairness loss climbs: as predictions sharpen, the "
        "same small original/counterfactual disagreement produces a larger KL. The "
        "penalty bounds the divergence rather than erasing it."
    )

    if REPRODUCED:
        with st.expander("Raw reproduced results (JSON)"):
            st.json(REPRODUCED)

# --------------------------------------------------------------------------
# Tab 3 — method
# --------------------------------------------------------------------------
with method_tab:
    st.subheader("Task setup")
    st.code(
        'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?',
        language="text",
    )
    st.write(
        "GPT-2 base (124M) reads that prompt. A linear head over the final "
        "token's hidden state produces the two paraphrase logits: y = W·h_final + b, "
        "with W ∈ ℝ^768×2."
    )

    st.subheader("Counterfactual data augmentation")
    from fairness_gpt2.identity import ETHNICITY_NAMES, GENDERED_NAMES, GENDERED_TERMS

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Substitution type": "Gendered name pairs",
                    "Count": len(GENDERED_NAMES),
                    "Example": "James ↔ Mary",
                },
                {
                    "Substitution type": "Ethnicity-associated name pairs",
                    "Count": len(ETHNICITY_NAMES),
                    "Example": "Connor ↔ Jamal",
                },
                {
                    "Substitution type": "Pronoun / gendered-term swaps",
                    "Count": len(GENDERED_TERMS),
                    "Example": "he ↔ she, king ↔ queen",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.write(
        "Swaps fire with probability 0.5 during training, and deterministically at evaluation."
    )

    st.subheader("Consistency regularization")
    st.latex(
        r"\mathcal{L}_{\text{fairness}} = \tfrac{1}{2}\big(\mathrm{KL}(p_{cf}\,\|\,p_{orig}) + \mathrm{KL}(p_{orig}\,\|\,p_{cf})\big)"
    )
    st.latex(
        r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda\,\mathcal{L}_{\text{fairness}},\quad \lambda = 0.5"
    )

    st.subheader("Metrics")
    st.markdown(
        "- **Subgroup accuracy gap** — max pairwise accuracy difference across "
        "male / female / neutral / name-only, counting only subgroups with ≥10 examples.\n"
        "- **Prediction flip rate** — fraction of identity-bearing examples whose "
        "predicted label changes under a deterministic swap."
    )

    st.subheader("Limits worth naming")
    st.markdown(
        "- Subgroup assignment uses fixed lexicons and misses implicit cues.\n"
        "- Swaps are binary-gender and cover a small name list.\n"
        "- `his` and `her` are each two words (possessive determiner vs. pronoun). "
        "The paper's flat mapping produces ungrammatical counterfactuals like "
        "*hers book*, which confounds the flip rate — a model can flip because the "
        "syntax broke, not because the identity changed. This implementation "
        "resolves them by syntactic role instead.\n"
        "- ~90% of the dev set is identity-free, so headline accuracy is dominated "
        "by examples the interventions never touch."
    )
