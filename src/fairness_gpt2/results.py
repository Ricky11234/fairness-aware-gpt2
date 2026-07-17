"""Results loading: keeps what the report *claimed* separate from what this
code *reproduced*.

    results/reported.json        transcribed from the report's tables
    results/reproduced/*.json    written by the training and eval scripts

The dashboard renders both side by side so it never silently passes off the
report's numbers as this pipeline's output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTED_PATH = ROOT / "results" / "reported.json"
REPRODUCED_DIR = ROOT / "results" / "reproduced"


def load_reported() -> dict[str, Any]:
    with open(REPORTED_PATH) as f:
        return json.load(f)


def load_reproduced() -> dict[str, Any]:
    """Collect every results/reproduced/*.json into a dict keyed by filename stem.

    Returns {} when nothing has been trained yet, which is the honest state for
    a fresh clone.
    """
    out: dict[str, Any] = {}
    if not REPRODUCED_DIR.exists():
        return out
    for path in sorted(REPRODUCED_DIR.glob("*.json")):
        try:
            with open(path) as f:
                out[path.stem] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def paraphrase_comparison(reported: dict, reproduced: dict) -> list[dict]:
    """Line up the report's Table 2 against reproduced runs, per model.

    Missing reproductions come back as None rather than being dropped, so the
    dashboard can show the gap.
    """
    key_for = {
        "Baseline": "baseline",
        "CDA": "cda",
        "CDA + Fairness Regularization": "cda_reg",
    }
    rows = []
    for entry in reported["main"]:
        repro = reproduced.get(key_for.get(entry["model"], ""), {})
        rows.append(
            {
                "model": entry["model"],
                "reported_acc": entry["dev_acc"],
                "reproduced_acc": repro.get("accuracy"),
                "reported_gap": entry["subgroup_gap"],
                "reproduced_gap": repro.get("subgroup_gap"),
                "reported_flip": entry["flip_rate"],
                "reproduced_flip": repro.get("flip_rate"),
            }
        )
    return rows


def secondary_comparison(reported: dict, reproduced: dict) -> list[dict]:
    """The report's Table 3 (SST, CFIMDB) plus the sonnet CHRF."""
    rows = []
    for entry in reported["other_tasks"]:
        stem = "sst" if entry["dataset"].startswith("SST") else "cfimdb"
        repro = reproduced.get(stem, {})
        rows.append(
            {
                "task": entry["dataset"],
                "metric": "Dev accuracy",
                "reported": entry["dev_acc"],
                "reproduced": repro.get("dev_accuracy"),
            }
        )
    sonnet = reproduced.get("sonnet", {})
    rows.append(
        {
            "task": "Shakespeare sonnets",
            "metric": "CHRF",
            "reported": reported["leaderboard"]["sonnet_chrf"],
            "reproduced": sonnet.get("chrf"),
        }
    )
    return rows


# The report's fairness contribution is the paraphrase task. SST, CFIMDB and the
# sonnets are CS224N framework requirements with no fairness component, so they
# are tracked separately and don't count against the primary scope.
PRIMARY_COMPONENTS = {
    "Paraphrase — baseline": "baseline",
    "Paraphrase — CDA": "cda",
    "Paraphrase — CDA + Reg.": "cda_reg",
}
SECONDARY_COMPONENTS = {
    "SST (5-class)": "sst",
    "CFIMDB (binary)": "cfimdb",
    "Sonnet generation": "sonnet",
}


def replication_status(reproduced: dict, primary_only: bool = True) -> dict[str, bool]:
    """Which components have actually been run here.

    Args:
        reproduced: output of ``load_reproduced()``.
        primary_only: report just the three paraphrase models. That's the
            report's actual contribution; the secondary tasks are framework
            filler and shouldn't drag the completion count down when they were
            deliberately skipped.
    """
    components = dict(PRIMARY_COMPONENTS)
    if not primary_only:
        components.update(SECONDARY_COMPONENTS)
    return {label: key in reproduced for label, key in components.items()}
