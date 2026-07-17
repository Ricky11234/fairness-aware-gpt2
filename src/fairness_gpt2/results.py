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


def replication_status(reproduced: dict) -> dict[str, bool]:
    """Which of the report's components have actually been run here."""
    return {
        "Paraphrase — baseline": "baseline" in reproduced,
        "Paraphrase — CDA": "cda" in reproduced,
        "Paraphrase — CDA + Reg.": "cda_reg" in reproduced,
        "SST (5-class)": "sst" in reproduced,
        "CFIMDB (binary)": "cfimdb" in reproduced,
        "Sonnet generation": "sonnet" in reproduced,
    }
