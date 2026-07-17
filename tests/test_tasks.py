"""Tests for the secondary tasks (SST, CFIMDB, sonnets) and the results pipeline."""

import csv
import json

import pytest

from fairness_gpt2.data import load_sentiment
from fairness_gpt2.model import NUM_LABELS, SENTIMENT_TEMPLATE
from fairness_gpt2.results import (
    paraphrase_comparison,
    replication_status,
    secondary_comparison,
)
from fairness_gpt2.train_sonnet import parse_sonnets

# --------------------------------------------------------------------------
# Sentiment loading
# --------------------------------------------------------------------------


@pytest.fixture
def sst_csv(tmp_path):
    path = tmp_path / "sst.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "label"])
        w.writerows(
            [
                ("0", "a stirring, funny and finally transporting film", "4"),
                ("1", "it's not life-affirming", "1"),
                ("2", "   ", "2"),  # blank -> dropped
            ]
        )
    return str(path)


def test_load_sentiment(sst_csv):
    ex = load_sentiment(sst_csv)
    assert len(ex) == 2
    assert [e.label for e in ex] == [4, 1]


def test_load_sentiment_accepts_text_column(tmp_path):
    path = tmp_path / "cf.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label"])
        w.writerow(["0", "A wonderful film.", "1"])
    assert load_sentiment(str(path))[0].label == 1


def test_load_sentiment_unlabelled(tmp_path):
    path = tmp_path / "test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence"])
        w.writerow(["0", "No label here."])
    assert load_sentiment(str(path))[0].label == -1


def test_label_counts_match_the_report():
    """SST is 5-class, CFIMDB and QQP binary (Tables 2 and 3)."""
    assert NUM_LABELS == {"qqp": 2, "sst": 5, "cfimdb": 2}


def test_sentiment_template_is_single_sentence():
    out = SENTIMENT_TEMPLATE.format(s="a fine film")
    assert "a fine film" in out
    assert "{s1}" not in out


# --------------------------------------------------------------------------
# Sonnet parsing
# --------------------------------------------------------------------------

SONNET_CORPUS = """I

From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:

II

When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a tatter'd weed of small worth held:

III

Look in thy glass and tell the face thou viewest
Now is the time that face should form another;
Whose fresh repair if now thou not renewest,
Thou dost beguile the world, unbless some mother.
"""


@pytest.fixture
def sonnet_file(tmp_path):
    path = tmp_path / "sonnets.txt"
    path.write_text(SONNET_CORPUS)
    return str(path)


def test_parse_sonnets_finds_all_three(sonnet_file):
    assert len(parse_sonnets(sonnet_file)) == 3


def test_parse_sonnets_strips_roman_numeral_headers(sonnet_file):
    for sonnet in parse_sonnets(sonnet_file):
        assert sonnet[0] not in ("I", "II", "III")
        assert len(sonnet) == 4


def test_parse_sonnets_handles_crlf_and_indentation(tmp_path):
    """The Gutenberg etext is CRLF with two-space indents. Parsing must survive
    both — this is the real-world format, not the tidy fixture above."""
    path = tmp_path / "crlf.txt"
    path.write_bytes(
        b"  I\r\n\r\n"
        b"  From fairest creatures we desire increase,\r\n"
        b"  That thereby beauty's rose might never die,\r\n"
        b"  But as the riper should by time decease,\r\n"
        b"    His tender heir might bear his memory:\r\n\r\n"
        b"  II\r\n\r\n"
        b"  When forty winters shall besiege thy brow,\r\n"
        b"  And dig deep trenches in thy beauty's field,\r\n"
        b"  Thy youth's proud livery so gazed on now,\r\n"
        b"    Will be a tatter'd weed of small worth held:\r\n"
    )
    sonnets = parse_sonnets(str(path))
    assert len(sonnets) == 2
    assert sonnets[0][0] == "From fairest creatures we desire increase,"
    assert "\r" not in "".join(sonnets[0])


def test_parse_sonnets_rejects_a_gutenberg_block_page(tmp_path):
    """PG serves an HTML block page to unknown user-agents. Writing that to disk
    silently is what produced the '0 sonnets parsed' failure — the downloader
    now validates, but the parser must not hallucinate sonnets out of HTML."""
    path = tmp_path / "blocked.txt"
    path.write_text(
        "<html><head><title>Access Denied</title></head>\n\n"
        "<body><p>Your IP has been automatically blocked.</p></body></html>\n"
    )
    assert len(parse_sonnets(str(path))) == 0


def test_parse_sonnets_skips_blocks_too_short_to_split(tmp_path):
    """A block needs more than the 3 prompt lines, or there's no reference left."""
    path = tmp_path / "s.txt"
    path.write_text("I\n\nOne line only\n\nII\n\na\nb\nc\nd\n")
    assert len(parse_sonnets(str(path))) == 1


# --------------------------------------------------------------------------
# Results pipeline
# --------------------------------------------------------------------------

REPORTED = {
    "main": [
        {"model": "Baseline", "dev_acc": 0.8955, "subgroup_gap": 0.0365, "flip_rate": 0.0456},
        {"model": "CDA", "dev_acc": 0.8947, "subgroup_gap": 0.0290, "flip_rate": 0.0355},
        {
            "model": "CDA + Fairness Regularization",
            "dev_acc": 0.8956,
            "subgroup_gap": 0.0432,
            "flip_rate": 0.0280,
        },
    ],
    "other_tasks": [
        {"dataset": "SST (5-class)", "dev_acc": 0.509},
        {"dataset": "CFIMDB (binary)", "dev_acc": 0.984},
    ],
    "leaderboard": {"qqp_test_accuracy": 0.876, "sonnet_chrf": 41.294},
}


def test_comparison_is_empty_before_training():
    """A fresh clone has reproduced nothing — say so rather than implying otherwise."""
    rows = paraphrase_comparison(REPORTED, {})
    assert len(rows) == 3
    assert all(r["reproduced_acc"] is None for r in rows)


def test_comparison_picks_up_a_reproduced_run():
    reproduced = {"cda_reg": {"accuracy": 0.8901, "subgroup_gap": 0.041, "flip_rate": 0.031}}
    rows = paraphrase_comparison(REPORTED, reproduced)
    reg = next(r for r in rows if r["model"] == "CDA + Fairness Regularization")
    assert reg["reproduced_acc"] == 0.8901
    assert reg["reported_acc"] == 0.8956
    # Untrained models stay None.
    assert next(r for r in rows if r["model"] == "Baseline")["reproduced_acc"] is None


def test_secondary_comparison_includes_sonnet_chrf():
    rows = secondary_comparison(REPORTED, {"sonnet": {"chrf": 39.5}, "sst": {"dev_accuracy": 0.5}})
    sonnet = next(r for r in rows if r["task"] == "Shakespeare sonnets")
    assert sonnet["reported"] == 41.294
    assert sonnet["reproduced"] == 39.5
    sst = next(r for r in rows if r["task"].startswith("SST"))
    assert sst["reproduced"] == 0.5
    cfimdb = next(r for r in rows if r["task"].startswith("CFIMDB"))
    assert cfimdb["reproduced"] is None


def test_replication_status_defaults_to_the_paraphrase_task():
    """Scope is QQP — the report's fairness contribution. Skipping the framework
    tasks shouldn't show up as an incomplete replication."""
    status = replication_status({"cda_reg": {}, "sst": {}})
    assert len(status) == 3
    assert status["Paraphrase — CDA + Reg."] is True
    assert status["Paraphrase — baseline"] is False
    assert "SST (5-class)" not in status


def test_replication_status_can_include_secondary_tasks():
    status = replication_status({"cda_reg": {}, "sst": {}}, primary_only=False)
    assert len(status) == 6
    assert status["SST (5-class)"] is True
    assert status["Sonnet generation"] is False


def test_load_reproduced_ignores_corrupt_files(tmp_path, monkeypatch):
    import fairness_gpt2.results as R

    d = tmp_path / "reproduced"
    d.mkdir()
    (d / "cda.json").write_text(json.dumps({"accuracy": 0.9}))
    (d / "broken.json").write_text("{not json")
    monkeypatch.setattr(R, "REPRODUCED_DIR", d)

    out = R.load_reproduced()
    assert "cda" in out
    assert "broken" not in out
