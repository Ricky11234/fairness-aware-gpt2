"""Metric and data-pipeline tests. No model weights required."""

import csv

import pytest
import torch

from fairness_gpt2.data import Pair, QQPDataset, build_counterfactual_pool, load_qqp
from fairness_gpt2.evaluate import subgroup_accuracy_gap
from fairness_gpt2.train import symmetric_kl

ROWS = [
    ("0", "How can he improve his credit score?", "How does he raise his credit score?", "1"),
    ("1", "Why did James study law?", "What made James choose law?", "1"),
    ("2", "Is Connor a good hire?", "What is the capital of Peru?", "0"),
    ("3", "How do I learn Python?", "Best way to learn Python?", "1"),
    ("4", "Why is she late?", "Why is her train delayed?", "0"),
]


@pytest.fixture
def qqp_csv(tmp_path):
    path = tmp_path / "mini.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence1", "sentence2", "is_duplicate"])
        w.writerows(ROWS)
    return str(path)


def test_load_qqp(qqp_csv):
    pairs = load_qqp(qqp_csv)
    assert len(pairs) == 5
    assert [p.label for p in pairs] == [1, 1, 0, 1, 0]


def test_load_qqp_accepts_raw_quora_columns(tmp_path):
    path = tmp_path / "raw.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question1", "question2", "is_duplicate"])
        w.writerow(["0", "Why did James study law?", "What made James choose law?", "1"])
    pairs = load_qqp(str(path))
    assert len(pairs) == 1 and pairs[0].label == 1


def test_unlabelled_test_file_gets_minus_one(tmp_path):
    path = tmp_path / "test.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence1", "sentence2"])
        w.writerow(["0", "Is he here?", "Did he arrive?"])
    assert load_qqp(str(path))[0].label == -1


def test_counterfactual_pool_only_swaps_identity_pairs(qqp_csv):
    pairs = load_qqp(qqp_csv)
    pool = build_counterfactual_pool(pairs, n=10)
    # "How do I learn Python?" carries no identity token, so it can't appear.
    assert all("Python" not in p.s1 for p in pool)
    assert all(p.pid.startswith("cf-") for p in pool)


def test_counterfactual_pool_preserves_labels(qqp_csv):
    pairs = load_qqp(qqp_csv)
    by_id = {p.pid: p.label for p in pairs}
    for cf in build_counterfactual_pool(pairs, n=10):
        assert cf.label == by_id[cf.pid.removeprefix("cf-")]


def test_dataset_returns_counterfactual_view(qqp_csv):
    ds = QQPDataset(load_qqp(qqp_csv), cda_prob=0.0, return_counterfactual=True)
    item = ds[1]
    assert "James" in item["s1"]
    assert "Mary" in item["cf_s1"]


def test_dataset_is_deterministic_given_a_seed(qqp_csv):
    pairs = load_qqp(qqp_csv)
    a = [QQPDataset(pairs, cda_prob=0.5, seed=11711)[i]["s1"] for i in range(5)]
    b = [QQPDataset(pairs, cda_prob=0.5, seed=11711)[i]["s1"] for i in range(5)]
    assert a == b


def test_symmetric_kl_is_zero_for_identical_logits():
    a = torch.tensor([[2.0, -1.0], [0.5, 0.5]])
    assert symmetric_kl(a, a).item() == pytest.approx(0.0, abs=1e-6)


def test_symmetric_kl_is_positive_and_symmetric():
    a = torch.tensor([[2.0, -1.0]])
    b = torch.tensor([[-1.0, 2.0]])
    assert symmetric_kl(a, b).item() > 0
    assert symmetric_kl(a, b).item() == pytest.approx(symmetric_kl(b, a).item(), rel=1e-6)


def test_symmetric_kl_grows_with_divergence():
    orig = torch.tensor([[2.0, -1.0]])
    near = torch.tensor([[1.9, -0.9]])
    far = torch.tensor([[-2.0, 1.0]])
    assert symmetric_kl(orig, near) < symmetric_kl(orig, far)


def test_subgroup_gap_ignores_small_subgroups(qqp_csv):
    """Every subgroup here is under the 10-example floor, so the gap is 0."""
    pairs = load_qqp(qqp_csv)
    preds = torch.tensor([1, 1, 0, 1, 0])
    assert subgroup_accuracy_gap(pairs, preds)["gap"] == 0.0


def test_subgroup_gap_computes_max_pairwise_difference():
    # 10 male pairs all correct, 10 female pairs all wrong -> gap of 1.0
    pairs = [Pair("Is he here?", "Did he arrive?", 1, f"m{i}") for i in range(10)]
    pairs += [Pair("Is she here?", "Did she arrive?", 1, f"f{i}") for i in range(10)]
    preds = torch.tensor([1] * 10 + [0] * 10)

    out = subgroup_accuracy_gap(pairs, preds)
    assert out["gap"] == pytest.approx(1.0)
    assert out["per_subgroup"]["male"]["acc"] == pytest.approx(1.0)
    assert out["per_subgroup"]["female"]["acc"] == pytest.approx(0.0)


def test_overlapping_pairs_count_in_both_groups():
    """A both-genders pair lands in male AND female — that overlap is what makes
    the report's Table 5 over-sum by 566."""
    pairs = [Pair("Is he here?", "Did he arrive?", 1, f"m{i}") for i in range(10)]
    pairs += [Pair("Is he taller than she is?", "Who is taller?", 1, "x0")]
    out = subgroup_accuracy_gap(pairs, torch.tensor([1] * 11))

    assert "mixed" not in out["per_subgroup"]
    assert out["per_subgroup"]["male"]["n"] == 11  # 10 + the overlapping pair
    assert out["per_subgroup"]["female"]["n"] == 1  # only the overlapping pair
    total = sum(v["n"] for v in out["per_subgroup"].values())
    assert total > len(pairs)  # over-sums, exactly as Table 5 does
