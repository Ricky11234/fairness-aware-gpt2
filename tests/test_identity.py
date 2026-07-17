"""Lexicon and swap-logic tests. No model weights required, so these run in CI."""

import pytest

from fairness_gpt2.identity import (
    ETHNICITY_NAMES,
    GENDERED_NAMES,
    GENDERED_TERMS,
    contains_identity,
    subgroup_of,
    swap_identity,
)


def test_table1_counts():
    """Table 1 of the report: 60 / 20 / 22."""
    assert len(GENDERED_NAMES) == 60
    assert len(ETHNICITY_NAMES) == 20
    assert len(GENDERED_TERMS) == 22


def test_name_lists_are_disjoint():
    """A name in both lists would make the two swap categories collide."""
    gendered = {n for pair in GENDERED_NAMES for n in pair}
    ethnicity = {n for pair in ETHNICITY_NAMES for n in pair}
    assert not (gendered & ethnicity)


def test_no_duplicate_entries():
    for pairs in (GENDERED_NAMES, ETHNICITY_NAMES, GENDERED_TERMS):
        flat = [n for pair in pairs for n in pair]
        assert len(flat) == len(set(flat))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Why did James leave?", "Why did Mary leave?"),
        ("Is he the king?", "Is she the queen?"),
        ("Connor applied today.", "Jamal applied today."),
        ("What is the capital of Peru?", "What is the capital of Peru?"),
    ],
)
def test_deterministic_swap(text, expected):
    assert swap_identity(text, p=1.0) == expected


def test_swap_is_an_involution():
    """Swapping twice returns the original for symmetric pairs."""
    s = "Why did James tell his brother about Connor?"
    assert swap_identity(swap_identity(s)) == s


def test_case_is_preserved():
    assert swap_identity("JAMES and James and james") == "MARY and Mary and mary"


def test_swap_respects_word_boundaries():
    """'he' must not fire inside 'the' or 'when'."""
    s = "When the theme changed, she left."
    assert swap_identity(s) == "When the theme changed, he left."


def test_probabilistic_swap_bounds():
    import random

    s = "James and John and Robert and Michael and William"
    always = swap_identity(s, p=1.0, rng=random.Random(0))
    never = swap_identity(s, p=0.0, rng=random.Random(0))
    assert never == s
    assert always != s


def test_contains_identity():
    assert contains_identity("Why did Mary go?")
    assert contains_identity("Is she here?")
    assert not contains_identity("How do I learn Python?")


@pytest.mark.parametrize(
    "s1,s2,expected",
    [
        ("Is he tall?", "Was the king tall?", "male"),
        ("Is she tall?", "Was the queen tall?", "female"),
        ("Is Connor a good hire?", "Would Connor work out?", "name-only"),
        ("How do I learn Python?", "Best way to learn Python?", "neutral"),
        ("Is he taller than she is?", "Who is taller?", "mixed"),
    ],
)
def test_subgroup_assignment(s1, s2, expected):
    assert subgroup_of(s1, s2) == expected


def test_known_limitation_possessive_his():
    """Documented in the README: the lexicon has no syntax, so 'his book'
    becomes 'hers book'. Locked in by a test so it can't regress silently."""
    assert swap_identity("his book") == "hers book"
