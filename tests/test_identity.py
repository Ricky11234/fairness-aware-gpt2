"""Lexicon and swap-logic tests. No model weights required, so these run in CI."""

import pytest

from fairness_gpt2.identity import (
    ETHNICITY_NAMES,
    GENDERED_NAMES,
    GENDERED_TERMS,
    contains_identity,
    subgroup_of,
    subgroups_of,
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
        ("Is he tall?", "Was the king tall?", {"male"}),
        ("Is she tall?", "Was the queen tall?", {"female"}),
        ("Is Connor a good hire?", "Would Connor work out?", {"name-only"}),
        ("How do I learn Python?", "Best way to learn Python?", {"neutral"}),
        # Both gendered terms -> BOTH groups. This overlap is what makes the
        # report's Table 5 sum to 40,996 against a 40,430-pair dev set.
        ("Is he taller than she is?", "Who is taller?", {"male", "female"}),
    ],
)
def test_subgroup_assignment(s1, s2, expected):
    assert subgroups_of(s1, s2) == expected


def test_names_alone_are_name_only_not_gendered():
    """Recovered from Table 5: names do NOT make a pair male/female. Treating
    'James' as male collapses name-only from 734 examples to ~14, and the
    subgroup gap then measures noise on a 14-example group."""
    assert subgroups_of("Why did James study law?", "What made James choose law?") == {"name-only"}
    assert subgroups_of("Is Mary here?", "Did Mary arrive?") == {"name-only"}


def test_a_name_plus_a_gendered_term_is_gendered():
    assert subgroups_of("Did James tell his brother?", "What did he say?") == {"male"}


def test_subgroup_of_joins_overlapping_groups_for_display():
    assert subgroup_of("Is he taller than she is?", "Who is taller?") == "female+male"


@pytest.mark.parametrize(
    "text,expected",
    [
        # "his" as possessive determiner -> "her"
        ("his book", "her book"),
        ("How can he improve his credit score?", "How can she improve her credit score?"),
        # "his" as possessive pronoun -> "hers"
        ("The car is his.", "The car is hers."),
        ("Is that his?", "Is that hers?"),
        # "her" as possessive determiner -> "his"
        ("her book", "his book"),
        (
            "What should she do to raise her credit score?",
            "What should he do to raise his credit score?",
        ),
        # "her" as object pronoun -> "him"
        ("I saw her.", "I saw him."),
        ("I gave her the book", "I gave him the book"),
        # "hers" is unambiguous
        ("Is that hers?", "Is that his?"),
    ],
)
def test_ambiguous_pronouns_resolve_by_syntactic_role(text, expected):
    """'his' and 'her' are each two different words. A flat lookup produces
    'hers book' / 'him credit score', which is ungrammatical — and an
    ungrammatical counterfactual confounds the flip rate, since the model may
    flip because the syntax broke rather than because the identity changed."""
    assert swap_identity(text) == expected


def test_ambiguous_pronoun_swaps_round_trip():
    for s in ("his book", "The car is his.", "I gave her the book", "I saw her."):
        assert swap_identity(swap_identity(s)) == s


def test_paper_literal_mode_reproduces_the_flat_mapping():
    """Table 1 lists `his <-> hers` flatly. Keep that reachable for anyone
    reproducing the report exactly."""
    assert swap_identity("his book", contextual_pronouns=False) == "hers book"
    assert swap_identity("I saw her.", contextual_pronouns=False) == "I saw him."
