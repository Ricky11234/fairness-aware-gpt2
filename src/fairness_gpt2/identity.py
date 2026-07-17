"""Identity substitution lexicons and swapping logic.

Implements the three substitution categories in Table 1 of the paper:
  - 60 gendered name pairs
  - 20 ethnicity-associated name pairs
  - 22 pronoun / gendered-term swaps
"""

from __future__ import annotations

import random
import re

# --------------------------------------------------------------------------
# Table 1, row 1: gendered name pairs (60)
# --------------------------------------------------------------------------
# fmt: off
GENDERED_NAMES: list[tuple[str, str]] = [
    ("james", "mary"), ("john", "patricia"), ("robert", "jennifer"),
    ("michael", "linda"), ("william", "elizabeth"), ("david", "barbara"),
    ("richard", "susan"), ("joseph", "jessica"), ("thomas", "sarah"),
    ("charles", "karen"), ("christopher", "nancy"), ("daniel", "lisa"),
    ("matthew", "margaret"), ("anthony", "betty"), ("mark", "sandra"),
    ("donald", "ashley"), ("steven", "dorothy"), ("paul", "kimberly"),
    ("andrew", "emily"), ("joshua", "donna"), ("kenneth", "michelle"),
    ("kevin", "carol"), ("brian", "amanda"), ("george", "melissa"),
    ("timothy", "deborah"), ("ronald", "stephanie"), ("edward", "rebecca"),
    ("jason", "laura"), ("jeffrey", "sharon"), ("ryan", "cynthia"),
    ("jacob", "kathleen"), ("gary", "amy"), ("nicholas", "angela"),
    ("eric", "shirley"), ("jonathan", "anna"), ("stephen", "brenda"),
    ("larry", "pamela"), ("justin", "nicole"), ("scott", "samantha"),
    ("brandon", "katherine"), ("benjamin", "christine"), ("samuel", "helen"),
    ("gregory", "debra"), ("alexander", "rachel"), ("patrick", "carolyn"),
    ("frank", "janet"), ("raymond", "maria"), ("jack", "catherine"),
    ("dennis", "heather"), ("jerry", "diane"), ("tyler", "olivia"),
    ("aaron", "julie"), ("jose", "joyce"), ("adam", "victoria"),
    ("nathan", "kelly"), ("henry", "christina"), ("zachary", "lauren"),
    ("douglas", "joan"), ("peter", "evelyn"), ("kyle", "judith"),
]
# fmt: on

# --------------------------------------------------------------------------
# Table 1, row 2: ethnicity-associated name pairs (20)
# Following the audit-study name lists of Bertrand & Mullainathan (2004).
# Deliberately disjoint from GENDERED_NAMES so the two passes cannot collide.
# --------------------------------------------------------------------------
# fmt: off
ETHNICITY_NAMES: list[tuple[str, str]] = [
    ("connor", "jamal"), ("cody", "darnell"), ("dustin", "tyrone"),
    ("todd", "leroy"), ("neil", "rasheed"), ("geoffrey", "kareem"),
    ("brett", "terrence"), ("wyatt", "hakim"), ("jay", "deshawn"),
    ("logan", "malik"), ("allison", "tanisha"), ("abigail", "latoya"),
    ("molly", "ebony"), ("meredith", "aisha"), ("carrie", "keisha"),
    ("kristen", "tamika"), ("laurie", "latonya"), ("jill", "nichelle"),
    ("katelyn", "shanice"), ("claire", "imani"),
]
# fmt: on

# --------------------------------------------------------------------------
# Table 1, row 3: pronoun / gendered-term swaps (22)
# --------------------------------------------------------------------------
# fmt: off
GENDERED_TERMS: list[tuple[str, str]] = [
    ("he", "she"), ("him", "her"), ("his", "hers"), ("himself", "herself"),
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"),
    ("father", "mother"), ("dad", "mom"), ("son", "daughter"),
    ("brother", "sister"), ("husband", "wife"), ("uncle", "aunt"),
    ("nephew", "niece"), ("grandfather", "grandmother"), ("king", "queen"),
    ("prince", "princess"), ("mr", "mrs"), ("sir", "madam"),
    ("gentleman", "lady"), ("male", "female"),
]
# fmt: on

ALL_PAIRS = GENDERED_NAMES + ETHNICITY_NAMES + GENDERED_TERMS


def _build_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Bidirectional lookup: a -> b and b -> a."""
    m: dict[str, str] = {}
    for a, b in pairs:
        m[a] = b
        m[b] = a
    return m


SWAP_MAP: dict[str, str] = _build_map(ALL_PAIRS)

# Subgroup membership sets (lowercase, word-level)
MALE_TERMS = {a for a, _ in GENDERED_TERMS}
FEMALE_TERMS = {b for _, b in GENDERED_TERMS}
MALE_NAMES = {a for a, _ in GENDERED_NAMES}
FEMALE_NAMES = {b for _, b in GENDERED_NAMES}
ETHNICITY_ALL = {n for pair in ETHNICITY_NAMES for n in pair}

_TOKEN_RE = re.compile(r"\b\w+\b")

# ---------------------------------------------------------------------------
# Ambiguous pronouns.
#
# English overloads two of the words in Table 1:
#     his  = possessive determiner ("his book")  AND  possessive pronoun ("it is his")
#     her  = possessive determiner ("her book")  AND  object pronoun ("I saw her")
#
# A flat lookup can't tell them apart, so a naive table produces "hers book" and
# "him credit score". Those are ungrammatical, and an ungrammatical counterfactual
# confounds the flip rate: the model may flip because the sentence broke, not
# because the identity changed. We disambiguate with a cheap syntactic test —
# a possessive determiner is followed by a noun phrase, the pronoun forms are not.
# ---------------------------------------------------------------------------
AMBIGUOUS_PRONOUNS = {"his", "her"}

# If one of these follows "his"/"her", the word is NOT acting as a possessive
# determiner (e.g. "I gave her the book" -> object pronoun, not "her book").
_FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "hers",
    "its",
    "our",
    "their",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "by",
    "as",
    "into",
    "and",
    "or",
    "but",
    "if",
    "so",
    "then",
    "than",
    "because",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "am",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "not",
    "no",
    "all",
    "some",
    "any",
    "back",
    "again",
    "too",
    "very",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "us",
    "them",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
}

# Sentence or clause boundary right after the word means nothing follows it.
_BOUNDARY_RE = re.compile(r"[.!?,;:\"')\]]")


def _acts_as_determiner(text: str, end: int) -> bool:
    """True if the word ending at `end` is followed by a noun phrase."""
    tail = text[end:]
    gap = re.match(r"[^\w]*", tail).group(0)
    if _BOUNDARY_RE.search(gap):
        return False
    m = re.match(r"[^\w]*(\w+)", tail)
    if not m:
        return False
    return m.group(1).lower() not in _FUNCTION_WORDS


def _resolve_ambiguous(tok_lower: str, text: str, end: int) -> str:
    determiner = _acts_as_determiner(text, end)
    if tok_lower == "his":
        return "her" if determiner else "hers"
    return "his" if determiner else "him"


def _match_case(source: str, target: str) -> str:
    """Carry the original token's capitalisation onto the replacement."""
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target.capitalize()
    return target


def swap_identity(
    text: str,
    p: float = 1.0,
    rng: random.Random | None = None,
    contextual_pronouns: bool = True,
) -> str:
    """Replace identity tokens with their counterfactual counterparts.

    Args:
        text: input sentence.
        p: per-token probability of applying a swap. ``p=1.0`` is the
           deterministic swap used at evaluation time (flip rate); ``p=0.5``
           is the stochastic augmentation used during CDA training.
        rng: optional Random instance for reproducibility.
        contextual_pronouns: resolve "his"/"her" by syntactic role instead of a
           flat lookup. Keeps counterfactuals grammatical so the flip rate
           measures identity rather than broken syntax. Set False to reproduce
           the report's literal Table 1 mapping (his -> hers always).
    """
    r = rng or random

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        low = tok.lower()
        sub = SWAP_MAP.get(low)
        if not sub:
            return tok
        if p < 1.0 and r.random() >= p:
            return tok
        if contextual_pronouns and low in AMBIGUOUS_PRONOUNS:
            sub = _resolve_ambiguous(low, text, m.end())
        return _match_case(tok, sub)

    return _TOKEN_RE.sub(repl, text)


def contains_identity(text: str) -> bool:
    """True if any identity token from any category appears in the text."""
    return any(t.lower() in SWAP_MAP for t in _TOKEN_RE.findall(text))


def subgroup_of(s1: str, s2: str) -> str:
    """Assign a question pair to one of: male / female / name-only / neutral / mixed.

    Rules (Section 6.2 of the paper, made explicit):
      - "male"      -> gendered pronouns/nouns or gendered names, male side only
      - "female"    -> same, female side only
      - "name-only" -> identity names present but no gendered pronoun/noun,
                       or only ethnicity-associated names
      - "mixed"     -> both male and female cues present (excluded from the gap)
      - "neutral"   -> no identity terms at all
    """
    toks = {t.lower() for t in _TOKEN_RE.findall(f"{s1} {s2}")}

    has_male_term = bool(toks & MALE_TERMS)
    has_female_term = bool(toks & FEMALE_TERMS)
    has_male_name = bool(toks & MALE_NAMES)
    has_female_name = bool(toks & FEMALE_NAMES)
    has_eth_name = bool(toks & ETHNICITY_ALL)

    male = has_male_term or has_male_name
    female = has_female_term or has_female_name

    if male and female:
        return "mixed"
    if male:
        return "male"
    if female:
        return "female"
    if has_eth_name:
        return "name-only"
    return "neutral"


if __name__ == "__main__":
    assert len(GENDERED_NAMES) == 60, len(GENDERED_NAMES)
    assert len(ETHNICITY_NAMES) == 20, len(ETHNICITY_NAMES)
    assert len(GENDERED_TERMS) == 22, len(GENDERED_TERMS)
    overlap = {n for p in GENDERED_NAMES for n in p} & {n for p in ETHNICITY_NAMES for n in p}
    assert not overlap, overlap
    print(swap_identity("Why did James tell his brother that Connor was late?"))
    print(subgroup_of("Is he a good king?", "Was the king kind?"))
    print("lexicons OK")
