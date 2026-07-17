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


def _match_case(source: str, target: str) -> str:
    """Carry the original token's capitalisation onto the replacement."""
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target.capitalize()
    return target


def swap_identity(text: str, p: float = 1.0, rng: random.Random | None = None) -> str:
    """Replace identity tokens with their counterfactual counterparts.

    Args:
        text: input sentence.
        p: per-token probability of applying a swap. ``p=1.0`` is the
           deterministic swap used at evaluation time (flip rate); ``p=0.5``
           is the stochastic augmentation used during CDA training.
        rng: optional Random instance for reproducibility.
    """
    r = rng or random
    if p >= 1.0:

        def repl(m: re.Match) -> str:
            tok = m.group(0)
            sub = SWAP_MAP.get(tok.lower())
            return _match_case(tok, sub) if sub else tok
    else:

        def repl(m: re.Match) -> str:
            tok = m.group(0)
            sub = SWAP_MAP.get(tok.lower())
            if sub and r.random() < p:
                return _match_case(tok, sub)
            return tok

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
