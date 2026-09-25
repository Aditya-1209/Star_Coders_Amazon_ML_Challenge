"""Unicode-preserving normalization and inexpensive pair features."""

from functools import lru_cache
import re
import unicodedata

from rapidfuzz import fuzz

LEGAL = frozenset("inc incorporated llc ltd limited private pvt corporation corp company co llp plc".split())
ADDRESS_WORDS = {"road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd", "drive": "dr", "lane": "ln", "apartment": "apt", "suite": "ste", "floor": "fl"}
ADDRESS_COMMON = frozenset("rd st ave blvd dr ln apt ste fl no near opposite plot door building".split())


def normalize(text):
    result = []
    for char in unicodedata.normalize("NFKC", text).casefold().replace("&", " and "):
        if "LATIN" in unicodedata.name(char, ""):
            char = "".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c))
        result.append(char if char.isalnum() or unicodedata.category(char).startswith("M") else " ")
    return " ".join("".join(result).split())


def name_core(name):
    tokens = name.split()
    core = " ".join(token for token in tokens if token not in LEGAL)
    return core or name


def address_normalize(text):
    return " ".join(ADDRESS_WORDS.get(token, token) for token in normalize(text).split())


def grams(text):
    return sorted({"g" + token[i:i + 3] for token in text.split() for i in range(max(0, len(token) - 2))})


def overlap(a, b):
    if not a or not b:
        return 0.0, 0.0
    common = len(a & b)
    return common / len(a | b), common / min(len(a), len(b))


@lru_cache(maxsize=40000)
def prepared(record):
    name = normalize(record.business_name)
    core = name_core(name)
    address = address_normalize(record.business_address)
    tokens = frozenset(core.split())
    address_tokens = frozenset(address.split())
    numbers = frozenset(re.findall(r"\d+", address))
    postal = frozenset(n for n in numbers if len(n) in (5, 6))
    first_number = next(iter(re.findall(r"\d+", address)), "")
    initials = "".join(t[0] for t in core.split())
    latin = sum("LATIN" in unicodedata.name(c, "") for c in core) / max(1, sum(c.isalpha() for c in core))
    return name, core, address, tokens, address_tokens, numbers, postal, first_number, initials, latin


FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "name_partial",
    "name_core_ratio", "name_core_exact", "name_token_jaccard", "name_token_containment",
    "name_length_ratio", "name_initials_equal", "name_is_other_initials", "latin_fraction_difference",
    "address_ratio", "address_token_sort", "address_token_set", "address_partial",
    "address_exact_nonempty", "address_token_jaccard", "address_token_containment",
    "address_missing_left", "address_missing_right", "number_jaccard", "number_containment",
    "first_number_equal", "first_number_conflict", "postal_shared", "postal_conflict",
    "name_rank_inverse", "address_rank_inverse", "gram_rank_inverse", "retrieval_channels",
]


def features(left, right, ranks):
    an, ac, aa, ant, aat, anum, apost, afirst, ai, alatin = prepared(left)
    bn, bc, ba, bnt, bat, bnum, bpost, bfirst, bi, blatin = prepared(right)
    nj, nc = overlap(ant, bnt)
    aj, az = overlap(aat, bat)
    dj, dc = overlap(anum, bnum)
    address_present = bool(aa and ba)
    return [
        fuzz.ratio(an, bn) / 100, fuzz.token_sort_ratio(an, bn) / 100,
        fuzz.token_set_ratio(an, bn) / 100, fuzz.partial_ratio(an, bn) / 100,
        fuzz.ratio(ac, bc) / 100, float(bool(ac) and ac == bc), nj, nc,
        min(len(ac), len(bc)) / max(1, len(ac), len(bc)),
        float(bool(ai) and ai == bi), float(ac.replace(" ", "") == bi or bc.replace(" ", "") == ai),
        abs(alatin - blatin),
        fuzz.ratio(aa, ba) / 100 if address_present else 0,
        fuzz.token_sort_ratio(aa, ba) / 100 if address_present else 0,
        fuzz.token_set_ratio(aa, ba) / 100 if address_present else 0,
        fuzz.partial_ratio(aa, ba) / 100 if address_present else 0,
        float(address_present and aa == ba), aj, az, float(not aa), float(not ba), dj, dc,
        float(bool(afirst and bfirst) and afirst == bfirst),
        float(bool(afirst and bfirst) and afirst != bfirst),
        float(bool(apost & bpost)), float(bool(apost and bpost) and not apost & bpost),
        *[1 / ranks[channel] if ranks[channel] else 0 for channel in range(3)],
        float(sum(bool(rank) for rank in ranks)),
    ]

