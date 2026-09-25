"""Text normalization shared by blocking and feature generation.

Everything here is rule based and derived from the challenge data itself:
Unicode transliteration (anyascii, ISC license) plus a small canonical
abbreviation table. No external lookups.
"""
from __future__ import annotations

import re

from anyascii import anyascii

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LEAD_ZERO = re.compile(r"\b0+(\d)")

# Legal-form and filler words removed from the "core" business name.
LEGAL = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "llp", "lp", "plc", "pllc", "pc", "sarl",
    "sas", "sasu", "eurl", "sci", "ei", "sa", "snc", "the", "and", "et", "of",
    "m", "s", "ms", "smt", "shri", "sri", "www", "com", "net", "org", "in", "fr",
    "dba", "praivet", "elelpi", "limitid", "limted", "pvtltd",
}

# Transliteration artefacts and abbreviations mapped to one canonical token.
CANON = {
    # legal forms
    "incorporated": "inc", "limited": "ltd", "private": "pvt", "praivet": "pvt",
    "corporation": "corp", "company": "co", "elelpi": "llp",
    # street types (US / India / France)
    "street": "st", "saint": "st", "str": "st", "avenue": "ave", "av": "ave",
    "avn": "ave", "road": "rd", "drive": "dr", "lane": "ln", "boulevard": "blvd",
    "bd": "blvd", "bld": "blvd", "court": "ct", "terrace": "ter", "place": "pl",
    "highway": "hwy", "parkway": "pkwy", "circle": "cir", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt",
    "suite": "ste", "floor": "fl", "number": "no", "building": "bldg",
    "trail": "trl", "way": "wy", "route": "rte", "mount": "mt", "fort": "ft",
    "rue": "rue", "r": "rue", "allee": "all", "impasse": "imp", "chemin": "ch",
    "residence": "res", "quai": "qu", "cours": "crs",
    "nagar": "ngr", "colony": "col", "sector": "sec", "near": "nr", "opp": "opp",
    "opposite": "opp", "main": "mn", "cross": "crs",
    # US states
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
MULTI = [
    ("new hampshire", "nh"), ("new jersey", "nj"), ("new mexico", "nm"),
    ("new york", "ny"), ("north carolina", "nc"), ("north dakota", "nd"),
    ("rhode island", "ri"), ("south carolina", "sc"), ("south dakota", "sd"),
    ("west virginia", "wv"), ("district of columbia", "dc"),
    ("tamil nadu", "tn"), ("tmilnatu", "tn"), ("uttar pradesh", "up"),
    ("madhya pradesh", "mp"), ("andhra pradesh", "ap"), ("west bengal", "wb"),
    ("himachal pradesh", "hp"), ("arunachal pradesh", "ar"),
    ("new delhi", "delhi"),
]
_MULTI_RE = [(re.compile(r"\b" + a + r"\b"), b) for a, b in MULTI]
DROP_ADDR = {"null", "none", "nan"}


def ascii_lower(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("°", " ").replace("º", " ")
    if not text.isascii():
        text = anyascii(text)
    return text.lower()


def tokens(text: str) -> list[str]:
    text = text.replace("&", " and ").replace("'", "")
    text = _NON_ALNUM.sub(" ", text)
    return text.split()


_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e"})
_LEAD_L = re.compile(r"^l(?=[nm][aeiou])")


def fix_ocr(tok: str) -> str:
    """Undo digit-for-letter swaps (F0rt, J0nes) and l-for-i (lndustries)."""
    if tok.isdigit():
        return tok
    if any(ch.isdigit() for ch in tok) and sum(ch.isalpha() for ch in tok) >= 2:
        tok = tok.translate(_OCR)
    return _LEAD_L.sub("i", tok)


TRANSLIT: dict[str, str] = {}


def load_translit(path) -> None:
    """Install the learned Indic-transliteration token map (see translit.py)."""
    import json
    TRANSLIT.clear()
    if path:
        with open(path, encoding="utf-8") as fh:
            TRANSLIT.update(json.load(fh))


def norm_name(raw: str | None) -> tuple[str, str]:
    """Return (canonical full name, core name without legal words)."""
    toks = [fix_ocr(t) for t in tokens(ascii_lower(raw))]
    if TRANSLIT and raw and not raw.isascii():
        toks = [TRANSLIT.get(t, t) for t in toks]
    full = [CANON.get(t, t) for t in toks]
    core = [t for t in toks if t not in LEGAL]
    core = [CANON.get(t, t) for t in core]
    return " ".join(full), " ".join(core)


def norm_addr(raw: str | None) -> str:
    text = ascii_lower(raw)
    text = text.replace("'", "")
    text = _NON_ALNUM.sub(" ", text)
    for pat, rep in _MULTI_RE:
        text = pat.sub(rep, text)
    text = _LEAD_ZERO.sub(r"\1", text)
    out = [CANON.get(t, t) for t in text.split() if t not in DROP_ADDR]
    return " ".join(out)
