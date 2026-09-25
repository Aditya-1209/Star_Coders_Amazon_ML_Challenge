"""Stage 2: candidate generation with weighted inverted-key blocking.

Every record is exploded into blocking keys (country-scoped):
  n  core-name token            c  first 8 chars of the space-free core name
  p  adjacent core-name pair    a  address token containing a digit
  w  alphabetic address token   q  address number + following token
Keys whose document frequency among Source 2/3 records exceeds a per-type cap
are ignored (they are uninformative and explode the join). Each surviving
shared key contributes its IDF weight. Direct retrieval unions the combined
top-K with separate name/address rankings; two-hop queries use combined top-K.
"""
from __future__ import annotations

import math
import time

import polars as pl

CAPS = {"n": 2000, "c": 500, "p": 500, "a": 2000, "w": 2000, "q": 500}
PAIR_SCHEMA = {"sidx": pl.UInt32, "tidx": pl.UInt32, "bscore": pl.Float32,
               "nkeys": pl.UInt32, "brank": pl.UInt32}


def _tok(col: str, alias: str) -> pl.Expr:
    return pl.col(col).str.split(" ").alias(alias)


def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Return (idx, kind, key) with key a u64 hash scoped by country."""
    base = df.select("idx", "country", _tok("core_n", "nt"), _tok("addr_n", "at"), "core_n")
    name = (
        base.select("idx", "country", "nt").explode("nt").filter(pl.col("nt").str.len_chars() >= 2)
        .with_columns(nxt=pl.col("nt").shift(-1).over("idx"))
    )
    addr = (
        base.select("idx", "country", "at").explode("at").filter(pl.col("at").str.len_chars() >= 1)
        .with_columns(nxt=pl.col("at").shift(-1).over("idx"))
    )
    has_digit = pl.col("at").str.contains(r"\d")
    parts = [
        name.select("idx", "country", kind=pl.lit("n"), s=pl.col("nt")),
        name.filter(pl.col("nxt").is_not_null()).select(
            "idx", "country", kind=pl.lit("p"),
            s=pl.min_horizontal("nt", "nxt") + "_" + pl.max_horizontal("nt", "nxt")),
        base.filter(pl.col("core_n").str.replace_all(" ", "").str.len_chars() >= 5).select(
            "idx", "country", kind=pl.lit("c"),
            s=pl.col("core_n").str.replace_all(" ", "").str.slice(0, 8)),
        addr.filter(has_digit).select("idx", "country", kind=pl.lit("a"), s=pl.col("at")),
        addr.filter(~has_digit & (pl.col("at").str.len_chars() >= 4)).select(
            "idx", "country", kind=pl.lit("w"), s=pl.col("at")),
        addr.filter(has_digit & pl.col("nxt").is_not_null()).select(
            "idx", "country", kind=pl.lit("q"), s=pl.col("at") + "_" + pl.col("nxt")),
    ]
    keys = pl.concat(parts)
    keys = keys.with_columns(
        key=(pl.col("kind") + "|" + pl.col("country") + "|" + pl.col("s")).hash(seed=7)
    ).select(pl.col("idx").cast(pl.UInt32), "kind", "key").unique(["idx", "key"])
    return keys


def build_target_index(tkeys: pl.DataFrame, n_targets: int, caps=CAPS) -> pl.DataFrame:
    if n_targets < 1:
        raise ValueError("The target corpus must contain at least one record")
    df = tkeys.group_by("key", "kind").agg(pl.len().alias("df"))
    cap = pl.col("kind").replace_strict(caps, return_dtype=pl.UInt32)
    df = df.filter(pl.col("df") <= cap)
    df = df.with_columns(w=(math.log(n_targets) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32))
    idx = tkeys.join(df.select("key", "w"), on="key", how="inner").select("key", "idx", "w", "kind")
    return idx.sort("key")


def generate(skeys: pl.DataFrame, tindex: pl.DataFrame, top_k: int = 50,
             chunk: int = 100_000, verbose: bool = True,
             name_k: int = 0, address_k: int = 0) -> pl.DataFrame:
    """Union the combined top-K with optional name/address-only top-K lists.

    Separate lists rescue name matches crowded out by address collisions and
    address matches whose names have severe OCR/transliteration noise. The
    original combined top-K is always retained. Graph expansion uses no extras.
    """
    if top_k < 1 or chunk < 1:
        raise ValueError("top_k and chunk must be positive")
    if name_k < 0 or address_k < 0:
        raise ValueError("channel candidate limits cannot be negative")
    diverse = bool(name_k or address_k)
    if diverse and "kind" not in tindex.columns:
        raise ValueError("Channel retrieval requires a target index with key kinds")
    keyset = tindex.select("key", "w").unique("key")
    sk = skeys.join(keyset, on="key", how="inner").select("idx", "key")
    ids = sk["idx"].unique().sort()
    out = []
    t0 = time.time()
    for i in range(0, len(ids), chunk):
        lo, hi = ids[i], ids[min(i + chunk, len(ids)) - 1]
        part = sk.filter(pl.col("idx").is_between(lo, hi))
        j = part.join(tindex, on="key", how="inner", suffix="_t")
        expressions = [pl.col("w").sum().alias("bscore"), pl.len().alias("nkeys")]
        if diverse:
            expressions.extend([
                pl.col("w").filter(pl.col("kind").is_in(["n", "c", "p"])).sum().alias("name_score"),
                pl.col("w").filter(pl.col("kind").is_in(["a", "w", "q"])).sum().alias("address_score"),
            ])
        g = j.group_by("idx", "idx_t").agg(expressions)
        # Hash-group iteration order is not a stable tie breaker.
        g = g.sort(["idx", "bscore", "idx_t"], descending=[False, True, False]).with_columns(
            brank=pl.int_range(1, pl.len() + 1).over("idx").cast(pl.UInt32)
        )
        keep = pl.col("brank") <= top_k
        # Start each ordinal ranking in target-ID order for reproducible ties.
        for channel, limit in (("name", name_k), ("address", address_k)):
            if limit:
                col = channel + "_score"
                g = g.sort("idx", "idx_t").with_columns(
                    pl.col(col).rank("ordinal", descending=True).over("idx").alias(channel + "_rank"))
                keep = keep | ((pl.col(col) > 0) & (pl.col(channel + "_rank") <= limit))
        g = g.filter(keep).rename({"idx": "sidx", "idx_t": "tidx"}).select(list(PAIR_SCHEMA))
        out.append(g.cast(PAIR_SCHEMA))
        if verbose:
            print(f"  block {i + chunk:,}/{len(ids):,} joined={len(j):,} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    return pl.concat(out) if out else pl.DataFrame(schema=PAIR_SCHEMA)
