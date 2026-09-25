"""Stage 2: candidate generation with weighted inverted-key blocking.

Every record is exploded into blocking keys (country-scoped):
  n  core-name token            c  first 8 chars of the space-free core name
  p  adjacent core-name pair    a  address token containing a digit
  w  alphabetic address token   q  address number + following token
Keys whose document frequency among Source 2/3 records exceeds a per-type cap
are ignored (they are uninformative and explode the join). Each surviving
shared key contributes its IDF weight; the top-K targets per Source 1 record
by summed weight form the candidate set.
"""
from __future__ import annotations

import math
import time

import polars as pl

CAPS = {"n": 2000, "c": 500, "p": 500, "a": 2000, "w": 2000, "q": 500}


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
    df = tkeys.group_by("key", "kind").agg(pl.len().alias("df"))
    cap = pl.col("kind").replace_strict(caps, return_dtype=pl.UInt32)
    df = df.filter(pl.col("df") <= cap)
    df = df.with_columns(w=(math.log(n_targets) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32))
    idx = tkeys.join(df.select("key", "w"), on="key", how="inner").select("key", "idx", "w")
    return idx.sort("key")


def generate(skeys: pl.DataFrame, tindex: pl.DataFrame, top_k: int = 50,
             chunk: int = 100_000, verbose: bool = True) -> pl.DataFrame:
    """Return (sidx, tidx, bscore, nkeys, brank) for the top-K targets per source1."""
    keyset = tindex.select("key", "w").unique("key")
    sk = skeys.join(keyset, on="key", how="inner").select("idx", "key")
    ids = sk["idx"].unique().sort()
    out = []
    t0 = time.time()
    for i in range(0, len(ids), chunk):
        lo, hi = ids[i], ids[min(i + chunk, len(ids)) - 1]
        part = sk.filter(pl.col("idx").is_between(lo, hi))
        j = part.join(tindex, on="key", how="inner", suffix="_t")
        g = j.group_by("idx", "idx_t").agg(bscore=pl.col("w").sum(), nkeys=pl.len())
        g = g.with_columns(
            brank=pl.col("bscore").rank("ordinal", descending=True).over("idx")
        ).filter(pl.col("brank") <= top_k)
        out.append(g.rename({"idx": "sidx", "idx_t": "tidx"}))
        if verbose:
            print(f"  block {i + chunk:,}/{len(ids):,} joined={len(j):,} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    return pl.concat(out)
