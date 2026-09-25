"""Stage 3: pairwise features for candidate pairs.

String similarities come from RapidFuzz's multi-threaded element-wise
``cpdist``; token-set statistics use Polars list operations. Blocking-derived
features describe how a pair ranks among the candidates of its Source 1
record and among the Source 1 records competing for the same target.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

STRING_FEATURES = [
    ("core", "core_ratio", fuzz.ratio),
    ("core", "core_tsort", fuzz.token_sort_ratio),
    ("core", "core_tset", fuzz.token_set_ratio),
    ("core", "core_partial", fuzz.partial_ratio),
    ("core", "core_jw", JaroWinkler.normalized_similarity),
    ("name", "name_ratio", fuzz.ratio),
    ("name", "name_tset", fuzz.token_set_ratio),
    ("cc", "cc_ratio", fuzz.ratio),
    ("cc", "cc_partial", fuzz.partial_ratio),
    ("addr", "addr_ratio", fuzz.ratio),
    ("addr", "addr_tsort", fuzz.token_sort_ratio),
    ("addr", "addr_tset", fuzz.token_set_ratio),
    ("addr", "addr_partial", fuzz.partial_ratio),
]


def _record_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Per-record derived columns used by the pair features."""
    return df.select(
        "idx",
        name=pl.col("name_n"),
        core=pl.col("core_n"),
        cc=pl.col("core_n").str.replace_all(" ", ""),
        addr=pl.col("addr_n"),
        ntok=pl.col("core_n").str.split(" ").list.eval(pl.element().filter(pl.element() != "")),
        atok=pl.col("addr_n").str.split(" ").list.eval(pl.element().filter(pl.element() != "")),
        nonlatin=(~pl.col("business_name").str.contains(r"^[\x00-\x7FÀ-ɏ]*$")).cast(pl.Int8),
    ).with_columns(
        dtok=pl.col("atok").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))),
    )


def record_frames(s1: pl.DataFrame, tg: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    left = _record_cols(s1)
    right = _record_cols(tg)
    left = left.rename({c: c + "_l" for c in left.columns if c != "idx"})
    right = right.rename({c: c + "_r" for c in right.columns if c != "idx"})
    return left, right


def pair_frame(cands: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame, n_s2: int) -> pl.DataFrame:
    """Attach left/right record columns to candidate pairs (keeps order)."""
    out = cands.join(left, left_on="sidx", right_on="idx", how="left", maintain_order="left")
    out = out.join(right, left_on="tidx", right_on="idx", how="left", maintain_order="left")
    return out.with_columns(is_s3=(pl.col("tidx") >= n_s2).cast(pl.Int8))


def _jacc(a: str, b: str, prefix: str) -> list[pl.Expr]:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len().cast(pl.Float32)
    la = pl.col(a).list.len().cast(pl.Float32)
    lb = pl.col(b).list.len().cast(pl.Float32)
    union = la + lb - inter
    return [
        (inter / pl.when(union > 0).then(union).otherwise(1)).alias(prefix + "_jacc"),
        (inter / pl.when(la > 0).then(la).otherwise(1)).alias(prefix + "_cont_l"),
        (inter / pl.when(lb > 0).then(lb).otherwise(1)).alias(prefix + "_cont_r"),
        inter.alias(prefix + "_inter"),
    ]


def compute(pairs: pl.DataFrame) -> pl.DataFrame:
    feats = {}
    for field, name, scorer in STRING_FEATURES:
        a = pairs[field + "_l"].to_list()
        b = pairs[field + "_r"].to_list()
        feats[name] = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    f = pl.DataFrame(feats)
    tok = pairs.select(
        *_jacc("ntok_l", "ntok_r", "ntok"),
        *_jacc("atok_l", "atok_r", "atok"),
        *_jacc("dtok_l", "dtok_r", "dtok"),
        first_num_eq=(pl.col("dtok_l").list.first() == pl.col("dtok_r").list.first()).fill_null(False).cast(pl.Int8),
        num_conflict=((pl.col("dtok_l").list.len() > 0) & (pl.col("dtok_r").list.len() > 0)
                      & (pl.col("dtok_l").list.set_intersection(pl.col("dtok_r")).list.len() == 0)).cast(pl.Int8),
        len_core_l=pl.col("core_l").str.len_chars(),
        len_core_r=pl.col("core_r").str.len_chars(),
        ntok_r=pl.col("ntok_r").list.len(),
        addr_len_l=pl.col("addr_l").str.len_chars(),
        addr_len_r=pl.col("addr_r").str.len_chars(),
        nonlatin_r=pl.col("nonlatin_r"),
        nonlatin_l=pl.col("nonlatin_l"),
        is_s3=pl.col("is_s3"),
        bscore=pl.col("bscore"),
        nkeys=pl.col("nkeys"),
        brank=pl.col("brank"),
        b_rel_s=pl.col("b_rel_s"),
        b_rel_t=pl.col("b_rel_t"),
        t_rank=pl.col("t_rank"),
        t_nc=pl.col("t_nc"),
        s_nc=pl.col("s_nc"),
    )
    return pl.concat([pairs.select("sidx", "tidx"), f, tok], how="horizontal")


def add_block_context(cands: pl.DataFrame) -> pl.DataFrame:
    """Blocking-score context across a Source 1's candidates and a target's claimants."""
    return cands.with_columns(
        b_rel_s=(pl.col("bscore") / pl.col("bscore").max().over("sidx")).cast(pl.Float32),
        s_nc=pl.len().over("sidx").cast(pl.UInt16),
        b_rel_t=(pl.col("bscore") / pl.col("bscore").max().over("tidx")).cast(pl.Float32),
        t_rank=pl.col("bscore").rank("ordinal", descending=True).over("tidx").cast(pl.UInt16),
        t_nc=pl.len().over("tidx").cast(pl.UInt32),
    )


FEATURE_COLS = None  # filled lazily: every column except the pair ids


def feature_names(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("sidx", "tidx", "label")]
