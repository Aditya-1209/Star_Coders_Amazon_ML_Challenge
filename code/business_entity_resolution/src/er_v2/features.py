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
    ("raw_name", "raw_name_ratio", fuzz.ratio),
    ("raw_name", "raw_name_tset", fuzz.token_set_ratio),
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
        country=pl.col("country"),
        raw_name=pl.col("business_name").fill_null("").str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ").str.strip_chars(),
        name=pl.col("name_n"),
        core=pl.col("core_n"),
        cc=pl.col("core_n").str.replace_all(" ", ""),
        addr=pl.col("addr_n"),
        ntok=pl.col("core_n").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True),
        atok=pl.col("addr_n").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True),
        name_num=pl.col("business_name").fill_null("").str.extract_all(r"\d+").list.unique(),
        addr_num=pl.col("addr_n").str.extract_all(r"\d+").list.unique(),
        house=pl.col("addr_n").str.extract(r"\b(\d{1,4}[a-z]?)\b", 1).fill_null(""),
        # Read raw addresses: normalization strips leading zeroes from ZIPs.
        # These are fallible features, never hard matching constraints.
        postcode=pl.when(pl.col("country") == "India")
        .then(pl.col("business_address").str.extract_all(r"\b\d{6}\b").list.last())
        .otherwise(pl.col("business_address").str.extract_all(r"\b\d{5}\b").list.last()).fill_null(""),
        nonlatin=(~pl.col("business_name").str.contains(r"^[\x00-\x7FÀ-ɏ]*$")).cast(pl.Int8),
    ).with_columns(
        dtok=pl.col("atok").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))),
    )


def record_frames(s1: pl.DataFrame, tg: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    left = _record_cols(s1)
    right = _record_cols(tg)
    # A common chain name is weaker evidence than a unique business name.
    # Counts use only unlabelled records in this split, never ground truth.
    counts = right.group_by("country", "core").agg(name_frequency=pl.len().cast(pl.UInt32))
    left = left.join(counts, on=["country", "core"], how="left").with_columns(
        pl.col("name_frequency").fill_null(0))
    right = right.join(counts, on=["country", "core"], how="left")
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


def compute(pairs: pl.DataFrame, workers: int = -1) -> pl.DataFrame:
    feats = {}
    for field, name, scorer in STRING_FEATURES:
        a = pairs[field + "_l"].to_list()
        b = pairs[field + "_r"].to_list()
        values = process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)
        present = ((pairs[field + "_l"].fill_null("") != "")
                   & (pairs[field + "_r"].fill_null("") != "")).to_numpy()
        values[~present] = 0.0
        feats[name] = values
    f = pl.DataFrame(feats)
    tok = pairs.select(
        *_jacc("ntok_l", "ntok_r", "ntok"),
        *_jacc("atok_l", "atok_r", "atok"),
        *_jacc("dtok_l", "dtok_r", "dtok"),
        *_jacc("name_num_l", "name_num_r", "name_num"),
        *_jacc("addr_num_l", "addr_num_r", "addr_num"),
        name_num_conflict=_number_conflict("name_num"),
        house_equal=_both_present("house") & (pl.col("house_l") == pl.col("house_r")),
        house_conflict=_both_present("house") & (pl.col("house_l") != pl.col("house_r")),
        postcode_equal=_both_present("postcode") & (pl.col("postcode_l") == pl.col("postcode_r")),
        postcode_conflict=_both_present("postcode") & (pl.col("postcode_l") != pl.col("postcode_r")),
        postcode_both_present=_both_present("postcode"),
        addr_both_present=_both_present("addr"),
        raw_name_equal=_both_present("raw_name") & (pl.col("raw_name_l") == pl.col("raw_name_r")),
        name_frequency_l=pl.col("name_frequency_l").cast(pl.Float32).log1p(),
        name_frequency_r=pl.col("name_frequency_r").cast(pl.Float32).log1p(),
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


def _both_present(field: str) -> pl.Expr:
    return ((pl.col(field + "_l").fill_null("") != "")
            & (pl.col(field + "_r").fill_null("") != ""))


def _number_conflict(field: str) -> pl.Expr:
    a, b = pl.col(field + "_l"), pl.col(field + "_r")
    return ((a.list.len() > 0) & (b.list.len() > 0)
            & (a.list.set_intersection(b).list.len() == 0))


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
