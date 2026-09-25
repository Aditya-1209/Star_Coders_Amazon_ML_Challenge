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


def _count_over(df: pl.DataFrame, col: str, alias: str) -> pl.Expr:
    """How many records of the same country share this exact non-empty value."""
    return (pl.when(pl.col(col) != "").then(pl.len().over("country", col)).otherwise(0)
            .cast(pl.UInt32).alias(alias))


def record_frames(s1: pl.DataFrame, tg: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-record columns (suffix _l for Source 1, _r for Source 2/3).

    Genericness counts are computed on the split being scored (train on train,
    test on test), per country, so they exist for unseen countries like France:
    a name shared by many businesses is weak evidence, a unique one is strong.
    """
    left = _record_cols(s1)
    right = _record_cols(tg)
    # A common chain name is weaker evidence than a unique business name.
    # Counts use only unlabelled records in this split, never ground truth.
    counts = right.group_by("country", "core").agg(name_frequency=pl.len().cast(pl.UInt32))
    left = left.join(counts, on=["country", "core"], how="left").with_columns(
        pl.col("name_frequency").fill_null(0))
    right = right.join(counts, on=["country", "core"], how="left")
    # Same-side genericness (r5): shared S1 names, shared addresses, and how many
    # Source 1 businesses carry a target's name.
    left = left.with_columns(_count_over(left, "core", "name_cnt"), _count_over(left, "addr", "addr_cnt"))
    right = right.with_columns(_count_over(right, "addr", "addr_cnt"))
    in_s1 = left.filter(pl.col("core") != "").group_by("country", "core").agg(name_in_s1=pl.len())
    right = right.join(in_s1, on=["country", "core"], how="left").with_columns(
        pl.col("name_in_s1").fill_null(0).cast(pl.UInt32))
    left = left.rename({c: c + "_l" for c in left.columns if c != "idx"})
    right = right.rename({c: c + "_r" for c in right.columns if c != "idx"})
    return left, right


UNSEEN_W = 15.0  # IDF weight for a token never seen among targets (~log of a 3M corpus)


def token_idf(tg: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Per-country IDF of name and address tokens among the split's Source 2/3 records.

    Computed from the split's own records only (no labels), so rare words dominate
    and region/department names or generic words ("club", "comite") stop mattering.
    """
    n = tg.group_by("country").agg(N=pl.len())
    out = {}
    for kind, col in (("n", "core_n"), ("a", "addr_n")):
        e = (tg.select("idx", "country", tok=pl.col(col).str.split(" ")).explode("tok")
             .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique(["idx", "tok"]))
        d = e.group_by("country", "tok").agg(df=pl.len()).join(n, on="country")
        out[kind] = d.select("country", "tok",
                             w=(pl.col("N").cast(pl.Float64) / pl.col("df")).log().cast(pl.Float32))
    return out


def _weighted_overlap(pairs: pl.DataFrame, lcol: str, rcol: str, idf: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """IDF-weighted Jaccard / containment and the heaviest unmatched token per side."""
    r = pairs.select(_r=pl.int_range(pl.len(), dtype=pl.UInt32), country=pl.col("country_l"),
                     L=pl.col(lcol), R=pl.col(rcol))

    def side(col: str) -> pl.DataFrame:
        e = (r.select("_r", "country", tok=pl.col(col)).explode("tok")
             .filter(pl.col("tok").is_not_null()).unique(["_r", "tok"]))
        return e.join(idf, on=["country", "tok"], how="left").with_columns(pl.col("w").fill_null(UNSEEN_W))

    L, R = side("L"), side("R")
    keys = ["_r", "tok"]
    agg = lambda df, name, how: df.group_by("_r").agg(getattr(pl.col("w"), how)().alias(name))
    parts = [agg(L, "tl", "sum"), agg(R, "tr", "sum"),
             agg(L.join(R.select(keys), on=keys, how="semi"), "ti", "sum"),
             agg(L.join(R.select(keys), on=keys, how="anti"), "ul", "max"),
             agg(R.join(L.select(keys), on=keys, how="anti"), "ur", "max")]
    out = r.select("_r")
    for p in parts:
        out = out.join(p, on="_r", how="left", maintain_order="left")
    out = out.fill_null(0.0)
    safe = lambda c: pl.when(c > 0).then(c).otherwise(1.0)
    union = pl.col("tl") + pl.col("tr") - pl.col("ti")
    return out.select(
        (pl.col("ti") / safe(union)).cast(pl.Float32).alias(prefix + "_jacc"),
        (pl.col("ti") / safe(pl.col("tl"))).cast(pl.Float32).alias(prefix + "_cont_l"),
        (pl.col("ti") / safe(pl.col("tr"))).cast(pl.Float32).alias(prefix + "_cont_r"),
        pl.col("ul").cast(pl.Float32).alias(prefix + "_unmatched_l"),
        pl.col("ur").cast(pl.Float32).alias(prefix + "_unmatched_r"),
    )


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


def compute(pairs: pl.DataFrame, workers: int = -1, idf: dict[str, pl.DataFrame] | None = None,
            enhanced: bool = False) -> pl.DataFrame:
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
        rescue=(pl.col("rescue") if "rescue" in pairs.columns else pl.lit(None, pl.Int8)),
        name_cnt_l=pl.col("name_cnt_l").log1p().cast(pl.Float32),
        addr_cnt_l=pl.col("addr_cnt_l").log1p().cast(pl.Float32),
        addr_cnt_r=pl.col("addr_cnt_r").log1p().cast(pl.Float32),
        name_in_s1_r=pl.col("name_in_s1_r").log1p().cast(pl.Float32),
    )
    frames = [pairs.select("sidx", "tidx"), f, tok]
    if idf is not None:
        frames.append(_weighted_overlap(pairs, "ntok_l", "ntok_r", idf["n"], "wn"))
        frames.append(_weighted_overlap(pairs, "atok_l", "atok_r", idf["a"], "wa"))
    if enhanced:
        from .name_features import compute_name_features
        frames.append(compute_name_features(pairs, workers))
    return pl.concat(frames, how="horizontal")


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


# Features whose scale depends on how large the split is (absolute counts, and
# IDF values log(N/df) with N = records per country). Train and test differ in
# size (e.g. US: 1.32M vs 0.66M businesses, 6.2M vs 3.8M records), so these
# made train and test pairs separable (adversarial AUC 0.93-0.97 vs ~0.72
# without them) and cost ~0.8 leaderboard points despite a better holdout.
# They are still computed but never used by any model.
SPLIT_DEPENDENT = {
    "name_frequency_l", "name_frequency_r", "name_cnt_l", "addr_cnt_l", "addr_cnt_r", "name_in_s1_r",
    "wn_jacc", "wn_cont_l", "wn_cont_r", "wn_unmatched_l", "wn_unmatched_r",
    "wa_jacc", "wa_cont_l", "wa_cont_r", "wa_unmatched_l", "wa_unmatched_r",
}


def feature_names(df: pl.DataFrame, profile: str = "enhanced") -> list[str]:
    from .name_features import EXTRA_FEATURES
    excluded = SPLIT_DEPENDENT | {"sidx", "tidx", "label", "fold", "w"}
    if profile == "baseline":
        excluded |= set(EXTRA_FEATURES)
    return [c for c in df.columns if c not in excluded]
