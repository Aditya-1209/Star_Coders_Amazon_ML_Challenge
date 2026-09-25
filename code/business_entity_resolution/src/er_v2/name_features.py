"""Pair-local evidence for spelling/transliteration and missing addresses.

No corpus frequencies, country sizes or labels enter these features. Token
alignment is directional mean-best Jaro-Winkler (Monge-Elkan), truncated to
12 tokens per name, in 10k-pair windows to bound the cross-token table.
"""
import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler

from .phonetic import phonetic_expr

EXTRA_FEATURES = ["token_align_l", "token_align_r", "token_align_min", "token_align_max",
                  "token_unmatched_l", "token_unmatched_r", "phonetic_jacc",
                  "phonetic_cont_l", "phonetic_cont_r", "noaddr_name_align",
                  "noaddr_name_exact", "name_length_balance", "name_token_balance"]


def _alignment(pairs: pl.DataFrame, workers: int) -> pl.DataFrame:
    results = []
    for part in pairs.iter_slices(10_000):
        p = part.select("ntok_l", "ntok_r").with_row_index("_row")
        left = p.select("_row", a=pl.col("ntok_l").list.head(12)).explode("a").drop_nulls()
        right = p.select("_row", b=pl.col("ntok_r").list.head(12)).explode("b").drop_nulls()
        cross = left.join(right, on="_row").filter((pl.col("a") != "") & (pl.col("b") != ""))
        sims = process.cpdist(cross["a"].to_list(), cross["b"].to_list(),
                              scorer=JaroWinkler.normalized_similarity,
                              workers=workers, dtype=np.float32)
        cross = cross.with_columns(sim=pl.Series(sims))
        out = p.select("_row")
        for column, side in (("a", "l"), ("b", "r")):
            best = cross.group_by("_row", column).agg(pl.col("sim").max())
            summary = best.group_by("_row").agg(
                pl.col("sim").mean().alias("token_align_" + side),
                (pl.col("sim") < 0.85).mean().cast(pl.Float32).alias("token_unmatched_" + side))
            out = out.join(summary, on="_row", how="left", maintain_order="left")
        results.append(out.drop("_row").fill_null(0.0))
    schema = {c: pl.Float32 for c in ("token_align_l", "token_unmatched_l",
                                     "token_align_r", "token_unmatched_r")}
    return pl.concat(results) if results else pl.DataFrame(schema=schema)


def compute_name_features(pairs: pl.DataFrame, workers: int) -> pl.DataFrame:
    out = _alignment(pairs, workers)
    codes = pairs.select(*[
        pl.col("ntok_" + side).list.eval(phonetic_expr(pl.element()))
        .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
        .list.unique().alias(side) for side in ("l", "r")])
    a, b = pl.col("l").list.len(), pl.col("r").list.len()
    inter = pl.col("l").list.set_intersection(pl.col("r")).list.len()
    safe = lambda x: pl.when(x > 0).then(x).otherwise(1)
    out = out.hstack(codes.select(
        phonetic_jacc=inter / safe(a + b - inter),
        phonetic_cont_l=inter / safe(a), phonetic_cont_r=inter / safe(b)))
    out = out.hstack(pairs.select(
        _noaddr=(pl.col("addr_l") == "") | (pl.col("addr_r") == ""),
        _equal=(pl.col("core_l") != "") & (pl.col("core_l") == pl.col("core_r")),
        name_length_balance=pl.min_horizontal(pl.col("core_l").str.len_chars(), pl.col("core_r").str.len_chars())
        / safe(pl.max_horizontal(pl.col("core_l").str.len_chars(), pl.col("core_r").str.len_chars())),
        name_token_balance=pl.min_horizontal(pl.col("ntok_l").list.len(), pl.col("ntok_r").list.len())
        / safe(pl.max_horizontal(pl.col("ntok_l").list.len(), pl.col("ntok_r").list.len()))))
    out = out.with_columns(
        token_align_min=pl.min_horizontal("token_align_l", "token_align_r"),
        token_align_max=pl.max_horizontal("token_align_l", "token_align_r"),
        noaddr_name_align=pl.when(pl.col("_noaddr")).then(pl.min_horizontal("token_align_l", "token_align_r")).otherwise(0),
        noaddr_name_exact=pl.col("_noaddr") & pl.col("_equal"))
    return out.select(pl.col(EXTRA_FEATURES).cast(pl.Float32))
