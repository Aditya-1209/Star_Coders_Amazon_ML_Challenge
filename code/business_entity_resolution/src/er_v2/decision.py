"""Per-business decision rule that maximizes expected macro F0.5.

For one Source 1 record with candidate probabilities p (sorted descending),
predicting the top-k candidates has plug-in expected score
    E[F0.5 | k] ~= 1.25 * sum(p[:k]) / (0.25 * (sum(p) + miss) + k)     (k >= 1)
    E[F0.5 | 0]  = P(no true match) ~= prod(1 - p) * exp(-miss)
where ``miss`` is the expected number of true matches that blocking never
retrieved (estimated on validation). We choose the k with the highest
expected score, then give each Source 2/3 record to one business only.
"""
from __future__ import annotations

import polars as pl


def best_per_target(pred: pl.DataFrame, score: str) -> pl.DataFrame:
    """Break equal scores by the lowest Source 1 row index, reproducibly."""
    return (pred.filter(pl.col(score).is_finite())
            .sort(["tidx", score, "sidx"], descending=[False, True, False])
            .unique("tidx", keep="first", maintain_order=True))


def expected_f05_select(pred: pl.DataFrame, score: str = "p2", miss: float = 0.25,
                        floor: float = 0.05) -> pl.DataFrame:
    """Return the chosen (sidx, tidx, score) pairs."""
    df = pred.filter(pl.col(score) >= floor).sort(["sidx", score, "tidx"], descending=[False, True, False])
    p = pl.col(score).cast(pl.Float64)
    # Stats for the empty prediction use all candidates, including those below the floor.
    tot = pred.group_by("sidx").agg(
        tot=pl.col(score).cast(pl.Float64).sum(),
        p_none=(1 - pl.col(score).cast(pl.Float64).clip(0, 0.999999)).log().sum().exp(),
    )
    df = df.join(tot, on="sidx", how="left").with_columns(
        k=pl.int_range(1, pl.len() + 1).over("sidx").cast(pl.Float64),
        cum=p.cum_sum().over("sidx"),
    )
    df = df.with_columns(ef=1.25 * pl.col("cum") / (0.25 * (pl.col("tot") + miss) + pl.col("k")))
    best = df.group_by("sidx").agg(
        best_ef=pl.col("ef").max(),
        best_k=pl.col("k").get(pl.col("ef").arg_max()),
        p_none=pl.col("p_none").first(),
    ).with_columns(empty_ef=pl.col("p_none") * pl.lit(miss).neg().exp())
    keep = best.filter(pl.col("best_ef") > pl.col("empty_ef")).select("sidx", "best_k")
    chosen = df.join(keep, on="sidx").filter(pl.col("k") <= pl.col("best_k"))
    chosen = best_per_target(chosen, score)
    return chosen.select("sidx", "tidx", score)
