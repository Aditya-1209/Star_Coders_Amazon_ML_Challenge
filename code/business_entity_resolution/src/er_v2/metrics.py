"""Macro F0.5 per Source 1 record, exactly as the challenge defines it."""
from __future__ import annotations

import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, anchors: pl.Series) -> dict:
    """pred/truth: (sidx, tidx) pairs; anchors: every evaluated sidx (incl. singletons)."""
    a = pl.DataFrame({"sidx": anchors.cast(pl.UInt32)}).unique("sidx")
    if a.is_empty():
        raise ValueError("Cannot evaluate macro F0.5 without Source 1 anchors")
    # The metric is defined on sets; duplicate rows must never multiply true positives.
    pred = pred.select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)).unique().join(a, on="sidx")
    truth = truth.select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)).unique().join(a, on="sidx")
    tp = pred.join(truth, on=["sidx", "tidx"]).group_by("sidx").agg(tp=pl.len())
    npred = pred.group_by("sidx").agg(np=pl.len())
    ntrue = truth.group_by("sidx").agg(nt=pl.len())
    df = a.join(tp, on="sidx", how="left").join(npred, on="sidx", how="left").join(ntrue, on="sidx", how="left").fill_null(0)
    df = df.with_columns(
        p=pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0),
        r=pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0),
    )
    df = df.with_columns(
        f=pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r")))
    )
    tot = df.select(pl.col("tp").sum(), pl.col("np").sum(), pl.col("nt").sum()).row(0)
    return {
        "macro_f05": df["f"].mean(),
        "pair_precision": tot[0] / max(tot[1], 1),
        "pair_recall": tot[0] / max(tot[2], 1),
        "anchors": len(df),
    }


# Test-set country mix (Source 1 rows): the leaderboard averages over these.
TEST_MIX = {"US": 663_106 / 1_732_544, "India": 809_986 / 1_732_544, "France": 259_452 / 1_732_544}
# France has no labels. Default estimate solved from leaderboard history:
# LB = mix-weighted(US, India holdout) + share_FR * F_FR  ->  F_FR ~ 0.87 (submission #1).
# Replace with the leave-one-country-out proxy (train.py --exclude-country India) when available.
FRANCE_DEFAULT = 0.87


def by_country(pred: pl.DataFrame, truth: pl.DataFrame, anchors: pl.DataFrame) -> dict:
    """anchors: (sidx, country) of every evaluated Source 1 record."""
    return {c: macro_f05(pred, truth, g["sidx"])
            for (c,), g in anchors.group_by("country") if len(g)}


def leaderboard_estimate(per_country: dict, france: float | None = None) -> dict:
    """Mix-weighted estimate of the leaderboard score from per-country holdout F0.5."""
    fr = FRANCE_DEFAULT if france is None else france
    known = {c: per_country[c]["macro_f05"] for c in per_country}
    est = sum(TEST_MIX[c] * known[c] for c in ("US", "India") if c in known) + TEST_MIX["France"] * fr
    return {"estimate": est, "france_assumed": fr, **{f"F_{c}": v for c, v in known.items()}}
