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
