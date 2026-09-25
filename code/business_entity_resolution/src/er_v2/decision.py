"""Tune exclusive matching decisions directly for validation macro F0.5.

The optional legacy ``expected_f05_select`` below uses an approximation:
for one Source 1 record with candidate probabilities p (sorted descending),
predicting the top-k candidates has plug-in expected score
    E[F0.5 | k] ~= 1.25 * sum(p[:k]) / (0.25 * (sum(p) + miss) + k)     (k >= 1)
    E[F0.5 | 0]  = P(no true match) ~= prod(1 - p) * exp(-miss)
where ``miss`` is the expected number of true matches that blocking never
retrieved (estimated on validation). We choose the k with the highest
expected score, then give each Source 2/3 record to one business only.
"""
from __future__ import annotations

import polars as pl

NO_MATCH_THRESHOLD = 1.0000001  # finite JSON value, above any probability


def tune_threshold(pred: pl.DataFrame, truth: pl.DataFrame, anchors: pl.Series,
                   score: str = "p2") -> tuple[float, float]:
    """Find the exact best observed cutoff for exclusive macro F0.5.

    Ownership depends on score, not threshold, so resolve it once. Each added
    pair changes only its source business's score. Sum those changes at each
    distinct positive probability; tied probabilities enter together. Includes empty
    predictions and businesses with no retrieved candidates. Ties prefer the
    higher cutoff. Only tuning-fold labels may be passed here.
    """
    a = pl.DataFrame({"sidx": anchors.cast(pl.UInt32)}).unique()
    if a.is_empty():
        raise ValueError("Cannot tune without Source 1 anchors")
    truth = truth.select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)).unique()
    truth = truth.join(a, on="sidx", how="semi")
    counts = truth.group_by("sidx").agg(nt=pl.len())
    a = a.join(counts, on="sidx", how="left").with_columns(pl.col("nt").fill_null(0))
    initial = float((a["nt"] == 0).sum())
    evaluated = pred.select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32), score)
    winners = best_per_target(evaluated.join(a, on="sidx", how="semi"), score)
    winners = winners.filter(pl.col(score).is_between(0, 1, closed="right")).join(a, on="sidx")
    if winners.is_empty():
        return initial / len(a), NO_MATCH_THRESHOLD
    winners = winners.join(truth.with_columns(hit=pl.lit(1.0)), on=["sidx", "tidx"], how="left")
    winners = winners.with_columns(pl.col("hit").fill_null(0.0)).sort(
        [score, "sidx", "tidx"], descending=[True, False, False])
    winners = winners.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("sidx").cast(pl.Float64),
        tp=pl.col("hit").cum_sum().over("sidx"))
    denominator = 0.25 * pl.col("nt") + pl.col("k")
    previous = pl.when(pl.col("k") == 1).then((pl.col("nt") == 0).cast(pl.Float64)).otherwise(
        1.25 * (pl.col("tp") - pl.col("hit")) / (denominator - 1))
    winners = winners.with_columns(delta=1.25 * pl.col("tp") / denominator - previous)
    cuts = winners.group_by(score).agg(pl.col("delta").sum()).sort(score, descending=True)
    cuts = cuts.with_columns(value=(initial + pl.col("delta").cum_sum()) / len(a))
    best = cuts.row(int(cuts["value"].arg_max()), named=True)
    if best["value"] <= initial / len(a):
        return initial / len(a), NO_MATCH_THRESHOLD
    return float(best["value"]), float(best[score])


def blend_scores(pred: pl.DataFrame, stage3_weight: float) -> pl.DataFrame:
    """Stage-2/3 mixture. New two-hop candidates have no stage-2 evidence."""
    if not 0 <= stage3_weight <= 1:
        raise ValueError("stage3_weight must be between zero and one")
    return pred.with_columns(score=(stage3_weight * pl.col("p3")
                                    + (1 - stage3_weight) * pl.col("p2").fill_null(0.0)))


def tune_blend(pred: pl.DataFrame, truth: pl.DataFrame, anchors: pl.Series) -> dict:
    """Choose model mixture and cutoff on tuning only, including stage 2 alone."""
    trials = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        metric, threshold = tune_threshold(blend_scores(pred, weight), truth, anchors, "score")
        trials.append({"stage3_weight": weight, "threshold": threshold, "macro_f05": metric})
    # Keep stage 2 on an exact tie; stage 3 has to improve the tuning objective.
    best = trials[0]
    for trial in trials[1:]:
        if trial["macro_f05"] > best["macro_f05"] + 1e-12:
            best = trial
    return {"stage3_weight": best["stage3_weight"], "threshold": best["threshold"],
            "tuning_trials": trials}


def best_per_target(pred: pl.DataFrame, score: str) -> pl.DataFrame:
    """Break equal scores by the lowest Source 1 row index, reproducibly."""
    return (pred.filter(pl.col(score).is_finite())
            .sort(["tidx", score, "sidx"], descending=[False, True, False])
            .unique("tidx", keep="first", maintain_order=True))


def decide_country(pred: pl.DataFrame, threshold: float, anchors: pl.DataFrame,
                   thresholds: dict[str, float], score: str = "p2") -> pl.DataFrame:
    """Country cutoffs with a global fallback for unlabeled/unseen countries."""
    if not thresholds:
        return best_per_target(pred.filter(pl.col(score) >= threshold), score)
    limits = anchors.select("sidx", _cut=pl.col("country").replace_strict(
        thresholds, default=threshold, return_dtype=pl.Float64))
    eligible = pred.join(limits, on="sidx", how="left").filter(
        pl.col(score) >= pl.col("_cut").fill_null(threshold)).drop("_cut")
    return best_per_target(eligible, score)


def tune_country_thresholds(pred: pl.DataFrame, truth: pl.DataFrame, anchors: pl.DataFrame,
                            global_threshold: float, score: str = "p2",
                            minimum_anchors: int = 1000) -> dict[str, float]:
    """Only pass tuning-fold anchors/truth, excluding any held-out countries.

    Require enough businesses and a >0.0001 tuning gain over the global cutoff.
    Countries absent from tuning labels always retain the global fallback.
    """
    from .metrics import macro_f05
    result = {}
    for (country,), group in anchors.group_by("country"):
        if len(group) < minimum_anchors:
            continue
        pairs = pred.join(group.select("sidx"), on="sidx", how="semi")
        value, cutoff = tune_threshold(pairs, truth, group["sidx"], score)
        baseline = macro_f05(best_per_target(pairs.filter(pl.col(score) >= global_threshold), score),
                             truth, group["sidx"])["macro_f05"]
        if value > baseline + 0.0001:
            result[country] = cutoff
    return result


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
