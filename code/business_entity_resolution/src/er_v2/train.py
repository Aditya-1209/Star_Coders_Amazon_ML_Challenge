"""Stage 4: two-stage LightGBM matcher, trained and evaluated on train folds.

Stage 1 scores each pair from its own features. Stage 2 adds context from the
stage-1 scores: how the pair ranks among its Source 1 record's candidates and
among all Source 1 records competing for the same Source 2/3 target (every
target belongs to at most one business). Folds are assigned by Source 1 id.
  folds 0/1 : stage-1 models (each predicts the other fold + everything else)
  fold  2   : stage-2 training      fold 3 : threshold tuning
  fold  4   : untouched holdout for the reported score
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .features import feature_names
from .metrics import macro_f05

N_FOLDS = 10
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=0, verbose=-1, seed=42)


def fold_expr() -> pl.Expr:
    return (pl.col("sidx").hash(seed=11) % N_FOLDS).cast(pl.Int8).alias("fold")


def load_feats(folder: Path, folds: list[int] | None = None, columns=None) -> pl.DataFrame:
    parts = []
    for p in sorted(folder.glob("part_*.parquet")):
        df = pl.read_parquet(p, columns=columns)
        if folds is not None:
            df = df.filter(fold_expr().is_in(folds))
        parts.append(df)
    return pl.concat(parts)


def X(df: pl.DataFrame, feats: list[str]):
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def fit(df: pl.DataFrame, feats: list[str], valid: pl.DataFrame | None, rounds: int) -> lgb.Booster:
    dtrain = lgb.Dataset(X(df, feats), df["label"].to_numpy(), feature_name=feats,
                         free_raw_data=True)
    sets, cbs = [dtrain], [lgb.log_evaluation(100)]
    if valid is not None:
        sets.append(lgb.Dataset(X(valid, feats), valid["label"].to_numpy(), reference=dtrain))
        cbs.append(lgb.early_stopping(50, verbose=False))
    return lgb.train(PARAMS, dtrain, rounds, valid_sets=sets, callbacks=cbs)


def stage1_scores(folder: Path, models: dict[int, lgb.Booster], default: lgb.Booster, feats) -> pl.DataFrame:
    """Out-of-fold stage-1 probability for every pair in the folder."""
    out = []
    for p in sorted(folder.glob("part_*.parquet")):
        df = pl.read_parquet(p).with_columns(fold_expr())
        x = X(df, feats)
        pred = default.predict(x, num_threads=0)
        for fold, model in models.items():
            mask = (df["fold"] == fold).to_numpy()
            if mask.any():
                pred[mask] = model.predict(x[mask], num_threads=0)
        out.append(df.select("sidx", "tidx").with_columns(p1=pl.Series(pred.astype(np.float32))))
    return pl.concat(out)


def context_features(scores: pl.DataFrame) -> pl.DataFrame:
    """Stage-2 context from stage-1 scores; target competition is global."""
    s = scores.with_columns(
        s_rank=pl.col("p1").rank("ordinal", descending=True).over("sidx").cast(pl.UInt16),
        s_max=pl.col("p1").max().over("sidx"),
        s_sum=pl.col("p1").sum().over("sidx"),
        s_n50=(pl.col("p1") > 0.5).sum().over("sidx").cast(pl.UInt16),
        t_prank=pl.col("p1").rank("ordinal", descending=True).over("tidx").cast(pl.UInt16),
        t_max=pl.col("p1").max().over("tidx"),
        t_sum=pl.col("p1").sum().over("tidx"),
    )
    top2 = scores.group_by("tidx").agg(t_second=pl.col("p1").top_k(2).min(), t_cnt=pl.len())
    s = s.join(top2, on="tidx", how="left")
    return s.with_columns(
        t_other=pl.when(pl.col("t_prank") == 1).then(
            pl.when(pl.col("t_cnt") > 1).then(pl.col("t_second")).otherwise(0.0))
        .otherwise(pl.col("t_max")),
        s_rel=pl.col("p1") / pl.col("s_max"),
    ).drop("t_second", "t_cnt")


def decide(pred: pl.DataFrame, threshold: float, score: str = "p2") -> pl.DataFrame:
    """Threshold, then give each target only to its best-scoring Source 1."""
    kept = pred.filter(pl.col(score) >= threshold)
    return kept.filter(pl.col(score) == pl.col(score).max().over("tidx")).unique("tidx", keep="first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--model-dir", default="models/v2")
    ap.add_argument("--rounds", type=int, default=1500)
    args = ap.parse_args()
    work, mdir = Path(args.work), Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    folder = work / "feats_train"
    t = time.time()

    sample = pl.read_parquet(next(folder.glob("part_*.parquet")), n_rows=1)
    f1 = feature_names(sample)
    base = load_feats(folder, [0, 1]).with_columns(fold_expr())
    print(f"stage1 data {len(base):,} pairs, {time.time() - t:.0f}s", flush=True)
    m0 = fit(base.filter(pl.col("fold") == 0), f1,
             base.filter(pl.col("fold") == 1).sample(2_000_000, seed=1), args.rounds)
    m1 = fit(base.filter(pl.col("fold") == 1), f1,
             base.filter(pl.col("fold") == 0).sample(2_000_000, seed=1), m0.best_iteration or args.rounds)
    del base
    m0.save_model(str(mdir / "stage1.txt"))
    print(f"stage1 trained ({m0.best_iteration} it), {time.time() - t:.0f}s", flush=True)

    scores = stage1_scores(folder, {0: m1, 1: m0}, m0, f1)
    ctx = context_features(scores)
    del scores
    print(f"stage1 scored + context, {time.time() - t:.0f}s", flush=True)

    ctx_cols = [c for c in ctx.columns if c not in ("sidx", "tidx")]
    f2 = f1 + ctx_cols
    ctx = ctx.filter(fold_expr().is_in([2, 3, 4]))
    tr = load_feats(folder, [2]).join(ctx, on=["sidx", "tidx"], how="left")
    va = load_feats(folder, [3]).sample(2_000_000, seed=2).join(ctx, on=["sidx", "tidx"], how="left")
    m2 = fit(tr, f2, va, args.rounds)
    del tr, va
    m2.save_model(str(mdir / "stage2.txt"))
    print(f"stage2 trained ({m2.best_iteration} it), {time.time() - t:.0f}s", flush=True)

    ev = []
    for p in sorted(folder.glob("part_*.parquet")):
        held = pl.read_parquet(p).with_columns(fold_expr()).filter(pl.col("fold") >= 3)
        held = held.join(ctx, on=["sidx", "tidx"], how="left")
        ev.append(held.select("sidx", "tidx", "fold", "p1", "label").with_columns(
            p2=pl.Series(m2.predict(X(held, f2), num_threads=0).astype(np.float32))))
    ev = pl.concat(ev)
    del ctx
    ev.write_parquet(work / "eval_preds.parquet")

    # truth + anchors for folds 3/4: all source1 records, incl. singletons / uncovered
    from .run_block import load_split
    from .run_features import ground_truth_pairs
    s1, tg = load_split(work, "train")
    truth = ground_truth_pairs(Path("student_resource/dataset"), s1, tg).with_columns(fold_expr())
    anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32)).with_columns(fold_expr())

    results = {}
    best = (0, 0.5)
    for thr in np.arange(0.2, 0.95, 0.025):
        r = macro_f05(decide(ev.filter(pl.col("fold") == 3), thr), truth.filter(pl.col("fold") == 3),
                      anchors.filter(pl.col("fold") == 3)["sidx"])
        if r["macro_f05"] > best[0]:
            best = (r["macro_f05"], float(thr))
    thr = best[1]
    for fold in (3, 4):
        e = ev.filter(pl.col("fold") == fold)
        a = anchors.filter(pl.col("fold") == fold)["sidx"]
        tr_ = truth.filter(pl.col("fold") == fold)
        results[f"fold{fold}_stage2_excl"] = macro_f05(decide(e, thr), tr_, a)
        results[f"fold{fold}_stage2_noexcl"] = macro_f05(e.filter(pl.col("p2") >= thr), tr_, a)
        results[f"fold{fold}_stage1_excl"] = macro_f05(decide(e, thr, "p1"), tr_, a)
        results[f"fold{fold}_oracle"] = macro_f05(e.join(tr_, on=["sidx", "tidx"]), tr_, a)
    results["threshold"] = thr
    results["stage1_features"] = f1
    results["stage2_features"] = f2
    results["best_iterations"] = {"stage1": m0.best_iteration, "stage2": m2.best_iteration}
    (mdir / "metrics.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if isinstance(v, dict) and "macro_f05" in v:
            print(f"{k:28s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
    print(f"threshold {thr:.3f}; total {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
