"""Stage 4: two-stage gradient-boosted (XGBoost, CUDA) matcher, trained and evaluated on train folds.

Stage 1 scores each pair from its own features. Stage 2 adds context from the
stage-1 scores: how the pair ranks among its Source 1 record's candidates and
among all Source 1 records competing for the same Source 2/3 target (every
target belongs to at most one business). Folds are assigned by Source 1 id.
  folds 0+8 / 1+9 : two stage-1 models (each predicts the other pair + everything else);
                    negatives subsampled to NEG_FRAC with weight 1/NEG_FRAC
  folds 2/5 : stage-2 training      fold 3 : threshold tuning
  fold  4   : untouched holdout for the reported score
--exclude-country C drops country C from every training set (not from evaluation):
the "unseen country" score for C is the proxy for France, which has no labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import xgboost as xgb
import numpy as np
import polars as pl

from .features import feature_names
from .metrics import by_country, leaderboard_estimate, macro_f05
from .decision import best_per_target
from .runtime import BATCH_ROWS, DEFAULT_THREADS, feature_parts, positive_int

N_FOLDS = 10
STAGE1_FOLDS = ([0, 8], [1, 9])
NEG_FRAC = 0.5
PRUNE = 0.001  # stage-1 floor that defines the final candidate set (see predict.py)
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              device="cuda", eta=0.08, max_depth=10, min_child_weight=5, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=42)


def predict(model: xgb.Booster, x) -> np.ndarray:
    """Predict with the best iteration; runs on the GPU when available."""
    if x.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    it = getattr(model, "best_iteration", None)
    rng = (0, it + 1) if it is not None else (0, 0)
    return model.inplace_predict(x, iteration_range=rng)


def predict_frame(model: xgb.Booster, df: pl.DataFrame, feats: list[str], batch_rows: int = BATCH_ROWS) -> np.ndarray:
    """Bound dense feature matrices instead of converting an entire shard at once."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    output = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), batch_rows):
        batch = df.slice(start, batch_rows)
        output[start:start + len(batch)] = predict(model, X(batch, feats))
    return output


def validation_sample(df: pl.DataFrame, seed: int, limit: int = 2_000_000) -> pl.DataFrame:
    if df.is_empty():
        raise ValueError("Validation fold has no candidate pairs; use a larger training sample")
    return df.sample(min(limit, len(df)), seed=seed)


def fold_expr() -> pl.Expr:
    return (pl.col("sidx").hash(seed=11) % N_FOLDS).cast(pl.Int8).alias("fold")


def load_feats(folder: Path, folds: list[int] | None = None, columns=None) -> pl.DataFrame:
    parts = []
    for p in feature_parts(folder):
        df = pl.read_parquet(p, columns=columns)
        if folds is not None:
            df = df.filter(fold_expr().is_in(folds))
        parts.append(df)
    return pl.concat(parts)


def load_train(folder: Path, folds: list[int], neg_frac: float = 1.0, exclude: pl.Series | None = None,
               seed: int = 7) -> pl.DataFrame:
    """Training rows of the given folds; optional negative subsampling (weight column ``w``)."""
    parts = []
    for p in feature_parts(folder):
        df = pl.read_parquet(p).filter(fold_expr().is_in(folds))
        if exclude is not None:
            df = df.filter(~pl.col("sidx").is_in(exclude))
        if neg_frac < 1.0:
            keep = (pl.col("label") == 1) | (
                (pl.struct("sidx", "tidx").hash(seed=seed) % 1_000_000) < int(neg_frac * 1_000_000))
            df = df.filter(keep).with_columns(
                w=pl.when(pl.col("label") == 1).then(1.0).otherwise(1.0 / neg_frac).cast(pl.Float32))
        parts.append(df)
    return pl.concat(parts)


def excluded_sidx(work: Path, countries: list[str]) -> pl.Series | None:
    if not countries:
        return None
    s1 = pl.read_parquet(work / "norm" / "train_source1.parquet", columns=["idx", "country"])
    return s1.filter(pl.col("country").is_in(countries))["idx"].cast(pl.UInt32)


def X(df: pl.DataFrame, feats: list[str]):
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def fit(df: pl.DataFrame, feats: list[str], valid: pl.DataFrame | None, rounds: int,
        params: dict | None = None) -> xgb.Booster:
    if df.is_empty() or df["label"].n_unique() < 2:
        raise ValueError("Training fold must contain positive and negative candidate pairs")
    weight = df["w"].to_numpy() if "w" in df.columns else None
    dtrain = xgb.QuantileDMatrix(X(df, feats), df["label"].to_numpy(), weight=weight, feature_names=feats)
    evals = [(dtrain, "train")]
    kw = {}
    if valid is not None:
        if valid.is_empty():
            raise ValueError("Validation fold has no candidate pairs")
        evals.append((xgb.QuantileDMatrix(X(valid, feats), valid["label"].to_numpy(),
                                          feature_names=feats, ref=dtrain), "valid"))
        kw["early_stopping_rounds"] = 50
    return xgb.train(PARAMS if params is None else params, dtrain, rounds, evals=evals, verbose_eval=100, **kw)


def stage1_scores(folder: Path, models: dict[int, xgb.Booster], default: xgb.Booster, feats,
                  batch_rows: int = BATCH_ROWS) -> pl.DataFrame:
    """Out-of-fold stage-1 probability for every pair in the folder."""
    out = []
    for p in feature_parts(folder):
        df = pl.read_parquet(p).with_columns(fold_expr())
        pred = predict_frame(default, df, feats, batch_rows)
        for fold, model in models.items():
            mask = (df["fold"] == fold).to_numpy()
            if mask.any():
                pred[mask] = predict_frame(model, df.filter(pl.col("fold") == fold), feats, batch_rows)
        out.append(df.select("sidx", "tidx").with_columns(p1=pl.Series(pred.astype(np.float32))))
    return pl.concat(out)


def context_features(scores: pl.DataFrame) -> pl.DataFrame:
    """Stage-2 context from stage-1 scores; target competition is global."""
    scores = scores.sort(["p1", "sidx", "tidx"], descending=[True, False, False])
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
        s_rel=pl.when(pl.col("s_max") > 0).then(pl.col("p1") / pl.col("s_max")).otherwise(0.0),
    ).drop("t_second", "t_cnt")


def decide(pred: pl.DataFrame, threshold: float, score: str = "p2") -> pl.DataFrame:
    """Threshold, then give each target only to its best-scoring Source 1."""
    kept = pred.filter(pl.col(score) >= threshold)
    return best_per_target(kept, score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--model-dir", default="models/v2")
    ap.add_argument("--dataset", default="student_resource/dataset")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=positive_int, default=DEFAULT_THREADS)
    ap.add_argument("--batch-rows", type=positive_int, default=BATCH_ROWS)
    ap.add_argument("--rounds", type=positive_int, default=1500)
    ap.add_argument("--exclude-country", nargs="*", default=[],
                    help="leave these countries out of all training (unseen-country proxy for France)")
    args = ap.parse_args()
    tag = "".join(f"_no{c}" for c in args.exclude_country)
    work, mdir = Path(args.work), Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    folder = work / "feats_train"
    params = {**PARAMS, "device": args.device, "nthread": args.threads}
    t = time.time()

    sample = pl.read_parquet(feature_parts(folder)[0], n_rows=1)
    f1 = feature_names(sample)
    excl = excluded_sidx(work, args.exclude_country)
    fa, fb = STAGE1_FOLDS
    a = load_train(folder, fa, NEG_FRAC, excl)
    b = load_train(folder, fb, NEG_FRAC, excl)
    print(f"stage1 data {len(a):,} + {len(b):,} pairs (negatives x{NEG_FRAC}), {time.time() - t:.0f}s", flush=True)
    m0 = fit(a, f1, validation_sample(b, seed=1), args.rounds, params)
    first_rounds = m0.best_iteration + 1 if m0.best_iteration is not None else args.rounds
    m1 = fit(b, f1, validation_sample(a, seed=1), first_rounds, params)
    del a, b
    m0.save_model(str(mdir / "stage1.json"))
    print(f"stage1 trained ({m0.best_iteration} it), {time.time() - t:.0f}s", flush=True)

    oof = {**{f: m1 for f in fa}, **{f: m0 for f in fb}}
    scores = stage1_scores(folder, oof, m0, f1, args.batch_rows)
    ctx = context_features(scores)
    del scores
    print(f"stage1 scored + context, {time.time() - t:.0f}s", flush=True)

    ctx_cols = [c for c in ctx.columns if c not in ("sidx", "tidx")]
    f2 = f1 + ctx_cols
    ctx_full = ctx.filter(pl.col("p1") >= PRUNE)
    ctx = ctx.filter(fold_expr().is_in([2, 3, 4, 5]))
    tr = load_train(folder, [2, 5], exclude=excl).join(ctx, on=["sidx", "tidx"], how="left")
    va = validation_sample(load_feats(folder, [3]), seed=2).join(ctx, on=["sidx", "tidx"], how="left")
    m2 = fit(tr, f2, va, args.rounds, params)
    del tr, va
    m2.save_model(str(mdir / "stage2.json"))
    print(f"stage2 trained ({m2.best_iteration} it), {time.time() - t:.0f}s", flush=True)

    # stage-2 scores for every pruned pair of every fold (stage 3 builds on them)
    allp = []
    for p in feature_parts(folder):
        part = pl.read_parquet(p).with_columns(fold_expr())
        part = part.join(ctx_full, on=["sidx", "tidx"], how="inner").filter(pl.col("p1") >= PRUNE)
        allp.append(part.select("sidx", "tidx", "fold", "p1", "label").with_columns(
            p2=pl.Series(predict_frame(m2, part, f2, args.batch_rows))))
    allp = pl.concat(allp)
    allp.write_parquet(work / f"stage2_train{tag}.parquet")
    del ctx_full
    ev = allp.filter(pl.col("fold").is_in([3, 4]))
    del ctx
    ev.write_parquet(work / f"eval_preds{tag}.parquet")

    # truth + anchors for folds 3/4: all source1 records, incl. singletons / uncovered
    from .run_block import load_split
    from .run_features import ground_truth_pairs
    s1, tg = load_split(work, "train")
    truth = ground_truth_pairs(Path(args.dataset), s1, tg).with_columns(fold_expr())
    anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country").with_columns(fold_expr())

    results = {}
    best = (0, 0.5)
    # tune only on countries seen in training, so an excluded country stays truly unseen
    tune_anchors = anchors.filter((pl.col("fold") == 3) & ~pl.col("country").is_in(args.exclude_country))["sidx"]
    for thr in np.arange(0.2, 0.95, 0.025):
        r = macro_f05(decide(ev.filter(pl.col("fold") == 3), thr), truth.filter(pl.col("fold") == 3),
                      tune_anchors)
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
    hold = ev.filter(pl.col("fold") == 4)
    per = by_country(decide(hold, thr), truth.filter(pl.col("fold") == 4), anchors.filter(pl.col("fold") == 4))
    results["fold4_by_country"] = per
    results["leaderboard_estimate"] = leaderboard_estimate(per)
    results["excluded_countries"] = args.exclude_country
    results["threshold"] = thr
    results["prune"] = PRUNE
    results["runtime"] = {"device": args.device, "threads": args.threads, "batch_rows": args.batch_rows}
    results["stage1_training_folds"] = list(STAGE1_FOLDS)
    results["stage1_neg_frac"] = NEG_FRAC
    results["stage2_training_folds"] = [2, 5]
    results["stage3_eligible_folds"] = [3, 4, 6, 7]
    for fold in (3, 4):
        results[f"fold{fold}_mean_candidates"] = len(ev.filter(pl.col("fold") == fold)) / len(
            anchors.filter(pl.col("fold") == fold))
    results["stage1_features"] = f1
    results["stage2_features"] = f2
    results["best_iterations"] = {"stage1": m0.best_iteration, "stage2": m2.best_iteration}
    (mdir / "metrics.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if isinstance(v, dict) and "macro_f05" in v:
            print(f"{k:28s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
    for c, v in per.items():
        print(f"fold4 {c:8s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
    print(f"leaderboard estimate {results['leaderboard_estimate']['estimate']:.4f} "
          f"(France assumed {results['leaderboard_estimate']['france_assumed']})")
    print(f"threshold {thr:.3f}; total {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
