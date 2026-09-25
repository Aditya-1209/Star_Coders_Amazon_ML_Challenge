"""Stage 4: two-stage gradient-boosted (XGBoost, CUDA) matcher, trained and evaluated on train folds.

Stage 1 scores each pair from its own features. Stage 2 adds context from the
stage-1 scores: how the pair ranks among its Source 1 record's candidates and
among all Source 1 records competing for the same Source 2/3 target (every
target belongs to at most one business). Folds are assigned by Source 1 id.
  folds 0/8 and 1/9 : complementary stage-1 models; average on unseen folds
  folds 2/5 : stage-2 training      fold 3 : early stopping and threshold tuning
  fold  4   : untouched holdout for the reported score
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
from .metrics import macro_f05
from .decision import best_per_target, tune_threshold
from .runtime import feature_parts, positive_int
from .matrix import quantile_matrix

N_FOLDS = 10
STAGE1_GROUPS = ((0, 8), (1, 9))
FEATURE_VERSION = "r4-accuracy-1"
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


def predict_frame(model: xgb.Booster, df: pl.DataFrame, feats: list[str], batch_rows: int = 250_000) -> np.ndarray:
    """Bound dense feature matrices instead of converting an entire shard at once."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    output = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), batch_rows):
        batch = df.slice(start, batch_rows)
        output[start:start + len(batch)] = predict(model, X(batch, feats))
    return output


def predict_ensemble(models: list[xgb.Booster], df: pl.DataFrame, feats: list[str],
                     batch_rows: int = 250_000) -> np.ndarray:
    """Average both complementary rankers for rows unseen by either model."""
    if not models:
        raise ValueError("At least one stage-1 model is required")
    values = predict_frame(models[0], df, feats, batch_rows)
    for model in models[1:]:
        values += predict_frame(model, df, feats, batch_rows)
    return values / len(models)


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


def X(df: pl.DataFrame, feats: list[str]):
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def fit(df: pl.DataFrame, feats: list[str], valid: pl.DataFrame | None, rounds: int,
        params: dict | None = None, matrix_batch_rows: int = 100_000) -> xgb.Booster:
    if df.is_empty() or df["label"].n_unique() < 2:
        raise ValueError("Training fold must contain positive and negative candidate pairs")
    params = dict(PARAMS if params is None else params)
    dtrain = quantile_matrix(df, feats, matrix_batch_rows, params)
    evals = [(dtrain, "train")]
    kw = {}
    if valid is not None:
        if valid.is_empty():
            raise ValueError("Validation fold has no candidate pairs")
        evals.append((quantile_matrix(valid, feats, matrix_batch_rows, params, reference=dtrain), "valid"))
        kw["early_stopping_rounds"] = 50
    return xgb.train(params, dtrain, rounds, evals=evals, verbose_eval=100, **kw)


def stage1_scores(folder: Path, models: dict[int, xgb.Booster], default: list[xgb.Booster], feats,
                  batch_rows: int = 250_000) -> pl.DataFrame:
    """Use complementary models on their training groups, ensemble elsewhere.

    Fold 3 influences early stopping, so only fold 4 is an untouched holdout.
    """
    out = []
    for p in feature_parts(folder):
        df = pl.read_parquet(p).with_columns(fold_expr())
        pred = np.empty(len(df), dtype=np.float32)
        unseen = ~df["fold"].is_in(list(models))
        if unseen.any():
            pred[unseen.to_numpy()] = predict_ensemble(default, df.filter(unseen), feats, batch_rows)
        grouped = {}
        for fold, model in models.items():
            grouped.setdefault(id(model), (model, []))[1].append(fold)
        for model, folds in grouped.values():
            mask = df["fold"].is_in(folds).to_numpy()
            if mask.any():
                pred[mask] = predict_frame(model, df.filter(pl.col("fold").is_in(folds)), feats, batch_rows)
        out.append(df.select("sidx", "tidx").with_columns(p1=pl.Series(pred.astype(np.float32))))
    return pl.concat(out)


def context_features(scores: pl.DataFrame) -> pl.DataFrame:
    """Global competition features, preserving input rows for bounded scoring.

    Order each ranking's ties inside its group, instead of globally reordering
    every candidate. The final join explicitly preserves the input order.
    """
    s = scores.with_columns(
        s_rank=pl.col("p1").rank("ordinal", descending=True).over("sidx", order_by="tidx").cast(pl.UInt16),
        s_max=pl.col("p1").max().over("sidx"),
        s_sum=pl.col("p1").sum().over("sidx"),
        s_n50=(pl.col("p1") > 0.5).sum().over("sidx").cast(pl.UInt16),
        t_prank=pl.col("p1").rank("ordinal", descending=True).over("tidx", order_by="sidx").cast(pl.UInt16),
        t_max=pl.col("p1").max().over("tidx"),
        t_sum=pl.col("p1").sum().over("tidx"),
    )
    top2 = scores.group_by("tidx").agg(t_second=pl.col("p1").top_k(2).min(), t_cnt=pl.len())
    s = s.join(top2, on="tidx", how="left", maintain_order="left")
    return s.with_columns(
        t_other=pl.when(pl.col("t_prank") == 1).then(
            pl.when(pl.col("t_cnt") > 1).then(pl.col("t_second")).otherwise(0.0))
        .otherwise(pl.col("t_max")),
        s_rel=pl.when(pl.col("s_max") > 0).then(pl.col("p1") / pl.col("s_max")).otherwise(0.0),
    ).drop("t_second", "t_cnt")


def attach_context(features: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Attach already aligned columns; fail instead of silently mis-scoring IDs."""
    if not features.select("sidx", "tidx").equals(context.select("sidx", "tidx")):
        raise ValueError("Feature/context row order differs; regenerate scores from the same feature shards")
    return features.hstack(context.drop("sidx", "tidx"))


class ContextBatches:
    """Retain only model-pruned rows, and select each shard by original offset."""

    def __init__(self, context: pl.DataFrame, prune: float):
        self.total_rows = len(context)
        self.position = 0
        self.frame = context.with_row_index("_row").filter(pl.col("p1") >= prune)
        self.row_positions = self.frame["_row"].to_numpy()

    def attach(self, features: pl.DataFrame) -> pl.DataFrame:
        start, stop = self.position, self.position + len(features)
        if stop > self.total_rows:
            raise ValueError("Feature shards contain more rows than the scored context")
        lo, hi = np.searchsorted(self.row_positions, [start, stop])
        part = self.frame.slice(int(lo), int(hi - lo))
        keep = np.zeros(len(features), dtype=bool)
        keep[self.row_positions[lo:hi] - start] = True
        result = attach_context(features.filter(keep), part.drop("_row"))
        self.position = stop
        return result

    def finish(self) -> None:
        if self.position != self.total_rows:
            raise ValueError("Feature shards ended before all scored context rows were consumed")


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
    ap.add_argument("--threads", type=positive_int, default=12)
    ap.add_argument("--batch-rows", type=positive_int, default=250_000)
    ap.add_argument("--matrix-batch-rows", type=positive_int, default=100_000)
    ap.add_argument("--hist-cache-nodes", type=positive_int, default=4096)
    ap.add_argument("--predict-device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--rounds", type=positive_int, default=1500)
    ap.add_argument("--extra-stage1-folds", action=argparse.BooleanOptionalAction, default=True,
                    help="use folds 8/9 as additional training data; disable to reduce training RAM")
    args = ap.parse_args()
    work, mdir = Path(args.work), Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    folder = work / "feats_train"
    params = {**PARAMS, "device": args.device, "nthread": args.threads,
              "max_cached_hist_node": args.hist_cache_nodes}
    predict_device = args.predict_device or args.device
    t = time.time()

    sample = pl.read_parquet(feature_parts(folder)[0], n_rows=1)
    f1 = feature_names(sample)
    if "postcode_conflict" not in f1 or "raw_name_ratio" not in f1:
        raise ValueError("Accuracy features are missing; regenerate features in a fresh workspace")
    # Fold 3 may tune early stopping, but fold 4 never influences fitting or selection.
    valid1 = validation_sample(load_feats(folder, [3]), seed=1)
    rankers = []
    groups = STAGE1_GROUPS if args.extra_stage1_folds else ((0,), (1,))
    for group in groups:
        base = load_feats(folder, list(group))
        print(f"stage1 folds {group}: {len(base):,} pairs", flush=True)
        rankers.append(fit(base, f1, valid1, args.rounds, params, args.matrix_batch_rows))
        del base
    del valid1
    m0, m1 = rankers
    m0.save_model(str(mdir / "stage1.json"))
    m1.save_model(str(mdir / "stage1_b.json"))
    for model in rankers:
        model.set_param({"device": predict_device})
    print(f"stage1 ensemble trained, {time.time() - t:.0f}s", flush=True)

    held_out = {fold: m1 for fold in groups[0]}
    held_out.update({fold: m0 for fold in groups[1]})
    scores = stage1_scores(folder, held_out, rankers, f1, args.batch_rows)
    ctx = context_features(scores)
    del scores
    print(f"stage1 scored + context, {time.time() - t:.0f}s", flush=True)

    ctx_cols = [c for c in ctx.columns if c not in ("sidx", "tidx")]
    f2 = f1 + ctx_cols
    ctx_full = ContextBatches(ctx, PRUNE)
    ctx = ctx.filter(fold_expr().is_in([2, 3, 4, 5]))
    tr = attach_context(load_feats(folder, [2, 5]), ctx.filter(fold_expr().is_in([2, 5])))
    va = validation_sample(attach_context(load_feats(folder, [3]), ctx.filter(fold_expr() == 3)), seed=2)
    del ctx
    m2 = fit(tr, f2, va, args.rounds, params, args.matrix_batch_rows)
    del tr, va
    m2.save_model(str(mdir / "stage2.json"))
    m2.set_param({"device": predict_device})
    print(f"stage2 trained ({m2.best_iteration} it), {time.time() - t:.0f}s", flush=True)

    # stage-2 scores for every pruned pair of every fold (stage 3 builds on them)
    allp = []
    for p in feature_parts(folder):
        part = pl.read_parquet(p).with_columns(fold_expr())
        part = ctx_full.attach(part)
        allp.append(part.select("sidx", "tidx", "fold", "p1", "label").with_columns(
            p2=pl.Series(predict_frame(m2, part, f2, args.batch_rows))))
    ctx_full.finish()
    allp = pl.concat(allp)
    allp.write_parquet(work / "stage2_train.parquet")
    del ctx_full, part, rankers, held_out
    ev = allp.filter(pl.col("fold").is_in([3, 4]))
    del allp
    ev.write_parquet(work / "eval_preds.parquet")

    # truth + anchors for folds 3/4: all source1 records, incl. singletons / uncovered
    from .run_block import load_split
    from .run_features import ground_truth_pairs
    s1, tg = load_split(work, "train", columns=["idx", "entity_id"])
    truth = ground_truth_pairs(Path(args.dataset), s1, tg).with_columns(fold_expr())
    anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32)).with_columns(fold_expr())
    del s1, tg

    results = {}
    retrieval = (pl.scan_parquet(work / "cands_train.parquet").select("sidx", "tidx")
                 .with_columns(fold_expr()).filter(pl.col("fold").is_in([3, 4])).collect(engine="streaming"))
    for fold in (3, 4):
        retrieved = retrieval.filter(pl.col("fold") == fold)
        tr_ = truth.filter(pl.col("fold") == fold)
        a = anchors.filter(pl.col("fold") == fold)["sidx"]
        results[f"fold{fold}_retrieval_oracle"] = macro_f05(retrieved.join(tr_, on=["sidx", "tidx"]), tr_, a)
    del retrieval, retrieved
    _, thr = tune_threshold(ev.filter(pl.col("fold") == 3), truth.filter(pl.col("fold") == 3),
                            anchors.filter(pl.col("fold") == 3)["sidx"])
    for fold in (3, 4):
        e = ev.filter(pl.col("fold") == fold)
        a = anchors.filter(pl.col("fold") == fold)["sidx"]
        tr_ = truth.filter(pl.col("fold") == fold)
        results[f"fold{fold}_stage2_excl"] = macro_f05(decide(e, thr), tr_, a)
        results[f"fold{fold}_stage2_noexcl"] = macro_f05(e.filter(pl.col("p2") >= thr), tr_, a)
        results[f"fold{fold}_stage1_excl"] = macro_f05(decide(e, thr, "p1"), tr_, a)
        results[f"fold{fold}_oracle"] = macro_f05(e.join(tr_, on=["sidx", "tidx"]), tr_, a)
    results["threshold"] = thr
    results["prune"] = PRUNE
    results["runtime"] = {"device": args.device, "predict_device": predict_device,
                          "threads": args.threads, "batch_rows": args.batch_rows,
                          "matrix_batch_rows": args.matrix_batch_rows, "hist_cache_nodes": args.hist_cache_nodes}
    results["stage2_training_folds"] = [2, 5]
    results["stage3_eligible_folds"] = [3, 4, 6, 7]
    results["feature_version"] = FEATURE_VERSION
    results["stage1_models"] = ["stage1.json", "stage1_b.json"]
    results["stage1_training_groups"] = groups
    results["early_stopping_fold"] = 3
    results["threshold_method"] = "exact_exclusive_macro_f05"
    for fold in (3, 4):
        results[f"fold{fold}_mean_candidates"] = len(ev.filter(pl.col("fold") == fold)) / len(
            anchors.filter(pl.col("fold") == fold))
    results["stage1_features"] = f1
    results["stage2_features"] = f2
    results["best_iterations"] = {"stage1": m0.best_iteration, "stage1_b": m1.best_iteration,
                                  "stage2": m2.best_iteration}
    (mdir / "metrics.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if isinstance(v, dict) and "macro_f05" in v:
            print(f"{k:28s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
    print(f"threshold {thr:.3f}; total {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
