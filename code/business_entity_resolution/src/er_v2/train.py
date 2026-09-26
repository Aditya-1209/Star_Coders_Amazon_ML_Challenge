"""Stage 4: two-stage gradient-boosted (XGBoost, CUDA) matcher, trained and evaluated on train folds.

Stage 1 scores each pair from its own features. Stage 2 adds context from the
stage-1 scores: how the pair ranks among its Source 1 record's candidates and
among all Source 1 records competing for the same Source 2/3 target (every
target belongs to at most one business). Folds are assigned by Source 1 id.
  folds 0+8 / 1+9 : complementary stage-1 models (negatives subsampled to NEG_FRAC,
                    weight 1/NEG_FRAC); each scores the other group, both average elsewhere
  Stage 2 trains, builds context and predicts on stage-1 survivors only
  (p1 >= PRUNE): exactly the candidate set it scores at inference.
  folds 2/5 : stage-2 training      fold 3 : early stopping and threshold tuning
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
from .decision import best_per_target, tune_threshold, tune_country_thresholds, decide_country
from .runtime import BATCH_ROWS, DEFAULT_THREADS, feature_parts, positive_int

from .folds import N_FOLDS, fold_expr
STAGE1_GROUPS = ((0, 8), (1, 9))
NEG_FRAC = 0.5
FEATURE_VERSION = "r7-pairlocal-1"
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


def predict_ensemble(models: list[xgb.Booster], df: pl.DataFrame, feats: list[str],
                     batch_rows: int = BATCH_ROWS) -> np.ndarray:
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


# ---- test-density simulation ("ghost" businesses) ------------------------------
# Test has ~5.75 Source 2/3 records per business vs 4.68 in train, i.e. about twice
# as many ownerless records. Removing a fixed ~19% of training businesses turns
# their records into ownerless ones, reproducing the test density. Their rows are
# dropped, and the blocking-competition features (b_rel_t, t_rank, t_nc) are
# recomputed without them. Test inference is unchanged.
GHOST: dict = {"ids": None, "ctx": None}
TEST_DENSITY_GHOST_FRAC = 0.19


def ghost_setup(work: Path, frac: float, seed: int = 23) -> None:
    if frac <= 0:
        GHOST.update(ids=None, ctx=None)
        return
    s1 = pl.read_parquet(work / "norm" / "train_source1.parquet", columns=["idx"])
    ids = s1.select(sidx=pl.col("idx").cast(pl.UInt32)).filter(
        (pl.col("sidx").hash(seed=seed) % 10_000) < int(frac * 10_000))["sidx"]
    ctx = (pl.scan_parquet(work / "cands_train.parquet").select("sidx", "tidx", "bscore")
           .filter(~pl.col("sidx").is_in(ids))
           .with_columns(
               b_rel_t=(pl.col("bscore") / pl.col("bscore").max().over("tidx")).cast(pl.Float32),
               t_rank=pl.col("bscore").rank("ordinal", descending=True).over("tidx").cast(pl.UInt16),
               t_nc=pl.len().over("tidx").cast(pl.UInt32))
           .select("sidx", "tidx", "b_rel_t", "t_rank", "t_nc").collect())
    GHOST.update(ids=ids, ctx=ctx)
    print(f"ghost regime: {len(ids):,} businesses removed ({frac:.0%})", flush=True)


def ghostify(df: pl.DataFrame) -> pl.DataFrame:
    """Drop ghost businesses and swap in competition features recomputed without them."""
    if GHOST["ids"] is None or "sidx" not in df.columns:
        return df
    df = df.filter(~pl.col("sidx").is_in(GHOST["ids"]))
    if "t_nc" not in df.columns:
        return df
    cols = df.columns
    df = df.drop("b_rel_t", "t_rank", "t_nc").join(GHOST["ctx"], on=["sidx", "tidx"], how="left",
                                                   maintain_order="left")
    return df.select(cols)


def load_feats(folder: Path, folds: list[int] | None = None, columns=None) -> pl.DataFrame:
    parts = []
    for p in feature_parts(folder):
        df = ghostify(pl.read_parquet(p, columns=columns))
        if folds is not None:
            df = df.filter(fold_expr().is_in(folds))
        parts.append(df)
    return pl.concat(parts)


def load_train(folder: Path, folds: list[int], neg_frac: float = 1.0, exclude: pl.Series | None = None,
               seed: int = 7, context=None, features=None) -> pl.DataFrame:
    """Training rows of the given folds; optional negative subsampling (weight column ``w``)."""
    parts = []
    for p in feature_parts(folder):
        df = ghostify(pl.read_parquet(p).filter(fold_expr().is_in(folds)))
        if exclude is not None:
            df = df.filter(~pl.col("sidx").is_in(exclude))
        cols = features if features is not None else feature_names(df)
        df = df.select("sidx", "tidx", "label", *cols)
        if context is not None:
            df = context.attach(df)
        if neg_frac < 1.0:
            keep = (pl.col("label") == 1) | (
                (pl.struct("sidx", "tidx").hash(seed=seed) % 1_000_000) < int(neg_frac * 1_000_000))
            df = df.filter(keep).with_columns(
                w=pl.when(pl.col("label") == 1).then(1.0).otherwise(1.0 / neg_frac).cast(pl.Float32))
        parts.append(df)
    return pl.concat(parts)


# ---- look-alike competition features (r9; off unless --lookalike) ---------------
# France has dozens of same-name businesses per town that differ only by street
# address. These features describe a candidate relative to the other stage-1
# survivors of the same Source 1 business, so the model can learn "when many
# candidates share the name, only the address decides". They are relative, not
# corpus counts, so they do not depend on split size.
LOOKALIKE = {"on": False}
LOOKALIKE_FEATURES = ["la_same_name", "la_n_same_name", "la_addr_rank_same", "la_addr_gap",
                      "la_name_gap", "la_n_house_eq", "la_house_unique"]


def lookalike_features(df: pl.DataFrame) -> pl.DataFrame:
    if not LOOKALIKE["on"] or df.is_empty() or "core_tset" not in df.columns:
        return df
    same = pl.col("core_tset") >= 90
    house = pl.col("house_equal").cast(pl.Int8) if "house_equal" in df.columns else pl.col("first_num_eq").cast(pl.Int8)
    df = df.with_columns(la_same_name=same.cast(pl.Int8), la_house=house)
    return df.with_columns(
        la_n_same_name=pl.col("la_same_name").sum().over("sidx").cast(pl.UInt16),
        la_addr_rank_same=pl.when(pl.col("la_same_name") == 1).then(
            (pl.col("addr_tset") * pl.col("la_same_name")).rank("min", descending=True).over("sidx")
        ).cast(pl.Float32),
        la_addr_gap=(pl.col("addr_tset").max().over("sidx") - pl.col("addr_tset")).cast(pl.Float32),
        la_name_gap=(pl.col("core_tset").max().over("sidx") - pl.col("core_tset")).cast(pl.Float32),
        la_n_house_eq=pl.col("la_house").sum().over("sidx").cast(pl.UInt16),
    ).with_columns(
        la_house_unique=((pl.col("la_house") == 1) & (pl.col("la_n_house_eq") == 1)).cast(pl.Int8),
    ).drop("la_house")


class ContextIndex:
    """Slice sorted context to a shard's source-ID bounds before its keyed join."""
    def __init__(self, context: pl.DataFrame):
        self.frame = context.sort("sidx", "tidx")
        self.ids = self.frame["sidx"].to_numpy()

    def attach(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame.join(self.frame.head(0), on=["sidx", "tidx"], how="inner")
        start = int(np.searchsorted(self.ids, frame["sidx"].min(), side="left"))
        stop = int(np.searchsorted(self.ids, frame["sidx"].max(), side="right"))
        return lookalike_features(frame.join(self.frame.slice(start, stop - start), on=["sidx", "tidx"],
                                             how="inner", maintain_order="left"))


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
    from .matrix import quantile_matrix
    params = PARAMS if params is None else params
    dtrain = quantile_matrix(df, feats, 100_000, params)
    evals = [(dtrain, "train")]
    kw = {}
    if valid is not None:
        if valid.is_empty():
            raise ValueError("Validation fold has no candidate pairs")
        evals.append((quantile_matrix(valid, feats, 100_000, params, reference=dtrain), "valid"))
        kw["early_stopping_rounds"] = 50
    return xgb.train(params, dtrain, rounds, evals=evals, verbose_eval=100, **kw)


def stage1_scores(folder: Path, models: dict[int, xgb.Booster], default: list[xgb.Booster], feats,
                  batch_rows: int = BATCH_ROWS, floor: float = 0.0, rescue=None) -> pl.DataFrame:
    """Use complementary models on their training groups, ensemble elsewhere.

    Keep p1 >= floor plus any explicitly supplied neural rescue shortlist.
    Fold 3 influences early stopping, so only fold 4 is an untouched holdout.
    """
    out = []
    for p in feature_parts(folder):
        df = ghostify(pl.read_parquet(p)).with_columns(fold_expr())
        pred = np.empty(len(df), dtype=np.float32)
        unseen = ~df["fold"].is_in(list(models)).to_numpy()
        if unseen.any():
            pred[unseen] = predict_ensemble(default, df.filter(pl.Series(unseen)), feats, batch_rows)
        for model in {id(value): value for value in models.values()}.values():
            folds = [fold for fold, value in models.items() if value is model]
            mask = df["fold"].is_in(folds).to_numpy()
            if mask.any():
                pred[mask] = predict_frame(model, df.filter(pl.Series(mask)), feats, batch_rows)
        current = df.select("sidx", "tidx").with_columns(p1=pl.Series(pred.astype(np.float32)))
        out.append(keep_candidates(current, floor, rescue))
        print(f"scored {p.name}: {len(df):,} pairs", flush=True)
    return pl.concat(out)


def keep_candidates(scores, floor, rescue=None):
    """Rescue is a keyed shortlist, never a blanket lowering of lexical thresholds."""
    if rescue is None:
        return scores.filter(pl.col("p1") >= floor)
    if isinstance(rescue, ContextIndex):
        # Never rebuild a 10M-row rescue hash table for every 150K-row feature shard.
        extra = rescue.attach(scores.filter(pl.col("p1") < floor)).select(scores.columns)
        return pl.concat([scores.filter(pl.col("p1") >= floor), extra])
    marked = scores.join(rescue.with_columns(_rescue=pl.lit(True)), on=["sidx", "tidx"], how="left")
    return marked.filter((pl.col("p1") >= floor) | pl.col("_rescue").fill_null(False)).drop("_rescue")


def neural_rescue(work, split, k):
    if not k:
        return None
    frame = (pl.scan_parquet(work / f"ncands_{split}.parquet").filter(pl.col("nrank") <= k)
             .select("sidx", "tidx").unique().collect(engine="streaming"))
    return ContextIndex(frame)


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
    ap.add_argument("--feature-profile", choices=["baseline", "enhanced"], default="baseline")
    ap.add_argument("--compare-baseline", action="store_true", help="select extra stage-2 features on fold 3 only")
    ap.add_argument("--country-thresholds", action="store_true", help="tune supported countries on fold 3; global fallback elsewhere")
    ap.add_argument("--max-depth", type=positive_int, default=10)
    ap.add_argument("--hist-cache-nodes", type=positive_int, default=2048)
    ap.add_argument("--exclude-country", nargs="*", default=[],
                    help="leave these countries out of all training (unseen-country proxy for France)")
    ap.add_argument("--extra-stage1-folds", action=argparse.BooleanOptionalAction, default=True,
                    help="use folds 8/9 as additional stage-1 training data")
    ap.add_argument("--neural-rescue-k", type=int, default=0)
    ap.add_argument("--lookalike", action="store_true",
                    help="stage-2 look-alike competition features (same-name candidates, address rank)")
    ap.add_argument("--neural", action="store_true",
                    help="add fine-tuned encoder similarity (work/emb) to stage 2; never to stage 1")
    ap.add_argument("--ghost-frac", type=float, default=0.0,
                    help=f"simulate test record density by removing this share of train businesses "
                         f"(test-like: {TEST_DENSITY_GHOST_FRAC})")
    ap.add_argument("--neg-frac", type=float, default=NEG_FRAC,
                    help="fraction of stage-1 negatives kept (reweighted); lowers training RAM")
    args = ap.parse_args()
    if args.neural_rescue_k < 0 or (args.neural_rescue_k and not args.neural):
        ap.error("neural-rescue-k requires --neural and a nonnegative k")
    tag = "".join(f"_no{c}" for c in args.exclude_country)
    if args.ghost_frac > 0:
        tag += f"_ghost{int(round(args.ghost_frac * 100))}"
    work, mdir = Path(args.work), Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    folder = work / "feats_train"
    params = {**PARAMS, "device": args.device, "nthread": args.threads,
              "max_depth": args.max_depth, "max_cached_hist_node": args.hist_cache_nodes}
    t = time.time()

    sample = pl.read_parquet(feature_parts(folder)[0], n_rows=1)
    # Keep the same stage-1 ensemble for both stage-2 ablations. This isolates
    # the effect of the new features and halves the overnight training cost.
    f1 = feature_names(sample, "baseline")
    pair_features = feature_names(sample, args.feature_profile)
    if args.feature_profile == "enhanced" and "token_align_min" not in pair_features:
        raise ValueError("Run feature generation with --enhanced first")
    if "postcode_conflict" not in f1 or "raw_name_ratio" not in f1:
        raise ValueError("Accuracy features are missing; regenerate features (run_features --overwrite)")
    excl = excluded_sidx(work, args.exclude_country)
    ghost_setup(work, args.ghost_frac)
    # Fold 3 may tune early stopping, but fold 4 never influences fitting or selection.
    valid1 = validation_sample(load_train(folder, [3], exclude=excl, features=f1), seed=1)
    rankers = []
    groups = STAGE1_GROUPS if args.extra_stage1_folds else ((0,), (1,))
    for group in groups:
        base = load_train(folder, list(group), args.neg_frac, excl, features=f1)
        print(f"stage1 folds {group}: {len(base):,} pairs (negatives x{args.neg_frac})", flush=True)
        rankers.append(fit(base, f1, valid1, args.rounds, params))
        del base
    del valid1
    m0, m1 = rankers
    m0.save_model(str(mdir / "stage1.json"))
    m1.save_model(str(mdir / "stage1_b.json"))
    print(f"stage1 ensemble trained, {time.time() - t:.0f}s", flush=True)

    held_out = {fold: m1 for fold in groups[0]}
    held_out.update({fold: m0 for fold in groups[1]})
    scores = stage1_scores(folder, held_out, rankers, f1, args.batch_rows, floor=PRUNE, rescue=neural_rescue(work, "train", args.neural_rescue_k))
    ctx = context_features(scores)
    del scores
    if args.neural:
        from .neural import Embeddings, neural_features
        ctx = ctx.join(neural_features(ctx, Embeddings(work, "train")), on=["sidx", "tidx"], how="left")
    print(f"stage1 scored + context, {time.time() - t:.0f}s", flush=True)

    ctx_cols = [c for c in ctx.columns if c not in ("sidx", "tidx")]
    LOOKALIKE["on"] = bool(args.lookalike)
    if args.lookalike:
        ctx_cols += LOOKALIKE_FEATURES
    f2 = pair_features + ctx_cols
    ctx_full = ContextIndex(ctx)
    del ctx
    tr = load_train(folder, [2, 5], exclude=excl, context=ctx_full)
    va_full = load_train(folder, [3], exclude=excl, context=ctx_full)
    va = validation_sample(va_full, seed=2)
    m2 = fit(tr, f2, va, args.rounds, params)
    trials = []
    if args.compare_baseline and args.feature_profile == "enhanced":
        from .run_block import load_split
        from .run_features import ground_truth_pairs
        s1_eval, tg_eval = load_split(work, "train", columns=["idx", "entity_id", "country"])
        truth3 = ground_truth_pairs(Path(args.dataset), s1_eval, tg_eval).filter(fold_expr() == 3)
        anchors3 = s1_eval.select(sidx=pl.col("idx").cast(pl.UInt32), country="country")
        anchors3 = anchors3.filter((fold_expr() == 3) & ~pl.col("country").is_in(args.exclude_country))
        if GHOST["ids"] is not None:
            anchors3 = anchors3.filter(~pl.col("sidx").is_in(GHOST["ids"]))
        del s1_eval, tg_eval
        baseline_feats = f1 + ctx_cols
        baseline = fit(tr, baseline_feats, va, args.rounds, params)
        baseline.save_model(str(mdir / "stage2_baseline.json"))
        m2.save_model(str(mdir / "stage2_enhanced.json"))
        for name, model, cols in (("baseline", baseline, baseline_feats), ("enhanced", m2, f2)):
            prediction = va_full.select("sidx", "tidx").with_columns(
                p2=pl.Series(predict_frame(model, va_full, cols, args.batch_rows)))
            value, threshold = tune_threshold(prediction, truth3, anchors3["sidx"])
            trials.append({"profile": name, "fold3_macro_f05": value, "threshold": threshold})
            print(f"stage2 feature ablation: {name} F0.5={value:.6f}", flush=True)
        if trials[0]["fold3_macro_f05"] >= trials[1]["fold3_macro_f05"]:
            m2, f2 = baseline, baseline_feats
        del baseline, prediction, truth3, anchors3
    del va_full
    del tr, va
    m2.save_model(str(mdir / "stage2.json"))
    print(f"stage2 trained ({m2.best_iteration} it), {time.time() - t:.0f}s", flush=True)

    # stage-2 scores for every pruned pair of every fold (stage 3 builds on them)
    allp = []
    for p in feature_parts(folder):
        part = ghostify(pl.read_parquet(p)).with_columns(fold_expr())
        part = ctx_full.attach(part)  # context already enforces the shared pruning/rescue rule
        allp.append(part.select("sidx", "tidx", "fold", "p1", "label").with_columns(
            p2=pl.Series(predict_frame(m2, part, f2, args.batch_rows))))
    allp = pl.concat(allp)
    allp.write_parquet(work / f"stage2_train{tag}.parquet")
    del ctx_full, part, rankers, held_out
    ev = allp.filter(pl.col("fold").is_in([3, 4]))
    del allp
    ev.write_parquet(work / f"eval_preds{tag}.parquet")

    # truth + anchors for folds 3/4: all source1 records, incl. singletons / uncovered
    from .run_block import load_split
    from .run_features import ground_truth_pairs
    s1, tg = load_split(work, "train", columns=["idx", "entity_id", "country"])
    truth = ground_truth_pairs(Path(args.dataset), s1, tg).with_columns(fold_expr())
    anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country").with_columns(fold_expr())
    if GHOST["ids"] is not None:  # ghosts are not businesses: their records are ownerless
        anchors = anchors.filter(~pl.col("sidx").is_in(GHOST["ids"]))
        truth = truth.filter(~pl.col("sidx").is_in(GHOST["ids"]))
    del s1, tg

    results = {}
    retrieval = (pl.scan_parquet(work / "cands_train.parquet").select("sidx", "tidx")
                 .with_columns(fold_expr()).filter(pl.col("fold").is_in([3, 4])).collect(engine="streaming"))
    retrieval = ghostify(retrieval)
    for fold in (3, 4):
        retrieved = retrieval.filter(pl.col("fold") == fold)
        tr_ = truth.filter(pl.col("fold") == fold)
        a = anchors.filter(pl.col("fold") == fold)["sidx"]
        results[f"fold{fold}_retrieval_oracle"] = macro_f05(retrieved.join(tr_, on=["sidx", "tidx"]), tr_, a)
    del retrieval, retrieved
    # tune only on countries seen in training, so an excluded country stays truly unseen
    tune_anchors = anchors.filter((pl.col("fold") == 3) & ~pl.col("country").is_in(args.exclude_country))["sidx"]
    _, thr = tune_threshold(ev.filter(pl.col("fold") == 3), truth.filter(pl.col("fold") == 3), tune_anchors)
    country_thresholds = tune_country_thresholds(
        ev.filter(pl.col("fold") == 3), truth.filter(pl.col("fold") == 3),
        anchors.filter((pl.col("fold") == 3) & ~pl.col("country").is_in(args.exclude_country)), thr
    ) if args.country_thresholds else {}
    for fold in (3, 4):
        e = ev.filter(pl.col("fold") == fold)
        a = anchors.filter(pl.col("fold") == fold)["sidx"]
        tr_ = truth.filter(pl.col("fold") == fold)
        results[f"fold{fold}_stage2_excl"] = macro_f05(decide_country(e, thr, anchors, country_thresholds), tr_, a)
        results[f"fold{fold}_stage2_global_reference"] = macro_f05(decide(e, thr), tr_, a)
        results[f"fold{fold}_stage2_noexcl"] = macro_f05(e.filter(pl.col("p2") >= thr), tr_, a)
        results[f"fold{fold}_stage1_excl"] = macro_f05(decide(e, thr, "p1"), tr_, a)
        results[f"fold{fold}_oracle"] = macro_f05(e.join(tr_, on=["sidx", "tidx"]), tr_, a)
    hold = ev.filter(pl.col("fold") == 4)
    per = by_country(decide_country(hold, thr, anchors, country_thresholds),
                     truth.filter(pl.col("fold") == 4), anchors.filter(pl.col("fold") == 4))
    results["fold4_by_country"] = per
    results["leaderboard_estimate"] = leaderboard_estimate(per)
    results["excluded_countries"] = args.exclude_country
    results["ghost_frac"] = args.ghost_frac
    results["threshold"] = thr
    results["country_thresholds"] = country_thresholds
    results["prune"] = PRUNE
    results["runtime"] = {"device": args.device, "threads": args.threads, "batch_rows": args.batch_rows}
    results["stage1_neg_frac"] = args.neg_frac
    results["stage2_training_folds"] = [2, 5]
    results["stage3_eligible_folds"] = [3, 4, 6, 7]
    results["feature_version"] = FEATURE_VERSION
    results["neural"] = bool(args.neural)
    results["neural_rescue_k"] = args.neural_rescue_k
    results["lookalike"] = bool(args.lookalike)
    results["feature_trials"] = trials
    results["feature_profile"] = "enhanced" if "token_align_min" in f2 else "baseline"
    results["stage1_feature_profile"] = "baseline"
    results["comparison_note"] = "Same expanded candidates and stage-1 ensemble; stage-2 feature ablation selected on fold 3"
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
    for c, v in per.items():
        print(f"fold4 {c:8s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
    print(f"leaderboard estimate {results['leaderboard_estimate']['estimate']:.4f} "
          f"(France assumed {results['leaderboard_estimate']['france_assumed']})")
    print(f"threshold {thr:.3f}; total {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
