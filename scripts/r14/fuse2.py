"""r14: re-decide the graph model's uncertain pairs with cross-encoder evidence.

Stage 3 (graph) scores every candidate. Pairs it is sure about (score <= LO or >= HI)
keep that score. Pairs in between are re-scored by a small XGBoost "fusion" model that
sees the graph score, the cross-encoder logit, both models' within-business and
within-target competition, and the strongest stage-3 pair features.

Protocol (no label leaks into fold 4):
  * fusion is cross-fitted on fold 3: model A trains on half 3A and predicts 3B, model B
    the reverse, so every fold-3 pair has an out-of-sample fused score;
  * the exclusive macro-F0.5 cutoff (global, and per country) is tuned on fold 3;
  * fold 4 (holdout) and test use the mean of models A and B, and fold 4 is reported
    once against the untouched graph baseline tuned the same way.

usage: python scripts/r14/fuse.py --work W --dataset D --ce-train CE_TR --ce-test CE_TE --output OUT
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from er_v2.decision import decide_country, tune_country_thresholds, tune_threshold
from er_v2.metrics import by_country, macro_f05
from er_v2.r10 import anchors, truth
from er_v2.train import fold_expr

LO, HI = 0.01, 0.99  # overridden by --band
PAIR_COLS = ["ncos", "ncos_rank_s", "ncos_gap_s", "ncos_rank_t", "ncos_gap_t", "raw_name_ratio",
             "name_ratio", "name_tset", "core_ratio", "core_jw", "cc_partial", "addr_ratio", "addr_tset",
             "addr_partial", "addr_both_present", "house_equal", "house_conflict", "postcode_equal",
             "postcode_conflict", "num_conflict", "name_num_conflict", "first_num_eq", "is_s3", "brank",
             "t_rank", "t_nc", "s_nc", "rescue", "token_align_min", "noaddr_name_exact", "noaddr_name_align",
             "p1", "hop_score", "hop_n", "n_anchor", "sup_both_max", "sup_name_tset", "sup_addr_tset",
             "sup_n90", "pa_max"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
          "max_depth": 6, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 5, "reg_lambda": 5.0, "seed": 1414}


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


def half():
    return (pl.col("sidx").hash(seed=1010) % 2).cast(pl.Int8)


def context(scores: pl.DataFrame) -> pl.DataFrame:
    """Competition among ALL graph candidates (label-free, computed before any fold split)."""
    s = pl.col("score")
    return scores.with_columns(
        g_rank_s=s.rank("ordinal", descending=True).over("sidx").cast(pl.UInt16),
        g_n50_s=(s >= 0.5).sum().over("sidx").cast(pl.UInt16),
        g_n99_s=(s >= HI).sum().over("sidx").cast(pl.UInt16),
        g_max_s=s.max().over("sidx"),
        g_sum_s=s.sum().over("sidx"),
        g_cnt_s=pl.len().over("sidx").cast(pl.UInt16),
        g_rank_t=s.rank("ordinal", descending=True).over("tidx").cast(pl.UInt16),
        g_max_t=s.max().over("tidx"),
        g_n50_t=(s >= 0.5).sum().over("tidx").cast(pl.UInt16),
        g_cnt_t=pl.len().over("tidx").cast(pl.UInt16),
    ).with_columns(g_gap_s=pl.col("g_max_s") - s, g_gap_t=pl.col("g_max_t") - s)


def ce_context(ce: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("ce_logit")
    return ce.with_columns(
        ce_prob=(1 / (1 + (-c).exp())).cast(pl.Float32),
        ce_rank_s=c.rank("ordinal", descending=True).over("sidx").cast(pl.UInt16),
        ce_gap_s=(c.max().over("sidx") - c).cast(pl.Float32),
        ce_npos_s=(c > 0).sum().over("sidx").cast(pl.UInt16),
        ce_rank_t=c.rank("ordinal", descending=True).over("tidx").cast(pl.UInt16),
        ce_gap_t=(c.max().over("tidx") - c).cast(pl.Float32),
        ce_npos_t=(c > 0).sum().over("tidx").cast(pl.UInt16),
    )


def build(work, split, ce_paths, folds=None, all_features=False):
    path = work / ("eval_preds_stage3.parquet" if split == "train" else "test_preds_stage3.parquet")
    scores = pl.read_parquet(path)
    if folds is not None:
        scores = scores.filter(pl.col("fold").is_in(folds))
    scores = context(scores.with_columns(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)))
    unsure = scores.filter((pl.col("score") > LO) & (pl.col("score") < HI))
    ce = None
    for k, path in enumerate(ce_paths):
        one = pl.read_parquet(path).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32),
                                           pl.col("ce_logit").alias(f"ce_{k}"))
        ce = one if ce is None else ce.join(one, on=["sidx", "tidx"], how="inner")
    names = [f"ce_{k}" for k in range(len(ce_paths))]
    ce = ce.with_columns(ce_logit=pl.mean_horizontal(names).cast(pl.Float32))
    if len(names) == 1:
        ce = ce.drop(names)
    unsure = unsure.join(ce_context(ce), on=["sidx", "tidx"], how="left")
    missing = unsure["ce_logit"].null_count()
    if missing:
        raise ValueError(f"{missing:,} uncertain {split} pairs have no cross-encoder score")
    source = pl.scan_parquet(work / f"stage3_{split}.parquet")
    cols = PAIR_COLS
    if all_features:
        # exactly the (shift-checked) features the stage-3 graph model itself used
        import json
        used = json.loads((work / "models/stage3_metrics.json").read_text())["features"]
        have = set(source.collect_schema().names())
        cols = [c for c in used if c in have and c not in unsure.columns]
    pairs = (source
             .select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32), *cols,
                     *([] if "direct" in unsure.columns or "direct" in cols else ["direct"]))
             .join(unsure.lazy().select("sidx", "tidx"), on=["sidx", "tidx"], how="semi")
             .unique(["sidx", "tidx"]).collect(engine="streaming"))
    unsure = unsure.join(pairs, on=["sidx", "tidx"], how="left")
    log(f"{split}: {len(scores):,} pairs, {len(unsure):,} uncertain, "
        f"{unsure['ncos'].null_count():,} without stage-3 features")
    return scores, unsure


def feature_names(frame):
    drop = {"sidx", "tidx", "fold", "label", "h"}
    return [c for c in frame.columns if c not in drop]


def matrix(frame, features):
    return xgb.DMatrix(frame.select(features).to_numpy().astype(np.float32), feature_names=features)


def fit(train, valid, features, args):
    params = {**PARAMS, "device": args.device, "nthread": args.threads, "max_depth": args.depth, "eta": args.eta}
    dtr, dva = matrix(train, features), matrix(valid, features)
    if args.business_weights:
        for d, f in ((dtr, train), (dva, valid)):
            d.set_weight((1.0 / f.group_by("sidx").agg(n=pl.len()).join(f.select("sidx"), on="sidx", how="right")["n"]).to_numpy())
    dtr.set_label(train["label"].to_numpy())
    dva.set_label(valid["label"].to_numpy())
    model = xgb.train(params, dtr, args.rounds, evals=[(dva, "valid")], early_stopping_rounds=100,
                      verbose_eval=False)
    log(f"fusion fit: {len(train):,} rows, best iteration {model.best_iteration}, "
        f"valid logloss {model.best_score:.5f}")
    return model


def predict(models, frame, features):
    d = matrix(frame, features)
    return np.mean([m.predict(d, iteration_range=(0, m.best_iteration + 1)) for m in models], axis=0)


def apply(scores, unsure, fused):
    upd = unsure.select("sidx", "tidx").with_columns(fused=pl.Series(fused.astype(np.float64)))
    out = scores.select("sidx", "tidx", "score").join(upd, on=["sidx", "tidx"], how="left")
    return out.with_columns(score=pl.coalesce("fused", pl.col("score").cast(pl.Float64))).drop("fused")


def decide(scores, anchor, choice):
    return decide_country(scores.join(anchor.select("sidx"), on="sidx", how="semi"), choice["threshold"],
                          anchor, choice["country_thresholds"], "score")


def tune(scores, target, anchor):
    value, threshold = tune_threshold(scores, target, anchor["sidx"], "score")
    cutoffs = tune_country_thresholds(scores, target, anchor, threshold, "score")
    choice = {"threshold": threshold, "country_thresholds": cutoffs, "fold3_global": value}
    choice["fold3"] = macro_f05(decide(scores, anchor, choice), target, anchor["sidx"])["macro_f05"]
    return choice


def main():
    global LO, HI
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ce-train", type=Path, nargs="+", required=True)
    ap.add_argument("--ce-test", type=Path, nargs="+")
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--business-weights", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--band", type=float, nargs=2, default=[LO, HI], metavar=("LO", "HI"))
    args = ap.parse_args()
    LO, HI = args.band
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"lo": LO, "hi": HI}

    target = truth(args, [3, 4]).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)).unique()
    country = anchors(args)
    a3, a4 = country.filter(fold_expr() == 3), country.filter(fold_expr() == 4)
    scores, unsure = build(args.work, "train", args.ce_train, [3, 4], args.all_features)
    unsure = unsure.join(target.with_columns(label=pl.lit(1, pl.Int8)), on=["sidx", "tidx"], how="left") \
                   .with_columns(pl.col("label").fill_null(0), h=half())
    features = feature_names(unsure)
    report["features"] = features
    u3, u4 = unsure.filter(pl.col("fold") == 3), unsure.filter(pl.col("fold") == 4)
    log(f"uncertain fold3 {len(u3):,} (pos {u3['label'].sum():,}), fold4 {len(u4):,} (pos {u4['label'].sum():,})")
    for name, frame in (("fold3", u3), ("fold4", u4)):
        ce = frame["ce_logit"].to_numpy()
        y = frame["label"].to_numpy()
        report[f"ce_alone_acc_{name}"] = float(((ce > 0) == (y == 1)).mean())
        report[f"graph_alone_acc_{name}"] = float(((frame["score"].to_numpy() > 0.5) == (y == 1)).mean())

    ua, ub = u3.filter(pl.col("h") == 0), u3.filter(pl.col("h") == 1)
    model_a, model_b = fit(ua, ub, features, args), fit(ub, ua, features, args)
    for i, m in enumerate((model_a, model_b)):
        m.save_model(args.output / f"fusion_{'ab'[i]}.json")
    imp = model_a.get_score(importance_type="gain")
    report["top_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:20])
    u3 = pl.concat([ub.with_columns(fused=pl.Series(predict([model_a], ub, features))),
                    ua.with_columns(fused=pl.Series(predict([model_b], ua, features)))])
    fused3 = apply(scores.filter(pl.col("fold") == 3), u3, u3["fused"].to_numpy())
    fused4 = apply(scores.filter(pl.col("fold") == 4), u4, predict([model_a, model_b], u4, features))
    base3, base4 = (scores.filter(pl.col("fold") == f).select("sidx", "tidx", "score") for f in (3, 4))

    results = {}
    for name, s3, s4 in (("graph_baseline", base3, base4), ("r14_fusion", fused3, fused4)):
        choice = tune(s3, target, a3)
        for key in ("global", "country"):
            c = choice if key == "country" else {**choice, "country_thresholds": {}}
            pred = decide(s4, a4, c)
            results[f"{name}_{key}"] = {"threshold": c["threshold"], "country_thresholds": c["country_thresholds"],
                                        "fold3": choice["fold3" if key == "country" else "fold3_global"],
                                        "fold4": macro_f05(pred, target, a4["sidx"]),
                                        "fold4_by_country": {k: v["macro_f05"] for k, v in by_country(pred, target, a4).items()}}
        log(f"{name}: fold3 {choice['fold3_global']:.5f}/{choice['fold3']:.5f}  "
            f"fold4 {results[name + '_global']['fold4']['macro_f05']:.5f}/{results[name + '_country']['fold4']['macro_f05']:.5f}")
    report["results"] = results
    # Selection uses fold 3 only; fold 4 is reported, never used to choose.
    best = max((k for k in results if k.startswith("r14_fusion")), key=lambda k: results[k]["fold3"])
    report["selected"] = best
    (args.output / "r14_metrics.json").write_text(json.dumps(report, indent=2))
    log(json.dumps({k: {"fold3": v["fold3"], "fold4": v["fold4"]["macro_f05"], "by_country": v["fold4_by_country"]}
                    for k, v in results.items()}, indent=1))

    if args.ce_test is None or args.no_test:
        return
    from er_v2.predict import write_lists
    tscores, tunsure = build(args.work, "test", args.ce_test, None, args.all_features)
    fused = apply(tscores, tunsure, predict([model_a, model_b], tunsure, features))
    tcountry = anchors(args, "test")
    chosen = results[best]
    matches = decide_country(fused, chosen["threshold"], tcountry, chosen["country_thresholds"], "score")
    s1 = pl.read_parquet(args.work / "norm/test_source1.parquet", columns=["idx", "entity_id"])
    targets = pl.concat([pl.read_parquet(args.work / f"norm/test_source{i}.parquet", columns=["entity_id"])
                         for i in (2, 3)])["entity_id"]
    fused = fused.sort("sidx", "score", descending=[False, True])
    write_lists(args.output / "candidate_pairs.tsv", s1, fused, targets, "candidate_entity_ids")
    write_lists(args.output / "matching_results.tsv", s1, matches.sort("sidx", "score", descending=[False, True]),
                targets, "matched_entity_ids")
    fused.write_parquet(args.output / "r14_test_scores.parquet")
    stats = matches.join(tcountry, on="sidx").group_by("country").agg(pairs=pl.len(), businesses=pl.col("sidx").n_unique())
    log(f"test matches: {len(matches):,}\n{stats}")


if __name__ == "__main__":
    main()
