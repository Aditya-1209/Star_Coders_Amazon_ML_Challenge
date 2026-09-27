"""r14 recalibration layer on top of r12: learn when a name match is weak evidence.

French names are built from generic words, so name agreement means less there; models
trained on distinctive US names are over-confident in France (leaderboard: every
stricter French cutoff helped). This layer re-scores r12's uncertain pairs with
country-relative name GENERICITY (token IDF ratios log(N/df) among the split's own
Source 2/3 records per country: split-size invariant, label-free), IDF-weighted name
overlap, address agreement, r12 context and cross-encoder logits. US/India labels
teach "generic name + address disagreement => different business"; France inherits it.

Protocol: cross-fit on fold 3 halves, cutoff tuned on fold 3, fold 4 reported once.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from er_v2.decision import decide_country, tune_threshold
from er_v2.features import token_idf
from er_v2.folds import fold_expr
from er_v2.metrics import by_country, macro_f05
from er_v2.predict import write_lists
from er_v2.r10 import anchors, truth

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, default=Path("D:/star_r10b/work"))
ap.add_argument("--dataset", type=Path, default=Path("C:/Users/ashma/Star_Coders_Amazon_ML_Challenge/student_resource/dataset"))
ap.add_argument("--r12", type=Path, default=Path("D:/star_r14/r12_export"))
ap.add_argument("--ce", type=Path, default=Path("D:/star_r14"))
ap.add_argument("--output", type=Path, default=Path("D:/star_r14/recal"))
ap.add_argument("--band", type=float, nargs=2, default=[0.005, 0.999])
ap.add_argument("--no-ce", action="store_true")
ap.add_argument("--no-generic", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--adapt", action="store_true", help="importance-weight training pairs toward French test pairs")
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
N = args.work / "norm"
LO, HI = args.band
R12_THR = 0.7190439701080322
ADDR = ["house_equal", "house_conflict", "postcode_equal", "postcode_conflict", "addr_both_present", "num_conflict",
        "first_num_eq", "addr_ratio", "addr_tset", "addr_partial", "atok_jacc", "addr_num_jacc", "core_ratio", "core_tset",
        "core_jw", "name_tset", "ntok_jacc", "name_num_conflict", "noaddr_name_exact", "is_s3", "ncos"]


def log(m):
    print(m, flush=True)


def records(split):
    s1 = pl.read_parquet(N / f"{split}_source1.parquet", columns=["idx", "entity_id", "country", "core_n"]).select(
        sidx=pl.col("idx").cast(pl.UInt32), sid="entity_id", country="country", core_s="core_n")
    tg = pl.concat([pl.read_parquet(N / f"{split}_source{i}.parquet", columns=["entity_id", "country", "core_n", "addr_n"])
                    for i in (2, 3)]).with_row_index("tidx").select(
        pl.col("tidx").cast(pl.UInt32), tid="entity_id", country_t="country", core_t="core_n", addr_t="addr_n")
    return s1, tg


def generic_features(pairs, s1, tg):
    """Name-token rarity per country (log N/df over this split's targets)."""
    idf = token_idf(tg.select(idx="tidx", country="country_t", core_n="core_t", addr_n="addr_t"))["n"]
    def side(frame, key, col):
        e = (frame.select(key, "country", tok=pl.col(col).str.split(" ")).explode("tok")
             .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique([key, "tok"])
             .join(idf, on=["country", "tok"], how="left").with_columns(pl.col("w").fill_null(pl.col("w").max().over("country"))))
        return e
    ps = pairs.select("sidx", "tidx", "country")
    ls = side(s1.join(ps.select("sidx").unique(), on="sidx", how="semi"), "sidx", "core_s")
    rt = side(tg.rename({"country_t": "country"}).join(ps.select("tidx").unique(), on="tidx", how="semi"), "tidx", "core_t")
    lagg = ls.group_by("sidx").agg(gen_s_min=pl.col("w").min(), gen_s_mean=pl.col("w").mean(), gen_s_sum=pl.col("w").sum(),
                                   gen_s_n=pl.len().cast(pl.Float32))
    ragg = rt.group_by("tidx").agg(gen_t_min=pl.col("w").min(), gen_t_mean=pl.col("w").mean(), gen_t_sum=pl.col("w").sum())
    shared = (ps.join(ls.select("sidx", "tok", "w"), on="sidx").join(rt.select("tidx", "tok"), on=["tidx", "tok"])
              .group_by("sidx", "tidx").agg(gen_shared=pl.col("w").sum(), gen_shared_max=pl.col("w").max()))
    out = pairs.join(lagg, on="sidx", how="left").join(ragg, on="tidx", how="left").join(shared, on=["sidx", "tidx"], how="left")
    return out.with_columns(pl.col("gen_shared", "gen_shared_max").fill_null(0.0)).with_columns(
        gen_wjacc=pl.col("gen_shared") / (pl.col("gen_s_sum") + pl.col("gen_t_sum") - pl.col("gen_shared")).clip(1e-6),
        gen_cont_s=pl.col("gen_shared") / pl.col("gen_s_sum").clip(1e-6),
        gen_cont_t=pl.col("gen_shared") / pl.col("gen_t_sum").clip(1e-6))


def build(split, r12_file, ce_files, fold=None):
    s1, tg = records(split)
    r = (pl.read_parquet(r12_file).select("sid", "tid", "score").join(s1.select("sidx", "sid", "country"), on="sid")
         .join(tg.select("tidx", "tid"), on="tid").select("sidx", "tidx", "country", pl.col("score").cast(pl.Float64)))
    s = pl.col("score")
    r = r.with_columns(r_rank_s=s.rank("ordinal", descending=True).over("sidx").cast(pl.Float32),
                       r_gap_s=(s.max().over("sidx") - s), r_n50_s=(s >= 0.5).sum().over("sidx").cast(pl.Float32),
                       r_n72_s=(s >= R12_THR).sum().over("sidx").cast(pl.Float32), r_sum_s=s.sum().over("sidx"),
                       r_logit=(s.clip(1e-7, 1 - 1e-7) / (1 - s.clip(1e-7, 1 - 1e-7))).log())
    band = r.filter((s > LO) & (s < HI))
    cols = [c for c in ADDR]
    feats = (pl.scan_parquet(args.work / f"stage3_{split}.parquet")
             .select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32), *cols)
             .join(band.lazy().select("sidx", "tidx"), on=["sidx", "tidx"], how="semi")
             .unique(["sidx", "tidx"]).collect(engine="streaming"))
    band = band.join(feats, on=["sidx", "tidx"], how="left")
    if not args.no_generic:
        band = generic_features(band, s1, tg)
    for k, f in enumerate(ce_files):
        ce = pl.read_parquet(f).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32), pl.col("ce_logit").alias(f"ce{k}"))
        band = band.join(ce, on=["sidx", "tidx"], how="left")
    log(f"{split} fold {fold}: {len(r):,} r12 pairs, {len(band):,} in band, "
        f"{band['addr_tset'].null_count():,} without pair feats")
    return r, band


FEATS = None


def matrix(frame):
    return xgb.DMatrix(frame.select(FEATS).to_numpy().astype(np.float32), feature_names=FEATS)


def fit(tr, va):
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
              "max_depth": 6, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
              "reg_lambda": 5.0, "seed": 1414}
    dtr, dva = matrix(tr), matrix(va)
    dtr.set_label(tr["y"].to_numpy()); dva.set_label(va["y"].to_numpy())
    if "dw" in tr.columns:
        dtr.set_weight(tr["dw"].to_numpy()); dva.set_weight(va["dw"].to_numpy())
    m = xgb.train(params, dtr, 3000, evals=[(dva, "v")], early_stopping_rounds=100, verbose_eval=False)
    log(f"fit {len(tr):,} rows, iter {m.best_iteration}, logloss {m.best_score:.5f}")
    return m


def pred(models, frame):
    d = matrix(frame)
    return np.mean([m.predict(d, iteration_range=(0, m.best_iteration + 1)) for m in models], axis=0)


def apply(r, band, values):
    upd = band.select("sidx", "tidx").with_columns(new=pl.Series(values.astype(np.float64)))
    return r.select("sidx", "tidx", "score").join(upd, on=["sidx", "tidx"], how="left").select(
        "sidx", "tidx", score=pl.coalesce("new", "score"))


ce_train = [] if args.no_ce else [args.ce / "ce_train_uncertain.parquet", args.ce / "ce_train_uncertain_b.parquet"]
ce_test = [] if args.no_ce else [args.ce / "ce_test_uncertain.parquet", args.ce / "ce_test_uncertain_b.parquet"]
target = truth(args, [3, 4]).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)).unique()
country = anchors(args)
a3, a4 = country.filter(fold_expr() == 3), country.filter(fold_expr() == 4)
r3, b3 = build("train", args.r12 / "r12_fold3.parquet", ce_train, 3)
r4, b4 = build("train", args.r12 / "r12_fold4.parquet", ce_train, 4)
lab = target.with_columns(y=pl.lit(1, pl.Int8))
b3 = b3.join(lab, on=["sidx", "tidx"], how="left").with_columns(pl.col("y").fill_null(0), h=(pl.col("sidx").hash(seed=1010) % 2))
b4 = b4.join(lab, on=["sidx", "tidx"], how="left").with_columns(pl.col("y").fill_null(0))
FEATS = [c for c in b3.columns if c not in {"sidx", "tidx", "country", "y", "h"}]
log(f"features ({len(FEATS)}): {FEATS}")
if args.adapt:
    rt, bt = build("test", args.r12 / "r12_test.parquet", ce_test)
    fr = bt.filter(pl.col("country") == "France")
    DF = [c for c in FEATS if not c.startswith("ce")]  # CE coverage differs by candidate source
    dom = pl.concat([fr.select(DF).sample(min(len(fr), 250_000), seed=1).with_columns(d=pl.lit(1)),
                     b3.select(DF).with_columns(d=pl.lit(0))])
    dd = xgb.DMatrix(dom.select(DF).to_numpy().astype(np.float32), label=dom["d"].to_numpy(), feature_names=DF)
    dm = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cuda", "max_depth": 5, "eta": 0.1,
                    "eval_metric": "auc", "subsample": 0.8, "seed": 7}, dd, 200)
    auc = dm.eval(dd)
    pd_ = dm.predict(xgb.DMatrix(b3.select(DF).to_numpy().astype(np.float32), feature_names=DF))
    prior = (dom["d"] == 1).mean() / (dom["d"] == 0).mean()
    w = np.clip(pd_ / (1 - pd_) / prior, 0.05, 20.0)
    b3 = b3.with_columns(dw=pl.Series(w.astype(np.float32)))
    dimp = dm.get_score(importance_type="gain")
    log(f"domain France-vs-train {auc}; weights mean {w.mean():.2f} p90 {np.quantile(w, .9):.2f} max {w.max():.1f}; "
        f"ESS {w.sum() ** 2 / (w ** 2).sum() / len(w):.3f}; top: " +
        json.dumps({k: round(v, 1) for k, v in sorted(dimp.items(), key=lambda kv: -kv[1])[:8]}))
ua, ub = b3.filter(pl.col("h") == 0), b3.filter(pl.col("h") == 1)
ma, mb = fit(ua, ub), fit(ub, ua)
imp = ma.get_score(importance_type="gain")
log("top gain: " + json.dumps({k: round(v, 1) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:15]}))
b3p = pl.concat([ub.with_columns(p=pl.Series(pred([ma], ub))), ua.with_columns(p=pl.Series(pred([mb], ua)))])
s3 = apply(r3, b3p, b3p["p"].to_numpy())
s4 = apply(r4, b4, pred([ma, mb], b4))
res = {}
for name, x3, x4 in (("r12", r3.select("sidx", "tidx", "score"), r4.select("sidx", "tidx", "score")), ("recal", s3, s4)):
    v3, thr = tune_threshold(x3, target, a3["sidx"], "score")
    p4 = decide_country(x4, thr, a4, {}, "score")
    m4 = macro_f05(p4, target, a4["sidx"])
    res[name] = {"threshold": thr, "fold3": v3, "fold4": m4["macro_f05"], "P": m4["pair_precision"], "R": m4["pair_recall"],
                 "by_country": {k: v["macro_f05"] for k, v in by_country(p4, target, a4).items()}}
    log(f"{name}: " + json.dumps(res[name]))
(args.output / "recal_metrics.json").write_text(json.dumps({"results": res, "features": FEATS}, indent=2))
if args.test:
    if not args.adapt:
        rt, bt = build("test", args.r12 / "r12_test.parquet", ce_test)
    st = apply(rt, bt, pred([ma, mb], bt))
    st.write_parquet(args.output / "recal_test_scores.parquet")
    ct = anchors(args, "test")
    thr = res["recal"]["threshold"]
    m = decide_country(st, thr, ct, {}, "score").join(ct, on="sidx")
    base = decide_country(rt.select("sidx", "tidx", "score"), R12_THR, ct, {}, "score").join(ct, on="sidx")
    for c in ("US", "India", "France"):
        a_, b_ = m.filter(pl.col("country") == c).select("sidx", "tidx"), base.filter(pl.col("country") == c).select("sidx", "tidx")
        log(f"test {c}: recal {len(a_):,} vs r12 {len(b_):,}  added {a_.join(b_, on=['sidx','tidx'], how='anti').height:,} "
            f"removed {b_.join(a_, on=['sidx','tidx'], how='anti').height:,}")
