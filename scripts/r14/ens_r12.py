"""Ensemble r12's final stacker scores with r14's fused scores.

Pairs are aligned by entity ids (robust to index differences). A pair scored by only
one model keeps that model's score. Weight w (r12 share) and the exclusive macro-F0.5
cutoff are chosen on fold 3; fold 4 is reported once, alongside r12 and r14 alone
(each re-tuned on fold 3 the same way).

usage: python scripts/r14/ens_r12.py --work W --dataset D --r12 DIR --r14 DIR --output OUT
"""
import argparse
import json
from pathlib import Path

import polars as pl

from er_v2.decision import decide_country, tune_threshold
from er_v2.folds import fold_expr
from er_v2.metrics import by_country, macro_f05
from er_v2.predict import write_lists
from er_v2.r10 import anchors, truth

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--dataset", type=Path, required=True)
ap.add_argument("--r12", type=Path, required=True, help="dir with r12_{fold3,fold4,test}.parquet")
ap.add_argument("--r14", type=Path, required=True, help="dir with r14_{fold3,fold4}.parquet, r14_test_scores.parquet")
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
N = args.work / "norm"


def ids(split):
    s1 = pl.read_parquet(N / f"{split}_source1.parquet", columns=["idx", "entity_id"]).select(
        sidx=pl.col("idx").cast(pl.UInt32), sid="entity_id")
    tg = pl.concat([pl.read_parquet(N / f"{split}_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)]) \
        .with_row_index("tidx").select(pl.col("tidx").cast(pl.UInt32), tid="entity_id")
    return s1, tg


def load(split, r12_file, r14_file):
    s1, tg = ids(split)
    a = pl.read_parquet(r12_file).select("sid", "tid", r12="score").join(s1, on="sid").join(tg, on="tid") \
        .select("sidx", "tidx", pl.col("r12").cast(pl.Float64))
    b = pl.read_parquet(r14_file).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32),
                                         pl.col("score").cast(pl.Float64).alias("r14"))
    both = a.join(b, on=["sidx", "tidx"], how="full", coalesce=True)
    print(f"{split}: r12 {len(a):,}, r14 {len(b):,}, both {both.filter(pl.col('r12').is_not_null() & pl.col('r14').is_not_null()).height:,}", flush=True)
    return both


def mix(frame, w):
    return frame.select("sidx", "tidx", score=pl.when(pl.col("r12").is_null()).then("r14")
                        .when(pl.col("r14").is_null()).then("r12")
                        .otherwise(w * pl.col("r12") + (1 - w) * pl.col("r14")))


country = anchors(args)
a3, a4 = country.filter(fold_expr() == 3), country.filter(fold_expr() == 4)
target = truth(args, [3, 4]).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)).unique()
f3 = load("train", args.r12 / "r12_fold3.parquet", args.r14 / "r14_fold3.parquet")
f4 = load("train", args.r12 / "r12_fold4.parquet", args.r14 / "r14_fold4.parquet")
trials = []
for w in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
    s3 = mix(f3, w)
    value, thr = tune_threshold(s3, target, a3["sidx"], "score")
    pred4 = decide_country(mix(f4, w), thr, a4, {}, "score")
    m4 = macro_f05(pred4, target, a4["sidx"])
    trials.append({"w_r12": w, "threshold": thr, "fold3": value, "fold4": m4["macro_f05"],
                   "fold4_by_country": {k: v["macro_f05"] for k, v in by_country(pred4, target, a4).items()}})
    print(json.dumps(trials[-1]), flush=True)
best = max(trials, key=lambda t: t["fold3"])  # chosen on fold 3 only
report = {"selected": best, "trials": trials}
(args.output / "ens_metrics.json").write_text(json.dumps(report, indent=2))
print("SELECTED", json.dumps(best), flush=True)

ft = load("test", args.r12 / "r12_test.parquet", args.r14 / "r14_test_scores.parquet")
scores = mix(ft, best["w_r12"]).sort("sidx", "score", descending=[False, True])
tc = anchors(args, "test")
matches = decide_country(scores, best["threshold"], tc, {}, "score")
s1 = pl.read_parquet(N / "test_source1.parquet", columns=["idx", "entity_id"])
targets = pl.concat([pl.read_parquet(N / f"test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
write_lists(args.output / "candidate_pairs.tsv", s1, scores, targets, "candidate_entity_ids")
write_lists(args.output / "matching_results.tsv", s1, matches.sort("sidx", "score", descending=[False, True]),
            targets, "matched_entity_ids")
scores.write_parquet(args.output / "ens_test_scores.parquet")
print(f"test matches {len(matches):,}", flush=True)
