"""Does the fine-tuned encoder add signal on top of the current stage 2?

Uses work/stage2_train.parquet (out-of-fold stage-2 scores for all survivors).
Trains a small XGBoost on folds 6/7 with (a) [p1, p2] and (b) [p1, p2 + neural
features], tunes the exclusive-F0.5 cutoff on fold 3, and reports fold 4.
None of these folds was used to fine-tune the encoder (folds 0/1/8/9).

usage: python scripts/analysis/neural_quickcheck.py
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from er_v2.decision import tune_threshold
from er_v2.metrics import by_country, macro_f05
from er_v2.neural import NEURAL_FEATURES, Embeddings, neural_features
from er_v2.run_block import load_split
from er_v2.run_features import ground_truth_pairs
from er_v2.train import decide, fold_expr

work = Path("work")
st2 = pl.read_parquet(work / "stage2_train.parquet").with_columns(fold_expr())
st2 = st2.filter(pl.col("fold").is_in([3, 4, 6, 7]))
st2 = st2.join(neural_features(st2.select("sidx", "tidx"), Embeddings(work, "train")), on=["sidx", "tidx"])
s1, tg = load_split(work, "train")
truth = ground_truth_pairs(Path("student_resource/dataset"), s1, tg).with_columns(fold_expr())
anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country").with_columns(fold_expr())
params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
              eta=0.05, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.9, seed=1)
res = {}
for name, feats in {"stage2_only": ["p1", "p2"], "stage2_plus_neural": ["p1", "p2", *NEURAL_FEATURES]}.items():
    tr, tu = st2.filter(pl.col("fold").is_in([6, 7])), st2.filter(pl.col("fold") == 3)
    d = xgb.DMatrix(tr.select(feats).to_numpy(), tr["label"].to_numpy(), feature_names=feats)
    v = xgb.DMatrix(tu.select(feats).to_numpy(), tu["label"].to_numpy(), feature_names=feats)
    m = xgb.train(params, d, 2000, evals=[(v, "v")], early_stopping_rounds=50, verbose_eval=False)
    pred = st2.with_columns(s=pl.Series(m.predict(xgb.DMatrix(st2.select(feats).to_numpy(), feature_names=feats),
                                                  iteration_range=(0, m.best_iteration + 1))))
    _, thr = tune_threshold(pred.filter(pl.col("fold") == 3), truth.filter(pl.col("fold") == 3),
                            anchors.filter(pl.col("fold") == 3)["sidx"], "s")
    h = pred.filter(pl.col("fold") == 4)
    a4 = anchors.filter(pl.col("fold") == 4)
    r = macro_f05(decide(h, thr, "s"), truth.filter(pl.col("fold") == 4), a4["sidx"])
    per = by_country(decide(h, thr, "s"), truth.filter(pl.col("fold") == 4), a4)
    res[name] = {"fold4": r, "threshold": thr, "by_country": {c: v["macro_f05"] for c, v in per.items()}}
    print(f"{name:20s} fold4 F0.5={r['macro_f05']:.4f} P={r['pair_precision']:.4f} R={r['pair_recall']:.4f} "
          f"| " + " ".join(f"{c}={v['macro_f05']:.4f}" for c, v in per.items()), flush=True)
# how separable are matches from non-matches by raw cosine alone?
h = st2.filter(pl.col("fold") == 4)
from sklearn.metrics import roc_auc_score
print(f"cosine-only AUC on fold-4 survivors: {roc_auc_score(h['label'].to_numpy(), h['ncos'].to_numpy()):.4f}; "
      f"p2 AUC: {roc_auc_score(h['label'].to_numpy(), h['p2'].to_numpy()):.4f}")
Path(work / "neural_quickcheck.json").write_text(json.dumps(res, indent=2, default=float))
