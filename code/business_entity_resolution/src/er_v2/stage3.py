"""Stage 3 driver: two-hop expansion + support features + final model.

train mode : uses work/stage2_train.parquet (out-of-fold stage-2 scores),
             trains on folds 6/7, tunes the threshold on fold 3, reports fold 4.
test mode  : uses work/test_preds.parquet, writes the final TSVs.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from .block import build_target_index, make_keys
from .features import compute, pair_frame, record_frames
from .graph import anchors_of, expand, support_features
from .metrics import macro_f05
from .run_block import load_split
from .train import PARAMS, X, decide, fold_expr, predict

TRAIN_FOLDS, TUNE_FOLD, HOLD_FOLD = [6, 7], 3, 4
BLOCK_COLS = ["bscore", "nkeys", "brank", "b_rel_s", "b_rel_t", "t_rank", "t_nc", "s_nc"]


def build(split: str, work: Path, stage2: pl.DataFrame, feats_dir: Path, log) -> pl.DataFrame:
    """Return stage-3 feature rows for direct (pruned) + two-hop candidates."""
    s1, tg = load_split(work, split)
    s1 = s1.with_columns(pl.col("idx").cast(pl.UInt32))
    tg = tg.with_columns(pl.col("idx").cast(pl.UInt32))
    n_s2 = len(pl.read_parquet(work / "norm" / f"{split}_source2.parquet", columns=["idx"]))
    anchors = anchors_of(stage2)
    log(f"anchors {len(anchors):,}")

    tkeys = make_keys(tg)
    tindex = build_target_index(tkeys, len(tg))
    tkeys = tkeys.drop("kind")
    gc.collect()
    hop = expand(anchors, tkeys, tindex)
    del tkeys, tindex
    gc.collect()
    log(f"two-hop pairs {len(hop):,}")

    direct = stage2.select("sidx", "tidx", "p1", "p2").with_columns(direct=pl.lit(1, pl.Int8))
    pairs = direct.join(hop, on=["sidx", "tidx"], how="full", coalesce=True)
    pairs = pairs.with_columns(pl.col("direct").fill_null(0))
    log(f"stage-3 candidates {len(pairs):,} ({len(pairs) / s1.height:.2f} per source1 overall)")

    # pairwise features: reuse stored ones for direct pairs, compute for new pairs
    keyset = pairs.select("sidx", "tidx")
    old = []
    for p in sorted(feats_dir.glob("part_*.parquet")):
        old.append(pl.read_parquet(p).join(keyset, on=["sidx", "tidx"], how="semi"))
    old = pl.concat(old)
    if "label" in old.columns:
        old = old.drop("label")
    new_pairs = keyset.join(old.select("sidx", "tidx"), on=["sidx", "tidx"], how="anti")
    left, right = record_frames(s1, tg)
    newf = new_pairs.with_columns(*[pl.lit(None, pl.Float32).alias(c) for c in BLOCK_COLS])
    newf = compute(pair_frame(newf, left, right, n_s2))
    log(f"computed features for {len(newf):,} new pairs")
    base = pl.concat([old, newf.select(old.columns)], how="vertical_relaxed")
    del old, newf, left
    gc.collect()

    txt = right.select("idx", "core_r", "addr_r", "cc_r")
    ids = pairs["sidx"].unique().sort()
    sup = []
    for i in range(0, len(ids), 300_000):
        chunk = pairs.join(pl.DataFrame({"sidx": ids[i:i + 300_000]}), on="sidx", how="semi")
        sup.append(support_features(chunk, anchors, txt))
    sup = pl.concat(sup)
    del right
    gc.collect()
    out = base.join(pairs, on=["sidx", "tidx"], how="left").join(sup, on=["sidx", "tidx"], how="left")
    log(f"support features done, {out.width} columns")
    return out


def feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("sidx", "tidx", "label", "fold")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--model-dir", default="models/v2")
    ap.add_argument("--output", default="output")
    ap.add_argument("--rounds", type=int, default=2000)
    args = ap.parse_args()
    work, mdir = Path(args.work), Path(args.model_dir)
    t = time.time()
    log = lambda m: print(f"[{time.time() - t:6.0f}s] {m}", flush=True)

    if args.split == "train":
        from .run_features import ground_truth_pairs
        st2 = pl.read_parquet(work / "stage2_train.parquet").with_columns(fold_expr())
        keep = TRAIN_FOLDS + [TUNE_FOLD, HOLD_FOLD]
        st2 = st2.filter(pl.col("fold").is_in(keep)).drop("label", "fold")
        df = build("train", work, st2, work / "feats_train", log)
        s1, tg = load_split(work, "train")
        truth = ground_truth_pairs(Path("student_resource/dataset"), s1, tg)
        df = df.join(truth.with_columns(label=pl.lit(1, pl.Int8)), on=["sidx", "tidx"], how="left")
        df = df.with_columns(pl.col("label").fill_null(0), fold_expr())
        df.write_parquet(work / "stage3_train.parquet")
        feats = feature_cols(df)
        tr = df.filter(pl.col("fold").is_in(TRAIN_FOLDS))
        va = df.filter(pl.col("fold") == TUNE_FOLD)
        dtr = xgb.QuantileDMatrix(X(tr, feats), tr["label"].to_numpy(), feature_names=feats)
        dva = xgb.QuantileDMatrix(X(va, feats), va["label"].to_numpy(), feature_names=feats, ref=dtr)
        m3 = xgb.train(PARAMS, dtr, args.rounds, evals=[(dva, "valid")], early_stopping_rounds=50,
                       verbose_eval=200)
        mdir.mkdir(parents=True, exist_ok=True)
        m3.save_model(str(mdir / "stage3.json"))
        log(f"stage3 trained ({m3.best_iteration} it)")
        ev = df.filter(pl.col("fold").is_in([TUNE_FOLD, HOLD_FOLD]))
        ev = ev.select("sidx", "tidx", "fold", "p2", "direct").with_columns(
            p3=pl.Series(predict(m3, X(ev, feats)).astype(np.float32)))
        truth = truth.with_columns(fold_expr())
        anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32)).with_columns(fold_expr())
        best = (0.0, 0.5)
        tune = ev.filter(pl.col("fold") == TUNE_FOLD)
        a3 = anchors.filter(pl.col("fold") == TUNE_FOLD)["sidx"]
        t3 = truth.filter(pl.col("fold") == TUNE_FOLD)
        for thr in np.arange(0.3, 0.95, 0.025):
            r = macro_f05(decide(tune, thr, "p3"), t3, a3)["macro_f05"]
            best = max(best, (r, float(thr)))
        thr = best[1]
        res = {"threshold": thr, "features": feats, "best_iteration": m3.best_iteration}
        for fold in (TUNE_FOLD, HOLD_FOLD):
            e = ev.filter(pl.col("fold") == fold)
            a = anchors.filter(pl.col("fold") == fold)["sidx"]
            tr_ = truth.filter(pl.col("fold") == fold)
            res[f"fold{fold}_stage3"] = macro_f05(decide(e, thr, "p3"), tr_, a)
            res[f"fold{fold}_stage2_ref"] = macro_f05(
                decide(e.filter(pl.col("p2").is_not_null()), meta_thr(work), "p2"), tr_, a)
            res[f"fold{fold}_oracle"] = macro_f05(e.join(tr_, on=["sidx", "tidx"]), tr_, a)
            res[f"fold{fold}_mean_candidates"] = len(e) / len(a)
        (mdir / "stage3_metrics.json").write_text(json.dumps(res, indent=2))
        for k, v in res.items():
            if isinstance(v, dict):
                print(f"{k:22s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
            elif k != "features":
                print(k, v)
        return

    # test
    from .predict import write_lists
    meta = json.loads((mdir / "stage3_metrics.json").read_text())
    st2 = pl.read_parquet(work / "test_preds.parquet")
    df = build("test", work, st2, work / "feats_test", log)
    m3 = xgb.Booster(model_file=str(mdir / "stage3.json"))
    m3.set_param({"device": "cuda"})
    preds = df.select("sidx", "tidx").with_columns(
        p3=pl.Series(predict(m3, X(df, meta["features"])).astype(np.float32)))
    preds.write_parquet(work / "test_preds_stage3.parquet")
    matches = decide(preds, meta["threshold"], "p3")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    norm = work / "norm"
    s1 = pl.read_parquet(norm / "test_source1.parquet", columns=["idx", "entity_id"])
    tg_ids = pl.concat([pl.read_parquet(norm / "test_source2.parquet", columns=["entity_id"]),
                        pl.read_parquet(norm / "test_source3.parquet", columns=["entity_id"])])["entity_id"]
    write_lists(out / "candidate_pairs.tsv", s1, preds.sort("sidx", "p3", descending=[False, True]),
                tg_ids, "candidate_entity_ids")
    write_lists(out / "matching_results.tsv", s1, matches.sort("sidx", "p3", descending=[False, True]),
                tg_ids, "matched_entity_ids")
    log(f"{len(matches):,} matches, {len(preds) / len(s1):.2f} candidates per source1")


def meta_thr(work: Path) -> float:
    for d in ("work/model_r2", "models/v2"):
        p = Path(d) / "metrics.json"
        if p.exists():
            return json.loads(p.read_text())["threshold"]
    return 0.675


if __name__ == "__main__":
    main()
