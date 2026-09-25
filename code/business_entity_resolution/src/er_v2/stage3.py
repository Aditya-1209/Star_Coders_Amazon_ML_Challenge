"""Stage 3 driver: two-hop expansion + support features + final model.

train mode : uses work/stage2_train.parquet (scores on folds unseen by stage-2 fitting),
             trains on folds 6/7, tunes the threshold on fold 3, reports fold 4.
test mode  : uses work/test_preds.parquet, writes the final TSVs.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import polars as pl
import xgboost as xgb

from .block import build_target_index, make_keys
from .features import compute, pair_frame, record_frames, token_idf
from .decision import NO_MATCH_THRESHOLD, blend_scores, tune_blend
from .graph import anchors_of, expand, hop_keep, support_features
from .metrics import by_country, leaderboard_estimate, macro_f05
from .run_block import load_split
from .runtime import BATCH_ROWS, DEFAULT_THREADS, feature_parts, positive_int
from .train import FEATURE_VERSION, PARAMS, decide, fit, fold_expr, predict_frame

TRAIN_FOLDS, TUNE_FOLD, HOLD_FOLD = [6, 7], 3, 4
BLOCK_COLS = ["bscore", "nkeys", "brank", "b_rel_s", "b_rel_t", "t_rank", "t_nc", "s_nc", "rescue"]


def build(split: str, work: Path, stage2: pl.DataFrame, feats_dir: Path, log,
          batch_rows: int = BATCH_ROWS, support_anchors: int = 100_000, workers: int = DEFAULT_THREADS,
          prune_hops: bool = True) -> pl.DataFrame:
    """Return stage-3 feature rows for direct (pruned) + two-hop candidates."""
    s1, tg = load_split(work, split)
    s1 = s1.with_columns(pl.col("idx").cast(pl.UInt32))
    tg = tg.with_columns(pl.col("idx").cast(pl.UInt32))
    n_s2 = len(pl.read_parquet(work / "norm" / f"{split}_source2.parquet", columns=["idx"]))
    anchors = anchors_of(stage2)
    log(f"anchors {len(anchors):,}")

    if anchors.is_empty():
        from .graph import HOP_SCHEMA
        hop = pl.DataFrame(schema=HOP_SCHEMA)
    else:
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
    for p in feature_parts(feats_dir):
        old.append(pl.scan_parquet(p).join(keyset.lazy(), on=["sidx", "tidx"], how="semi")
                   .collect(engine="streaming"))
    old = pl.concat(old)
    if "label" in old.columns:
        old = old.drop("label")
    new_pairs = keyset.join(old.select("sidx", "tidx"), on=["sidx", "tidx"], how="anti")
    left, right = record_frames(s1, tg)
    idf = token_idf(tg)
    new_parts = []
    for batch in new_pairs.iter_slices(batch_rows):
        batch = batch.with_columns(*[pl.lit(None, pl.Float32).alias(c) for c in BLOCK_COLS])
        new_parts.append(compute(pair_frame(batch, left, right, n_s2), workers=workers, idf=idf).select(old.columns))
    newf = pl.concat(new_parts) if new_parts else old.head(0)
    log(f"computed features for {len(newf):,} new pairs")
    base = pl.concat([old, newf.select(old.columns)], how="vertical_relaxed")
    del old, newf, new_parts, new_pairs, left, s1, tg, idf
    gc.collect()

    txt = right.select("idx", "core_r", "addr_r", "cc_r")
    ids = pairs["sidx"].unique().sort()
    sup = []
    for i in range(0, len(ids), support_anchors):
        chunk = pairs.join(pl.DataFrame({"sidx": ids[i:i + support_anchors]}), on="sidx", how="semi")
        sup.append(support_features(chunk, anchors, txt, workers=workers))
    sup = pl.concat(sup) if sup else support_features(pairs, anchors, txt, workers=workers)
    del right
    gc.collect()
    out = base.join(pairs, on=["sidx", "tidx"], how="left").join(sup, on=["sidx", "tidx"], how="left")
    if prune_hops:
        before = len(out)
        out = out.filter(hop_keep())
        log(f"two-hop support filter: {before:,} -> {len(out):,} candidates")
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
    ap.add_argument("--dataset", default="student_resource/dataset")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=positive_int, default=DEFAULT_THREADS)
    ap.add_argument("--batch-rows", type=positive_int, default=BATCH_ROWS)
    ap.add_argument("--support-anchors", type=positive_int, default=100_000)
    ap.add_argument("--rounds", type=positive_int, default=2000)
    ap.add_argument("--exclude-country", nargs="*", default=[],
                    help="leave these countries out of stage-3 training (unseen-country proxy)")
    args = ap.parse_args()
    tag = "".join(f"_no{c}" for c in args.exclude_country)
    work, mdir = Path(args.work), Path(args.model_dir)
    t = time.time()
    log = lambda m: print(f"[{time.time() - t:6.0f}s] {m}", flush=True)
    build_options = {"batch_rows": args.batch_rows, "support_anchors": args.support_anchors,
                     "workers": args.threads}
    stage2_meta = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
    if stage2_meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Regenerate accuracy features and retrain er_v2.train before stage 3")

    if args.split == "train":
        from .run_features import ground_truth_pairs
        st2 = pl.read_parquet(work / f"stage2_train{tag}.parquet").with_columns(fold_expr())
        keep = TRAIN_FOLDS + [TUNE_FOLD, HOLD_FOLD]
        st2 = st2.filter(pl.col("fold").is_in(keep)).drop("label", "fold")
        df = build("train", work, st2, work / "feats_train", log, prune_hops=False, **build_options)
        s1, tg = load_split(work, "train")
        truth = ground_truth_pairs(Path(args.dataset), s1, tg)
        df = df.join(truth.with_columns(label=pl.lit(1, pl.Int8)), on=["sidx", "tidx"], how="left")
        df = df.with_columns(pl.col("label").fill_null(0), fold_expr())
        df.write_parquet(work / f"stage3_train{tag}.parquet")
        feats = feature_cols(df)
        country = s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country")
        seen = ~pl.col("sidx").is_in(
            country.filter(pl.col("country").is_in(args.exclude_country))["sidx"])
        tr = df.filter(pl.col("fold").is_in(TRAIN_FOLDS) & seen)
        va = df.filter(pl.col("fold") == TUNE_FOLD)
        m3 = fit(tr, feats, va, args.rounds, {**PARAMS, "device": args.device, "nthread": args.threads})
        del tr, va
        mdir.mkdir(parents=True, exist_ok=True)
        m3.save_model(str(mdir / "stage3.json"))
        log(f"stage3 trained ({m3.best_iteration} it)")
        ev = df.filter(pl.col("fold").is_in([TUNE_FOLD, HOLD_FOLD]))
        # Evaluate exactly the candidate set the test run keeps (same hop filter).
        ev = ev.filter(hop_keep())
        ev = ev.select("sidx", "tidx", "fold", "p2", "direct").with_columns(
            p3=pl.Series(predict_frame(m3, ev, feats, args.batch_rows)))
        truth = truth.with_columns(fold_expr())
        anchors = country.with_columns(fold_expr())
        del df, s1, tg
        tune = ev.filter(pl.col("fold") == TUNE_FOLD)
        a3 = anchors.filter((pl.col("fold") == TUNE_FOLD) & ~pl.col("country").is_in(args.exclude_country))["sidx"]
        t3 = truth.filter(pl.col("fold") == TUNE_FOLD)
        selection = tune_blend(tune, t3, a3)
        thr = selection["threshold"]
        pure_threshold = selection["tuning_trials"][-1]["threshold"]
        res = {**selection, "stage3_threshold": pure_threshold,
               "features": feats, "best_iteration": m3.best_iteration,
               "feature_version": FEATURE_VERSION, "selection_fold": TUNE_FOLD,
               "runtime": {"device": args.device, "threads": args.threads,
               "batch_rows": args.batch_rows, "support_anchors": args.support_anchors}}
        ev = blend_scores(ev, selection["stage3_weight"])
        ev.write_parquet(work / "eval_preds_stage3.parquet")
        for fold in (TUNE_FOLD, HOLD_FOLD):
            e = ev.filter(pl.col("fold") == fold)
            a = anchors.filter(pl.col("fold") == fold)["sidx"]
            tr_ = truth.filter(pl.col("fold") == fold)
            res[f"fold{fold}_selected"] = macro_f05(decide(e, thr, "score"), tr_, a)
            res[f"fold{fold}_stage3"] = macro_f05(decide(e, pure_threshold, "p3"), tr_, a)
            res[f"fold{fold}_stage2_ref"] = macro_f05(
                decide(e.filter(pl.col("p2").is_not_null()), meta_thr(mdir), "p2"), tr_, a)
            res[f"fold{fold}_oracle"] = macro_f05(e.join(tr_, on=["sidx", "tidx"]), tr_, a)
            res[f"fold{fold}_mean_candidates"] = len(e) / len(a)
        hold = ev.filter(pl.col("fold") == HOLD_FOLD)
        per = by_country(decide(hold, thr, "score"), truth.filter(pl.col("fold") == HOLD_FOLD),
                         anchors.filter(pl.col("fold") == HOLD_FOLD))
        res["fold4_by_country"] = per
        res["leaderboard_estimate"] = leaderboard_estimate(per)
        res["excluded_countries"] = args.exclude_country
        for c, v in per.items():
            print(f"fold4 {c:8s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
        print(f"leaderboard estimate {res['leaderboard_estimate']['estimate']:.4f}")
        (mdir / "stage3_metrics.json").write_text(json.dumps(res, indent=2))
        for k, v in res.items():
            if k in ("fold4_by_country", "leaderboard_estimate", "runtime", "tuning_trials"):
                continue
            if isinstance(v, dict) and "macro_f05" in v:
                print(f"{k:22s} F0.5={v['macro_f05']:.4f} P={v['pair_precision']:.4f} R={v['pair_recall']:.4f}")
            elif k != "features":
                print(k, v)
        return

    # test
    from .predict import write_lists
    meta = json.loads((mdir / "stage3_metrics.json").read_text())
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Stage-3 accuracy features changed; retrain er_v2.stage3 --split train first")
    st2 = pl.read_parquet(work / "test_preds.parquet")
    df = build("test", work, st2, work / "feats_test", log, **build_options)
    m3 = xgb.Booster(model_file=str(mdir / "stage3.json"))
    m3.set_param({"device": args.device, "nthread": args.threads})
    preds = df.select("sidx", "tidx", "p2").with_columns(
        p3=pl.Series(predict_frame(m3, df, meta["features"], args.batch_rows)))
    preds = blend_scores(preds, meta["stage3_weight"])
    preds.write_parquet(work / "test_preds_stage3.parquet")
    df.write_parquet(work / "stage3_test.parquet")  # reused by selftrain.py
    matches = decide(preds, meta["threshold"], "score")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    norm = work / "norm"
    s1 = pl.read_parquet(norm / "test_source1.parquet", columns=["idx", "entity_id"])
    tg_ids = pl.concat([pl.read_parquet(norm / "test_source2.parquet", columns=["entity_id"]),
                        pl.read_parquet(norm / "test_source3.parquet", columns=["entity_id"])])["entity_id"]
    write_lists(out / "candidate_pairs.tsv", s1, preds.sort("sidx", "score", descending=[False, True]),
                tg_ids, "candidate_entity_ids")
    write_lists(out / "matching_results.tsv", s1, matches.sort("sidx", "score", descending=[False, True]),
                tg_ids, "matched_entity_ids")
    log(f"{len(matches):,} matches, {len(preds) / len(s1):.2f} candidates per source1")


def meta_thr(model_dir: Path) -> float:
    """Use the same model directory as the run; never guess a threshold."""
    value = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["threshold"]
    if not 0 <= value <= NO_MATCH_THRESHOLD:
        raise ValueError(f"Invalid stage-2 threshold in {model_dir / 'metrics.json'}")
    return float(value)


if __name__ == "__main__":
    main()
