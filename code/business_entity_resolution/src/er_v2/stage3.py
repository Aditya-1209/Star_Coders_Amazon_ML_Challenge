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
from tempfile import TemporaryDirectory

import polars as pl
import numpy as np
import xgboost as xgb

from .block import make_keys
from .indexing import build_index
from .features import compute, pair_frame, record_frames, token_idf
from .decision import NO_MATCH_THRESHOLD, blend_scores, tune_blend, tune_country_thresholds, decide_country
from .graph import anchors_of, expand, hop_keep, support_features
from .metrics import by_country, leaderboard_estimate, macro_f05
from .run_block import load_split, split_countries, parquet_rows
from .runtime import BATCH_ROWS, DEFAULT_THREADS, feature_parts, positive_int
from .train import FEATURE_VERSION, PARAMS, decide, fit, fold_expr, predict_frame

TRAIN_FOLDS, TUNE_FOLD, HOLD_FOLD = [6, 7], 3, 4
BLOCK_COLS = ["bscore", "nkeys", "brank", "b_rel_s", "b_rel_t", "t_rank", "t_nc", "s_nc", "rescue"]


def build(split: str, work: Path, stage2: pl.DataFrame, feats_dir: Path, log, neural: bool = False,
          batch_rows: int = BATCH_ROWS, support_anchors: int = 100_000, workers: int = DEFAULT_THREADS,
          prune_hops: bool = True, block_chunk: int = 2_000) -> pl.DataFrame:
    emb = None
    if neural:
        from .neural import Embeddings
        emb = Embeddings(work, split)
    with TemporaryDirectory(prefix="graph-", dir=work) as tmp:
        paths = []
        for i, country in enumerate(split_countries(work, split)):
            log(f"building graph for {country}")
            frame = _build_country(split, work, stage2, feats_dir, log, batch_rows,
                                   support_anchors, workers, prune_hops, block_chunk, country, emb)
            path = Path(tmp) / f"{i}.parquet"
            frame.write_parquet(path)
            paths.append(path)
            del frame
            gc.collect()
        return pl.scan_parquet(paths).collect(engine="streaming")


def _build_country(split, work, stage2, feats_dir, log, batch_rows, support_anchors,
                   workers, prune_hops, block_chunk, country, emb=None):
    """Return stage-3 feature rows for direct (pruned) + two-hop candidates."""
    s1, tg = load_split(work, split, country)
    s1 = s1.with_columns(pl.col("idx").cast(pl.UInt32))
    tg = tg.with_columns(pl.col("idx").cast(pl.UInt32))
    n_s2 = parquet_rows(work / "norm" / f"{split}_source2.parquet")
    stage2 = stage2.join(s1.select(sidx="idx"), on="sidx", how="semi")
    anchors = anchors_of(stage2)
    log(f"anchors {len(anchors):,}")

    if anchors.is_empty():
        from .graph import HOP_SCHEMA
        hop = pl.DataFrame(schema=HOP_SCHEMA)
    else:
        n_targets = sum(parquet_rows(work / "norm" / f"{split}_source{i}.parquet") for i in (2, 3))
        tindex = build_index(tg.select("idx", "country", "core_n", "addr_n"), n_targets, work)
        query_records = tg.join(anchors.select(idx="a").unique(), on="idx", how="semi")
        chunks = []
        for query in query_records.iter_slices(50_000):
            local_anchors = anchors.join(query.select(a="idx"), on="a", how="semi")
            chunks.append(expand(local_anchors, make_keys(query), tindex, block_chunk))
        hop = pl.concat(chunks).group_by("sidx", "tidx").agg(
            pl.col("hop_score").max(), pl.col("hop_rank").min(),
            pl.col("hop_n").sum(), pl.col("hop_pa").max())
        from .graph import HOP_SCHEMA
        hop = hop.cast(HOP_SCHEMA)
        del query_records, chunks, tindex
    gc.collect()
    log(f"two-hop pairs {len(hop):,}")

    direct = stage2.select("sidx", "tidx", "p1", "p2").with_columns(direct=pl.lit(1, pl.Int8))
    pairs = direct.join(hop, on=["sidx", "tidx"], how="full", coalesce=True)
    pairs = pairs.with_columns(pl.col("direct").fill_null(0))
    if emb is not None:
        from .neural import neural_features
        pairs = pairs.join(neural_features(pairs, emb), on=["sidx", "tidx"], how="left")
        log("attached fine-tuned encoder features to stage-3 candidates")
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
    idf = token_idf(tg) if "wn_jacc" in old.columns else None
    new_parts = []
    for batch in new_pairs.iter_slices(batch_rows):
        batch = batch.with_columns(*[pl.lit(None, pl.Float32).alias(c) for c in BLOCK_COLS])
        new_parts.append(compute(pair_frame(batch, left, right, n_s2), workers=workers, idf=idf,
                                 enhanced="token_align_min" in old.columns).select(old.columns).cast(old.schema))
    newf = pl.concat(new_parts) if new_parts else old.head(0)
    log(f"computed features for {len(newf):,} new pairs")
    base = pl.concat([old, newf.select(old.columns)], how="vertical_relaxed")
    del old, newf, new_parts, new_pairs, left, s1, tg, idf
    gc.collect()

    txt = right.select("idx", "core_r", "addr_r", "cc_r")
    pairs = pairs.sort("sidx", "tidx")
    source_ids = pairs["sidx"].to_numpy()
    ids = pairs["sidx"].unique().sort()
    sup = []
    for i in range(0, len(ids), support_anchors):
        lo = int(np.searchsorted(source_ids, ids[i], side="left"))
        hi = int(np.searchsorted(source_ids, ids[min(i + support_anchors, len(ids)) - 1], side="right"))
        chunk = pairs.slice(lo, hi - lo)
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


def feature_cols(df: pl.DataFrame, profile: str = "enhanced") -> list[str]:
    from .features import SPLIT_DEPENDENT
    from .features import feature_names
    return feature_names(df, profile)


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
    ap.add_argument("--block-chunk", type=positive_int, default=2_000)
    ap.add_argument("--feature-profile", choices=["baseline", "enhanced"], default="baseline")
    ap.add_argument("--compare-baseline", action="store_true")
    ap.add_argument("--country-thresholds", action="store_true")
    ap.add_argument("--hist-cache-nodes", type=positive_int, default=2048)
    ap.add_argument("--exclude-country", nargs="*", default=[],
                    help="leave these countries out of stage-3 training (unseen-country proxy)")
    args = ap.parse_args()
    tag = "".join(f"_no{c}" for c in args.exclude_country)
    work, mdir = Path(args.work), Path(args.model_dir)
    t = time.time()
    log = lambda m: print(f"[{time.time() - t:6.0f}s] {m}", flush=True)
    build_options = {"batch_rows": args.batch_rows, "support_anchors": args.support_anchors,
                     "workers": args.threads, "block_chunk": args.block_chunk}
    stage2_meta = json.loads((mdir / "metrics.json").read_text(encoding="utf-8"))
    if stage2_meta.get("ghost_frac", 0) > 0:
        raise ValueError("Stage 3 does not yet support ghost training; use a non-ghost stage-2 model")
    if stage2_meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Regenerate accuracy features and retrain er_v2.train before stage 3")

    if args.split == "train":
        from .run_features import ground_truth_pairs
        st2 = pl.read_parquet(work / f"stage2_train{tag}.parquet").with_columns(fold_expr())
        keep = TRAIN_FOLDS + [TUNE_FOLD, HOLD_FOLD]
        st2 = st2.filter(pl.col("fold").is_in(keep)).drop("label", "fold")
        df = build("train", work, st2, work / "feats_train", log, neural=bool(stage2_meta.get("neural")),
                   prune_hops=False, **build_options)
        s1, tg = load_split(work, "train", columns=["idx", "entity_id", "country"])
        truth = ground_truth_pairs(Path(args.dataset), s1, tg)
        df = df.join(truth.with_columns(label=pl.lit(1, pl.Int8)), on=["sidx", "tidx"], how="left")
        df = df.with_columns(pl.col("label").fill_null(0), fold_expr())
        df.write_parquet(work / f"stage3_train{tag}.parquet")
        feats = feature_cols(df, args.feature_profile)
        country = s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country")
        seen = ~pl.col("sidx").is_in(
            country.filter(pl.col("country").is_in(args.exclude_country))["sidx"])
        tr = df.filter(pl.col("fold").is_in(TRAIN_FOLDS) & seen)
        va = df.filter((pl.col("fold") == TUNE_FOLD) & seen)
        params = {**PARAMS, "device": args.device, "nthread": args.threads,
                  "max_cached_hist_node": args.hist_cache_nodes}
        m3 = fit(tr, feats, va, args.rounds, params)
        trials = []
        if args.compare_baseline and args.feature_profile == "enhanced":
            baseline_feats = feature_cols(df, "baseline")
            baseline = fit(tr, baseline_feats, va, args.rounds, params)
            baseline.save_model(str(mdir / "stage3_baseline.json"))
            m3.save_model(str(mdir / "stage3_enhanced.json"))
            selection_data = va.filter(hop_keep())
            selection_anchors = country.filter((fold_expr() == TUNE_FOLD) & seen)["sidx"]
            selection_truth = truth.filter(fold_expr() == TUNE_FOLD)
            for name, model, columns in (("baseline", baseline, baseline_feats), ("enhanced", m3, feats)):
                predictions = selection_data.select("sidx", "tidx", "p2").with_columns(
                    p3=pl.Series(predict_frame(model, selection_data, columns, args.batch_rows)))
                trial = tune_blend(predictions, selection_truth, selection_anchors)
                value = max(t["macro_f05"] for t in trial["tuning_trials"])
                trials.append({"profile": name, "fold3_macro_f05": value, **trial})
                log(f"stage3 feature ablation: {name} F0.5={value:.6f}")
            if trials[0]["fold3_macro_f05"] >= trials[1]["fold3_macro_f05"]:
                m3, feats = baseline, baseline_feats
            del baseline, selection_data, selection_anchors, selection_truth, predictions
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
        res = {**selection, "stage3_threshold": pure_threshold, "feature_trials": trials,
               "feature_profile": "enhanced" if "token_align_min" in feats else "baseline",
               "features": feats, "best_iteration": m3.best_iteration,
               "feature_version": FEATURE_VERSION, "selection_fold": TUNE_FOLD,
               "runtime": {"device": args.device, "threads": args.threads,
               "batch_rows": args.batch_rows, "support_anchors": args.support_anchors}}
        ev = blend_scores(ev, selection["stage3_weight"])
        country_thresholds = tune_country_thresholds(
            ev.filter(pl.col("fold") == TUNE_FOLD), t3,
            anchors.filter((pl.col("fold") == TUNE_FOLD) & ~pl.col("country").is_in(args.exclude_country)),
            thr, "score") if args.country_thresholds else {}
        res["country_thresholds"] = country_thresholds
        ev.write_parquet(work / "eval_preds_stage3.parquet")
        for fold in (TUNE_FOLD, HOLD_FOLD):
            e = ev.filter(pl.col("fold") == fold)
            a = anchors.filter(pl.col("fold") == fold)["sidx"]
            tr_ = truth.filter(pl.col("fold") == fold)
            res[f"fold{fold}_selected"] = macro_f05(decide_country(e, thr, anchors, country_thresholds, "score"), tr_, a)
            res[f"fold{fold}_global_reference"] = macro_f05(decide(e, thr, "score"), tr_, a)
            res[f"fold{fold}_stage3"] = macro_f05(decide(e, pure_threshold, "p3"), tr_, a)
            res[f"fold{fold}_stage2_ref"] = macro_f05(
                decide_country(e.filter(pl.col("p2").is_not_null()), meta_thr(mdir), anchors,
                               stage2_meta.get("country_thresholds", {}), "p2"), tr_, a)
            res[f"fold{fold}_oracle"] = macro_f05(e.join(tr_, on=["sidx", "tidx"]), tr_, a)
            res[f"fold{fold}_mean_candidates"] = len(e) / len(a)
        hold = ev.filter(pl.col("fold") == HOLD_FOLD)
        per = by_country(decide_country(hold, thr, anchors, country_thresholds, "score"), truth.filter(pl.col("fold") == HOLD_FOLD),
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
    df = build("test", work, st2, work / "feats_test", log, neural=bool(stage2_meta.get("neural")),
               **build_options)
    m3 = xgb.Booster(model_file=str(mdir / "stage3.json"))
    m3.set_param({"device": args.device, "nthread": args.threads})
    preds = df.select("sidx", "tidx", "p2").with_columns(
        p3=pl.Series(predict_frame(m3, df, meta["features"], args.batch_rows)))
    preds = blend_scores(preds, meta["stage3_weight"])
    preds.write_parquet(work / "test_preds_stage3.parquet")
    df.write_parquet(work / "stage3_test.parquet")  # reused by selftrain.py
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    norm = work / "norm"
    s1 = pl.read_parquet(norm / "test_source1.parquet", columns=["idx", "entity_id", "country"])
    matches = decide_country(preds, meta["threshold"], s1.select(sidx="idx", country="country"),
                             meta.get("country_thresholds", {}), "score")
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
