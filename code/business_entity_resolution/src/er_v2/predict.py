"""Stage 5: score test candidates with the two-stage model and write outputs.

Candidate generation is a cascade: key blocking (combined and channel lists) followed by the
stage-1 ensemble acting as a learned candidate ranker; pairs with p1 below
``prune`` are discarded. The surviving pairs are the stage-2
candidate set: exactly the pairs the stage-2 matcher runs inference over.

Writes, in Source 1 file order and with one row per Source 1 record:
  output/matching_results.tsv   final matches
  output/candidate_pairs.tsv    the exact candidate set scored by the model
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import xgboost as xgb
import polars as pl

from .train import context_features, decide, predict_frame, predict_ensemble
from .decision import NO_MATCH_THRESHOLD
from .runtime import feature_parts, positive_int


def write_lists(path: Path, s1: pl.DataFrame, pairs: pl.DataFrame, tg_ids: pl.Series, col: str) -> None:
    ids = pairs.with_columns(tid=tg_ids.gather(pairs["tidx"]))
    lists = ids.group_by("sidx").agg(pl.col("tid").unique().sort().str.join(","))
    out = s1.select(pl.col("idx").cast(pl.UInt32).alias("sidx"), source1_entity_id="entity_id")
    out = out.join(lists, on="sidx", how="left", maintain_order="left").select(
        "source1_entity_id", pl.col("tid").fill_null("").alias(col))
    # A stopped write must not leave a plausible-looking partial final TSV.
    temporary = path.with_name(path.name + ".tmp")
    out.write_csv(temporary, separator="\t", quote_style="never")
    temporary.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--model-dir", default="models/v2")
    ap.add_argument("--output", default="output")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--prune", type=float, default=None, help="stage-1 score floor for candidates")
    ap.add_argument("--feats", default=None, help="test feature folder (default work/feats_test)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=positive_int, default=12)
    ap.add_argument("--batch-rows", type=positive_int, default=250_000)
    args = ap.parse_args()
    work, mdir, out = Path(args.work), Path(args.model_dir), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((mdir / "metrics.json").read_text())
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    prune = args.prune if args.prune is not None else meta.get("prune", 0.001)
    if not 0 <= thr <= NO_MATCH_THRESHOLD or not 0 <= prune <= 1:
        raise ValueError("Invalid saved threshold or prune value")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise ValueError("An explicit --threshold must be between 0 and 1")
    f1, f2 = meta["stage1_features"], meta["stage2_features"]
    rankers = [xgb.Booster(model_file=str(mdir / name))
               for name in meta.get("stage1_models", ["stage1.json"])]
    m2 = xgb.Booster(model_file=str(mdir / "stage2.json"))
    for model in rankers:
        model.set_param({"device": args.device, "nthread": args.threads})
    m2.set_param({"device": args.device, "nthread": args.threads})
    t = time.time()

    parts = feature_parts(Path(args.feats or work / "feats_test"))
    scores = []
    for p in parts:
        df = pl.read_parquet(p)
        scores.append(df.select("sidx", "tidx").with_columns(
            p1=pl.Series(predict_ensemble(rankers, df, f1, args.batch_rows))))
    scores = pl.concat(scores)
    ctx = context_features(scores)
    print(f"stage1 done {len(scores):,} pairs, {time.time() - t:.0f}s", flush=True)
    del scores

    preds = []
    for p in parts:
        df = pl.read_parquet(p).join(ctx, on=["sidx", "tidx"], how="left", maintain_order="left")
        df = df.filter(pl.col("p1") >= prune)
        preds.append(df.select("sidx", "tidx", "p1").with_columns(
            p2=pl.Series(predict_frame(m2, df, f2, args.batch_rows))))
    preds = pl.concat(preds)
    preds.write_parquet(work / "test_preds.parquet")
    print(f"stage2 done on {len(preds):,} pruned candidates (p1>={prune}), {time.time() - t:.0f}s", flush=True)

    norm = work / "norm"
    s1 = pl.read_parquet(norm / "test_source1.parquet", columns=["idx", "entity_id"])
    tg_ids = pl.concat([
        pl.read_parquet(norm / "test_source2.parquet", columns=["entity_id"]),
        pl.read_parquet(norm / "test_source3.parquet", columns=["entity_id"]),
    ])["entity_id"]
    matches = decide(preds, thr)
    write_lists(out / "candidate_pairs.tsv", s1, preds.sort("sidx", "p2", descending=[False, True]),
                tg_ids, "candidate_entity_ids")
    write_lists(out / "matching_results.tsv", s1, matches.sort("sidx", "p2", descending=[False, True]),
                tg_ids, "matched_entity_ids")
    n_nonempty = matches["sidx"].n_unique()
    print(f"threshold {thr:.3f}: {len(matches):,} matched pairs, {n_nonempty:,}/{len(s1):,} "
          f"source1 with >=1 match, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
