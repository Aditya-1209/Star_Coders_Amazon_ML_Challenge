"""Stage 5: score test candidates with the two-stage model and write outputs.

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
import numpy as np
import polars as pl

from .train import X, context_features, decide, predict


def write_lists(path: Path, s1: pl.DataFrame, pairs: pl.DataFrame, tg_ids: pl.Series, col: str) -> None:
    ids = pairs.with_columns(tid=tg_ids.gather(pairs["tidx"]))
    lists = ids.group_by("sidx").agg(pl.col("tid").unique(maintain_order=True).str.join(","))
    out = s1.select(pl.col("idx").cast(pl.UInt32).alias("sidx"), source1_entity_id="entity_id")
    out = out.join(lists, on="sidx", how="left", maintain_order="left").select(
        "source1_entity_id", pl.col("tid").fill_null("").alias(col))
    out.write_csv(path, separator="\t", quote_style="never")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--model-dir", default="models/v2")
    ap.add_argument("--output", default="output")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    work, mdir, out = Path(args.work), Path(args.model_dir), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((mdir / "metrics.json").read_text())
    thr = args.threshold if args.threshold is not None else meta["threshold"]
    f1, f2 = meta["stage1_features"], meta["stage2_features"]
    m1 = xgb.Booster(model_file=str(mdir / "stage1.json"))
    m2 = xgb.Booster(model_file=str(mdir / "stage2.json"))
    m1.set_param({"device": "cuda"})
    m2.set_param({"device": "cuda"})
    t = time.time()

    parts = sorted((work / "feats_test").glob("part_*.parquet"))
    scores = []
    for p in parts:
        df = pl.read_parquet(p)
        scores.append(df.select("sidx", "tidx").with_columns(
            p1=pl.Series(predict(m1, X(df, f1)).astype(np.float32))))
    scores = pl.concat(scores)
    ctx = context_features(scores)
    print(f"stage1 done {len(scores):,} pairs, {time.time() - t:.0f}s", flush=True)
    del scores

    preds = []
    for p in parts:
        df = pl.read_parquet(p).join(ctx, on=["sidx", "tidx"], how="left", maintain_order="left")
        preds.append(df.select("sidx", "tidx").with_columns(
            p2=pl.Series(predict(m2, X(df, f2)).astype(np.float32))))
    preds = pl.concat(preds)
    preds.write_parquet(work / "test_preds.parquet")
    print(f"stage2 done, {time.time() - t:.0f}s", flush=True)

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
