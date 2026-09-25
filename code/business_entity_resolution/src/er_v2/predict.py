"""Stage 5: score test candidates with the two-stage model and write outputs.

Candidate generation is a cascade: key blocking (top 64) followed by the
stage-1 model acting as a learned candidate ranker; pairs with p1 below
``prune`` are discarded. The surviving pairs (~5 per Source 1) are the final
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

from .train import context_features, select_matches, predict_frame
from .runtime import feature_parts, positive_int, file_digest


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
    if not 0 <= thr <= 1 or not 0 <= prune <= 1:
        raise ValueError("Threshold and prune must be between 0 and 1")
    f1, f2 = meta["stage1_features"], meta["stage2_features"]
    m1 = xgb.Booster(model_file=str(mdir / "stage1.json"))
    m2 = xgb.Booster(model_file=str(mdir / "stage2.json"))
    m1.set_param({"device": args.device, "nthread": args.threads})
    m2.set_param({"device": args.device, "nthread": args.threads})
    t = time.time()

    folder = Path(args.feats or work / 'feats_test')
    parts = feature_parts(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8')) if (folder / 'manifest.json').exists() else {}
    cache = manifest.get('stage1_cache')
    scores = []
    if cache:
        if cache['model_sha256'] != file_digest(mdir / 'stage1.json') or prune < cache['prune']:
            raise ValueError('Stage-1 cache model differs or requested prune is lower; regenerate test features')
        score_paths = sorted(folder.glob('scores_*.parquet'))
        actual = [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in score_paths]
        if actual != cache['parts']:
            raise ValueError('Stage-1 score cache is incomplete or changed')
        scores = [pl.read_parquet(p) for p in score_paths]
    else:
        for p in parts:
            df = pl.read_parquet(p)
            scores.append(df.select("sidx", "tidx").with_columns(
                p1=pl.Series(predict_frame(m1, df, f1, args.batch_rows))))
    scores = pl.concat(scores)
    ctx = context_features(scores)
    print(f"stage1 done {len(scores):,} pairs, {time.time() - t:.0f}s", flush=True)
    del scores
    ctx = ctx.filter(pl.col('p1') >= prune)

    preds = []
    for p in parts:
        df = pl.read_parquet(p).join(ctx, on=["sidx", "tidx"], how="inner", maintain_order="left")
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
    policy = meta.get('decision_policy', {'kind': 'threshold', 'threshold': thr})
    if args.threshold is not None:
        policy = {'kind': 'threshold', 'threshold': thr}
    matches = select_matches(preds, policy)
    write_lists(out / "candidate_pairs.tsv", s1, preds.sort("sidx", "p2", descending=[False, True]),
                tg_ids, "candidate_entity_ids")
    write_lists(out / "matching_results.tsv", s1, matches.sort("sidx", "p2", descending=[False, True]),
                tg_ids, "matched_entity_ids")
    n_nonempty = matches["sidx"].n_unique()
    print(f"threshold {thr:.3f}: {len(matches):,} matched pairs, {n_nonempty:,}/{len(s1):,} "
          f"source1 with >=1 match, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
