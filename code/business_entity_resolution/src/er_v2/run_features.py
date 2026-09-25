"""Compute pair features for a split's candidates, in shards."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from .features import add_block_context, compute, pair_frame, record_frames
from .run_block import load_split
from .runtime import positive_int, file_digest


def ground_truth_pairs(dataset: Path, s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    gt = pl.read_csv(dataset / "train" / "train_ground_truth.tsv", separator="\t",
                     infer_schema=False).with_columns(pl.col("matched_entity_ids").str.split(","))
    gt = gt.explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "")
    gt = gt.join(s1.select("entity_id", sidx="idx"), left_on="source1_entity_id", right_on="entity_id")
    gt = gt.join(tg.select("entity_id", tidx="idx"), left_on="matched_entity_ids", right_on="entity_id")
    return gt.select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--dataset", default="student_resource/dataset")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--shard-pairs", type=positive_int, default=2_000_000)
    ap.add_argument("--workers", type=positive_int, default=12)
    ap.add_argument('--stage1-model-dir', type=Path, help='Test only: score stage 1 now and store only survivors')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = ap.parse_args()
    model = meta = None
    if args.stage1_model_dir:
        if args.split != 'test':
            ap.error('--stage1-model-dir is only valid for test (training needs out-of-fold scores)')
        import xgboost as xgb
        from .train import predict_frame
        meta = json.loads((args.stage1_model_dir / 'metrics.json').read_text(encoding='utf-8'))
        model = xgb.Booster(model_file=str(args.stage1_model_dir / 'stage1.json'))
        model.set_param({'device': args.device, 'nthread': args.workers})
    work = Path(args.work)
    out = work / f"feats_{args.split}"
    if out.exists():
        raise FileExistsError(f"Refusing to mix new and old feature shards in {out}; use a fresh --work directory")
    out.mkdir(parents=True)
    (out / "_INCOMPLETE").write_text("Feature generation has not completed.\n", encoding="utf-8")
    t = time.time()
    s1, tg = load_split(work, args.split, columns=['idx', 'entity_id', 'business_name', 'name_n', 'core_n', 'addr_n'])
    s1 = s1.with_columns(pl.col("idx").cast(pl.UInt32))
    tg = tg.with_columns(pl.col("idx").cast(pl.UInt32))
    n_s2 = pl.scan_parquet(work / "norm" / f"{args.split}_source2.parquet").select(pl.len()).collect().item()
    cands = add_block_context(pl.read_parquet(work / f"cands_{args.split}.parquet"))
    if args.split == "train":
        gt = ground_truth_pairs(Path(args.dataset), s1, tg).with_columns(label=pl.lit(1, pl.Int8))
        cands = cands.join(gt, on=["sidx", "tidx"], how="left").with_columns(pl.col("label").fill_null(0))
    cands = cands.sort("sidx", "brank")
    left, right = record_frames(s1, tg)
    del s1, tg
    sid = cands["sidx"]
    n = len(cands)
    start, shard = 0, 0
    while start < n or shard == 0:
        stop = min(start + args.shard_pairs, n)
        while stop < n and sid[stop] == sid[stop - 1]:
            stop += 1
        part = cands.slice(start, stop - start)
        feats = compute(pair_frame(part, left, right, n_s2), workers=args.workers)
        if "label" in part.columns:
            feats = feats.with_columns(part["label"])
        if model is not None:
            scores = feats.select('sidx', 'tidx').with_columns(
                p1=pl.Series(predict_frame(model, feats, meta['stage1_features'])))
            # All scores are needed for identical global competition context.
            scores.write_parquet(out / f'scores_{shard:03d}.parquet')
            feats = feats.filter(scores['p1'] >= meta.get('prune', 0.001))
        destination = out / f"part_{shard:03d}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        feats.write_parquet(temporary)
        temporary.replace(destination)
        print(f"shard {shard}: {stop:,}/{n:,} pairs, {time.time() - t:.0f}s", flush=True)
        start, shard = stop, shard + 1
    manifest = {"version": 1, "split": args.split, "pairs": n,
                "parts": [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                          for p in sorted(out.glob("part_*.parquet"))]}
    if model is not None:
        manifest['stage1_cache'] = {
            'model_sha256': file_digest(args.stage1_model_dir / 'stage1.json'),
            'prune': meta.get('prune', 0.001),
            'parts': [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                      for p in sorted(out.glob('scores_*.parquet'))],
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "_INCOMPLETE").unlink()


if __name__ == "__main__":
    main()
