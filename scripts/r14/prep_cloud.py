"""Build a compact upload bundle for training a second cross-encoder (CE-B) on a cloud GPU.

CE-B differs from the local CE-A by its business sample (hash seed) and size, so the two
models make partly independent errors. Only token rows that CE-B training, validation
and uncertain-pair scoring touch are shipped; the VM rebuilds full-size arrays (zeros
elsewhere) so er_v2.r10_ce.PairTokens works unchanged.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from er_v2.folds import fold_expr
from er_v2.r10_ce import TRAIN_FOLDS
from er_v2.run_block import load_split
from er_v2.run_features import ground_truth_pairs

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--dataset", type=Path, required=True)
ap.add_argument("--pairs", type=Path, required=True, help="dir with pairs_{train,test}_uncertain.parquet")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--businesses", type=int, default=500_000)
ap.add_argument("--seed", type=int, default=2027)
args = ap.parse_args()
out = args.out
(out / "work/ce_tokens").mkdir(parents=True, exist_ok=True)


def sample(folds, limit, seed, negatives=4):
    """er_v2.r10_ce.samples with a different business-sampling seed."""
    s1, tg = load_split(args.work, "train", columns=["idx", "entity_id", "country"])
    anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32)).filter(fold_expr().is_in(folds))
    anchors = anchors.with_columns(key=pl.col("sidx").hash(seed=seed)).sort("key").head(limit).drop("key")
    truth = ground_truth_pairs(args.dataset, s1, tg).join(anchors, on="sidx", how="semi")
    del s1, tg
    cands = (pl.scan_parquet(args.work / "cands_train.parquet").join(anchors.lazy(), on="sidx", how="semi")
             .select("sidx", "tidx", "brank").collect(engine="streaming"))
    neg = cands.join(truth, on=["sidx", "tidx"], how="anti")
    lexical = neg.sort("sidx", "brank", "tidx").group_by("sidx", maintain_order=True).head(max(1, negatives // 2))
    ann = (pl.scan_parquet(args.work / "ncands_train.parquet").join(anchors.lazy(), on="sidx", how="semi")
           .select("sidx", "tidx", "nrank").collect(engine="streaming"))
    ann = ann.join(truth, on=["sidx", "tidx"], how="anti").sort("sidx", "nrank", "tidx") \
             .group_by("sidx", maintain_order=True).head(max(1, negatives // 2))
    rnd = neg.with_columns(key=pl.struct("sidx", "tidx").hash(seed=seed + 78)).sort("sidx", "key") \
             .group_by("sidx", maintain_order=True).head(1)
    neg = pl.concat([x.select("sidx", "tidx") for x in (lexical, ann, rnd)]).unique().with_columns(label=pl.lit(0, pl.Int8))
    res = pl.concat([truth.with_columns(label=pl.lit(1, pl.Int8)), neg]).sort("sidx", "tidx")
    return res.with_columns(w=(len(res) / res["sidx"].n_unique() / pl.len().over("sidx")).cast(pl.Float32))


tr = sample(TRAIN_FOLDS, args.businesses, args.seed)
tr.write_parquet(out / "work/ce_train_sample.parquet")
va = pl.read_parquet(args.work / "ce_valid_sample.parquet")
va.write_parquet(out / "work/ce_valid_sample.parquet")
print(f"CE-B sample: {len(tr):,} pairs, {tr['sidx'].n_unique():,} businesses, pos {tr['label'].mean():.3f}", flush=True)

ptr = pl.read_parquet(args.pairs / "pairs_train_uncertain.parquet")
pte = pl.read_parquet(args.pairs / "pairs_test_uncertain.parquet")
shutil.copy2(args.pairs / "pairs_train_uncertain.parquet", out / "pairs_train_uncertain.parquet")
shutil.copy2(args.pairs / "pairs_test_uncertain.parquet", out / "pairs_test_uncertain.parquet")
need = {"train": pl.concat([tr.select("sidx", "tidx"), va.select("sidx", "tidx"), ptr.select("sidx", "tidx")]),
        "test": pte.select("sidx", "tidx")}
src = args.work / "ce_tokens"
for f in ("train.json", "test.json"):
    shutil.copy2(src / f, out / "work/ce_tokens" / f)
shutil.copytree(src / "tokenizer", out / "work/ce_tokens/tokenizer", dirs_exist_ok=True)
for split, frame in need.items():
    for side, col in (("s1", "sidx"), ("tg", "tidx")):
        idx = np.unique(frame[col].to_numpy().astype(np.int64))
        arr = np.load(src / f"{split}_{side}.npy", mmap_mode="r")
        rows = np.asarray(arr[idx])
        lengths = np.load(src / f"{split}_{side}_lengths.npy")
        width = int(lengths[idx].max())
        np.savez_compressed(out / f"work/ce_tokens/{split}_{side}.sparse.npz", shape=np.array(arr.shape),
                            idx=idx.astype(np.uint32), rows=rows[:, :width], lengths=lengths)
        print(f"{split}_{side}: {len(idx):,} of {arr.shape[0]:,} rows, width {width}", flush=True)
