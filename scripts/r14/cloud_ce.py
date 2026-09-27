"""VM side of r14 CE-B: rebuild token arrays, train on the uploaded sample, score uncertain pairs.

usage (from the bundle root): python cloud_ce.py unpack|train
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"


def unpack():
    for f in sorted((WORK / "ce_tokens").glob("*.sparse.npz")):
        name = f.name.replace(".sparse.npz", "")
        z = np.load(f)
        full = np.zeros(tuple(z["shape"]), dtype=np.uint32)
        full[z["idx"].astype(np.int64), :z["rows"].shape[1]] = z["rows"]
        np.save(WORK / "ce_tokens" / f"{name}.npy", full)
        np.save(WORK / "ce_tokens" / f"{name}_lengths.npy", z["lengths"])
        print(f"{name}: {full.shape}, {len(z['idx']):,} rows filled", flush=True)


def train(seed, batch, threads):
    from er_v2.neural import BASE_MODEL
    from er_v2.r10_ce import fit_model
    args = argparse.Namespace(work=WORK, base_model=BASE_MODEL, device="cuda", max_length=128, batch=batch,
                              score_batch=1024, accumulation=1, checkpointing=False, epochs=1, patience=1,
                              threads=threads, seed=seed, lr=2e-5)
    tr = pl.read_parquet(WORK / "ce_train_sample.parquet")
    va = pl.read_parquet(WORK / "ce_valid_sample.parquet")
    t0 = time.time()
    fit_model(args, tr, va)
    print(f"CE-B trained in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "unpack":
        unpack()
    elif stage == "train":
        train(seed=2027, batch=64, threads=10)
