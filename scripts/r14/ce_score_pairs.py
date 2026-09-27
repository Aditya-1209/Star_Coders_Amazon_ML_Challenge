"""Score an explicit (sidx, tidx) pair list with the trained cross-encoder.

r14 only scores pairs the graph model is unsure about (score in (LO, HI)); confident
pairs keep the graph verdict. Reuses r10_ce's token cache and OOM-safe inference.

usage: python scripts/r14/ce_score_pairs.py --work W --split train|test --pairs P.parquet --out O.parquet
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from er_v2.r10_ce import PairTokens, score_logits

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--split", choices=["train", "test"], required=True)
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--chunk", type=int, default=200_000)
args = ap.parse_args()

import torch
from transformers import AutoModelForSequenceClassification

t0 = time.time()
cache = PairTokens(args.work, args.split)
model = AutoModelForSequenceClassification.from_pretrained(args.work / "ce_model").to("cuda").eval()
pairs = pl.read_parquet(args.pairs).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32)).unique()
out = []
for i in range(0, len(pairs), args.chunk):
    frame = pairs.slice(i, args.chunk)
    values = score_logits(model, cache, frame, "cuda", args.batch)
    out.append(frame.with_columns(ce_logit=pl.Series(values, dtype=pl.Float32)))
    done = min(i + args.chunk, len(pairs))
    rate = done / (time.time() - t0)
    print(f"{args.split}: {done:,}/{len(pairs):,} pairs, {rate:,.0f}/s, eta {(len(pairs) - done) / rate / 60:.1f} min", flush=True)
res = pl.concat(out)
if not np.isfinite(res["ce_logit"].to_numpy()).all():
    raise RuntimeError("non-finite CE scores")
tmp = args.out.with_suffix(".partial.parquet")
res.write_parquet(tmp)
tmp.replace(args.out)
print(f"wrote {len(res):,} scores to {args.out} in {time.time() - t0:.0f}s", flush=True)
