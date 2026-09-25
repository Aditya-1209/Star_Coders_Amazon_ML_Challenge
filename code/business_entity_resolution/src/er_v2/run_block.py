"""Run blocking for a whole split and save candidates to parquet."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from .block import build_target_index, generate, make_keys


def load_split(work: Path, split: str):
    norm = work / "norm"
    s1 = pl.read_parquet(norm / f"{split}_source1.parquet")
    s2 = pl.read_parquet(norm / f"{split}_source2.parquet")
    s3 = pl.read_parquet(norm / f"{split}_source3.parquet")
    tg = pl.concat([s2, s3.with_columns(pl.col("idx") + len(s2))])
    return s1, tg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", required=True)
    ap.add_argument("--top-k", type=int, default=64)
    args = ap.parse_args()
    work = Path(args.work)
    t = time.time()
    s1, tg = load_split(work, args.split)
    tindex = build_target_index(make_keys(tg), len(tg))
    print(f"target index {len(tindex):,} postings in {time.time() - t:.0f}s", flush=True)
    cands = generate(make_keys(s1), tindex, top_k=args.top_k)
    cands = cands.sort("sidx", "brank")
    cands.write_parquet(work / f"cands_{args.split}.parquet")
    print(f"{args.split}: {len(cands):,} candidate pairs for "
          f"{cands['sidx'].n_unique():,}/{len(s1):,} source1 in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
