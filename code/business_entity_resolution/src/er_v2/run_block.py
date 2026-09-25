"""Run blocking for a whole split and save candidates to parquet."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from .block import build_target_index, generate, make_keys
from .runtime import positive_int


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
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--top-k", type=positive_int, default=64)
    ap.add_argument("--name-k", type=int, default=16, help="extra name-only candidates; 0 disables")
    ap.add_argument("--address-k", type=int, default=8, help="extra address-only candidates; 0 disables")
    args = ap.parse_args()
    work = Path(args.work)
    t = time.time()
    s1, tg = load_split(work, args.split)
    tindex = build_target_index(make_keys(tg), len(tg))
    print(f"target index {len(tindex):,} postings in {time.time() - t:.0f}s", flush=True)
    cands = generate(make_keys(s1), tindex, top_k=args.top_k,
                     name_k=args.name_k, address_k=args.address_k)
    cands = cands.sort("sidx", "brank")
    cands.write_parquet(work / f"cands_{args.split}.parquet")
    metadata = {"top_k": args.top_k, "name_k": args.name_k, "address_k": args.address_k,
                "pairs": len(cands), "anchors": len(s1)}
    (work / f"cands_{args.split}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{args.split}: {len(cands):,} candidate pairs for "
          f"{cands['sidx'].n_unique():,}/{len(s1):,} source1 in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
