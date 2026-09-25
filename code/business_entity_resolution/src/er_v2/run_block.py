"""Run blocking for a whole split and save candidates to parquet."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import polars as pl

from .block import RESCUE_CAPS, build_target_index, generate, make_keys, make_rescue_keys, merge_candidates
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
    ap.add_argument("--rescue-k", type=int, default=8, help="extra typo-channel candidates; 0 disables")
    args = ap.parse_args()
    work = Path(args.work)
    t = time.time()
    s1_all, tg_all = load_split(work, args.split)
    n_targets = len(tg_all)  # global count keeps IDF weights identical to unpartitioned blocking
    parts = []
    # Keys are country-scoped, so blocking one country at a time gives the same
    # candidates with a fraction of the peak memory.
    for country in s1_all["country"].unique().sort():
        s1 = s1_all.filter(pl.col("country") == country)
        tg = tg_all.filter(pl.col("country") == country)
        if tg.is_empty():
            continue
        tindex = build_target_index(make_keys(tg), n_targets)
        base = generate(make_keys(s1), tindex, top_k=args.top_k,
                        name_k=args.name_k, address_k=args.address_k, verbose=False)
        del tindex
        gc.collect()
        if args.rescue_k > 0:
            rindex = build_target_index(make_rescue_keys(tg), n_targets, RESCUE_CAPS)
            rescue = generate(make_rescue_keys(s1), rindex, top_k=args.rescue_k, verbose=False)
            part = merge_candidates(base, rescue, args.top_k)
            del rindex, rescue
        else:
            part = base.with_columns(rescue=pl.lit(0, pl.Int8))
        parts.append(part)
        print(f"  {country}: {len(part):,} pairs for {part['sidx'].n_unique():,}/{len(s1):,} "
              f"source1, {time.time() - t:.0f}s", flush=True)
        del base, s1, tg
        gc.collect()
    cands = pl.concat(parts).sort("sidx", "brank")
    cands.write_parquet(work / f"cands_{args.split}.parquet")
    metadata = {"top_k": args.top_k, "name_k": args.name_k, "address_k": args.address_k,
                "rescue_k": args.rescue_k, "pairs": len(cands), "anchors": len(s1_all)}
    (work / f"cands_{args.split}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{args.split}: {len(cands):,} candidate pairs for "
          f"{cands['sidx'].n_unique():,}/{len(s1_all):,} source1 in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
