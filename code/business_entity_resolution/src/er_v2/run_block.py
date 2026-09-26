"""Run blocking for a whole split and save candidates to parquet."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from .block import (RESCUE_CAPS, PHONETIC_CAPS, generate, make_keys, make_rescue_keys,
                    make_phonetic_keys, merge_candidates)
from .indexing import build_index
from .runtime import positive_int


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def split_countries(work: Path, split: str) -> list[str]:
    return (pl.scan_parquet(work / "norm" / f"{split}_source1.parquet")
            .select("country").unique().collect()["country"].sort().to_list())


def load_split(work: Path, split: str, country: str | None = None, columns=None):
    norm = work / "norm"
    n_s2 = parquet_rows(norm / f"{split}_source2.parquet")
    def read(side):
        frame = pl.scan_parquet(norm / f"{split}_source{side}.parquet")
        if country is not None:
            frame = frame.filter(pl.col("country") == country)
        if columns is not None:
            frame = frame.select(columns)
        if side == 3:
            frame = frame.with_columns(pl.col("idx") + n_s2)
        return frame.collect(engine="streaming")
    return read(1), pl.concat([read(2), read(3)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--candidate-prefix", choices=["cands", "key"], default="cands")
    ap.add_argument("--top-k", type=positive_int, default=64)
    ap.add_argument("--name-k", type=int, default=16, help="extra name-only candidates; 0 disables")
    ap.add_argument("--address-k", type=int, default=8, help="extra address-only candidates; 0 disables")
    ap.add_argument("--rescue-k", type=int, default=8, help="extra typo-channel candidates; 0 disables")
    ap.add_argument("--phonetic-k", type=int, default=0, help="extra consonant-class candidates; experimental")
    ap.add_argument("--block-chunk", type=positive_int, default=2_000)
    ap.add_argument("--augment-phonetic", action="store_true", help="reuse existing original channels, replace only phonetic extras")
    args = ap.parse_args()
    work = Path(args.work)
    t = time.time()
    n_targets = sum(parquet_rows(work / "norm" / f"{args.split}_source{i}.parquet") for i in (2, 3))
    n_anchors = parquet_rows(work / "norm" / f"{args.split}_source1.parquet")
    if args.augment_phonetic:
        previous = json.loads((work / f"{args.candidate_prefix}_{args.split}.json").read_text())
        for key in ("top_k", "name_k", "address_k", "rescue_k"):
            if previous[key] != getattr(args, key):
                raise ValueError(f"Cannot reuse original channels with changed {key}")
    scratch = TemporaryDirectory(prefix="candidates-", dir=work)
    paths, pair_count, covered = [], 0, 0
    # Keys are country-scoped, so blocking one country at a time gives the same
    # candidates with a fraction of the peak memory.
    for country in split_countries(work, args.split):
        s1, tg = load_split(work, args.split, country, ["idx", "country", "core_n", "addr_n"])
        if tg.is_empty():
            continue
        print(f"indexing {country}: {len(tg):,} targets", flush=True)
        if args.augment_phonetic:
            base = (pl.scan_parquet(work / f"{args.candidate_prefix}_{args.split}.parquet")
                    .filter(pl.col("rescue") != 2)
                    .join(s1.select(sidx="idx").lazy(), on="sidx", how="semi").collect(engine="streaming"))
            part = base
        else:
            tindex = build_index(tg, n_targets, work)
            base = generate(make_keys(s1), tindex, top_k=args.top_k,
                            name_k=args.name_k, address_k=args.address_k, chunk=args.block_chunk, verbose=True)
            del tindex
            gc.collect()
            if args.rescue_k > 0:
                rindex = build_index(tg, n_targets, work, make_rescue_keys, RESCUE_CAPS)
                rescue = generate(make_rescue_keys(s1), rindex, top_k=args.rescue_k, chunk=args.block_chunk, verbose=False)
                part = merge_candidates(base, rescue, args.top_k)
                del rindex, rescue
            else:
                part = base.with_columns(rescue=pl.lit(0, pl.Int8))
        if args.phonetic_k > 0:
            pindex = build_index(tg, n_targets, work, make_phonetic_keys, PHONETIC_CAPS)
            phonetic = generate(make_phonetic_keys(s1), pindex, top_k=args.phonetic_k,
                                chunk=args.block_chunk, verbose=False)
            phonetic = phonetic.join(part.select("sidx", "tidx"), on=["sidx", "tidx"], how="anti")
            phonetic = phonetic.with_columns(rescue=pl.lit(2, pl.Int8),
                                              brank=pl.col("brank") + args.top_k + args.rescue_k)
            part = pl.concat([part, phonetic])
            del pindex, phonetic
        path = Path(scratch.name) / f"{len(paths)}.parquet"
        part.write_parquet(path)
        paths.append(path)
        pair_count += len(part)
        covered += part["sidx"].n_unique()
        print(f"  {country}: {len(part):,} pairs for {part['sidx'].n_unique():,}/{len(s1):,} "
              f"source1, {time.time() - t:.0f}s", flush=True)
        del base, part, s1, tg
        gc.collect()
    destination = work / f"{args.candidate_prefix}_{args.split}.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    pl.scan_parquet(paths).sink_parquet(temporary)
    temporary.replace(destination)
    scratch.cleanup()
    metadata = {"top_k": args.top_k, "name_k": args.name_k, "address_k": args.address_k,
                "rescue_k": args.rescue_k, "phonetic_k": args.phonetic_k,
                "pairs": pair_count, "anchors": n_anchors}
    (work / f"{args.candidate_prefix}_{args.split}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{args.split}: {pair_count:,} candidate pairs for "
          f"{covered:,}/{n_anchors:,} source1 in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
