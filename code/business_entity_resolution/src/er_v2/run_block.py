"""Run blocking for a whole split and save candidates to parquet."""
from __future__ import annotations

import argparse
import json
import gc
import tempfile
import time
from pathlib import Path

import polars as pl

from .block import PAIR_SCHEMA, build_target_index, generate, make_keys
from .runtime import positive_int


def parquet_rows(path: Path) -> int:
    from pyarrow.parquet import ParquetFile
    return ParquetFile(path).metadata.num_rows


def target_count(work: Path, split: str) -> int:
    return sum(parquet_rows(work / "norm" / f"{split}_source{source}.parquet") for source in (2, 3))


def split_countries(work: Path, split: str) -> list[str]:
    return sorted(pl.read_parquet(work / "norm" / f"{split}_source1.parquet",
                                  columns=["country"])["country"].unique().to_list())


def load_split(work: Path, split: str, country: str | None = None, columns=None):
    """Filter before collecting; indices always refer to the entire split."""
    norm = work / "norm"
    n_s2 = parquet_rows(norm / f"{split}_source2.parquet")
    def read(source):
        frame = pl.scan_parquet(norm / f"{split}_source{source}.parquet")
        if source == 3:
            frame = frame.with_columns(pl.col("idx") + n_s2)
        if country is not None:
            frame = frame.filter(pl.col("country") == country)
        if columns is not None:
            frame = frame.select(columns)
        return frame.collect(engine="streaming")
    s1 = read(1)
    tg = pl.concat([read(2), read(3)])
    return s1, tg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--top-k", type=positive_int, default=64)
    ap.add_argument("--name-k", type=int, default=16, help="extra name-only candidates; 0 disables")
    ap.add_argument("--address-k", type=int, default=8, help="extra address-only candidates; 0 disables")
    ap.add_argument("--block-chunk", type=positive_int, default=5_000)
    ap.add_argument("--country-partition", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    t = time.time()
    countries = split_countries(work, args.split) if args.country_partition else [None]
    n_targets = target_count(work, args.split)
    total, anchors, covered = 0, 0, 0
    # Spill completed countries so their candidates do not compete with the
    # next country's key index. Final order is irrelevant: sidx/tidx are global.
    with tempfile.TemporaryDirectory(prefix="blocking-", dir=work) as temporary:
        paths = []
        for index, country in enumerate(countries):
            s1, tg = load_split(work, args.split, country, ["idx", "country", "core_n", "addr_n"])
            anchors += len(s1)
            if tg.is_empty():
                cands = pl.DataFrame(schema=PAIR_SCHEMA)
            else:
                # Keep global N in IDF to preserve the unpartitioned ranking.
                tindex = build_target_index(make_keys(tg), n_targets)
                source_keys = make_keys(s1)
                del s1, tg
                cands = generate(source_keys, tindex, top_k=args.top_k, chunk=args.block_chunk,
                                 name_k=args.name_k, address_k=args.address_k)
                del source_keys, tindex
            total += len(cands)
            covered += cands["sidx"].n_unique()
            path = Path(temporary) / f"part_{index:03d}.parquet"
            cands.sort("sidx", "brank").write_parquet(path)
            paths.append(path)
            del cands
            gc.collect()
            print(f"block country={country or 'all'} complete, {time.time() - t:.0f}s", flush=True)
        destination = work / f"cands_{args.split}.parquet"
        staging = destination.with_suffix(".parquet.tmp")
        pl.scan_parquet(paths).sink_parquet(staging)
        staging.replace(destination)
    metadata = {"top_k": args.top_k, "name_k": args.name_k, "address_k": args.address_k,
                "pairs": total, "anchors": anchors, "block_chunk": args.block_chunk,
                "country_partition": args.country_partition}
    (work / f"cands_{args.split}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{args.split}: {total:,} candidate pairs for "
          f"{covered:,}/{anchors:,} source1 in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
