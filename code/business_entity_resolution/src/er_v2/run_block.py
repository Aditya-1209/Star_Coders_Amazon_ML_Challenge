"""Run blocking for a whole split and save candidates to parquet."""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import polars as pl

from .block import build_target_index, generate, make_keys, make_rescue_keys, merge_candidates, RESCUE_CAPS
from .runtime import positive_int


def load_split(work: Path, split: str, columns=None):
    norm = work / "norm"
    s1 = pl.read_parquet(norm / f"{split}_source1.parquet", columns=columns)
    s2 = pl.read_parquet(norm / f"{split}_source2.parquet", columns=columns)
    s3 = pl.read_parquet(norm / f"{split}_source3.parquet", columns=columns)
    tg = pl.concat([s2, s3.with_columns(pl.col("idx") + len(s2))])
    return s1, tg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--top-k", type=positive_int, default=64)
    ap.add_argument("--rescue-k", type=int, default=12, help="Additional typo-channel candidates; 0 disables")
    ap.add_argument("--chunk", type=positive_int, default=10_000)
    args = ap.parse_args()
    work = Path(args.work)
    t = time.time()
    if args.rescue_k < 0:
        ap.error('--rescue-k must be nonnegative')
    norm = work / 'norm'
    cols = ['idx', 'country', 'core_n', 'addr_n']
    sscan = pl.scan_parquet(norm / f'{args.split}_source1.parquet').select(cols)
    t2 = pl.scan_parquet(norm / f'{args.split}_source2.parquet').select(cols)
    n2 = t2.select(pl.len()).collect().item()
    t3 = (pl.scan_parquet(norm / f'{args.split}_source3.parquet').select(cols)
          .with_columns(pl.col('idx') + n2))
    tscan = pl.concat([t2, t3])
    # Country partitioning is equivalent to the original country-scoped keys.
    # Retain GLOBAL target count for identical base IDF weights.
    nt = tscan.select(pl.len()).collect().item()
    paths = []
    for ci, country in enumerate(sscan.select('country').unique().collect()['country'].sort()):
        s1 = sscan.filter(pl.col('country') == country).collect()
        tg = tscan.filter(pl.col('country') == country).collect()
        idx = build_target_index(make_keys(tg), max(nt, 1))
        base = generate(make_keys(s1), idx, top_k=args.top_k, chunk=args.chunk)
        del idx
        gc.collect()
        if args.rescue_k:
            idx = build_target_index(make_rescue_keys(tg), max(nt, 1), RESCUE_CAPS)
            rescue = generate(make_rescue_keys(s1), idx, top_k=args.rescue_k, chunk=args.chunk)
            cands = merge_candidates(base, rescue, args.top_k)
            del idx, rescue
        else:
            cands = base.with_columns(rescue=pl.lit(0, pl.Int8))
        path = work / f'cands_{args.split}_country_{ci}.parquet'
        cands.write_parquet(path)
        paths.append(path)
        print(f'{country}: {len(cands):,} pairs; elapsed {time.time()-t:.1f}s', flush=True)
        del s1, tg, base, cands
        gc.collect()
    destination = work / f'cands_{args.split}.parquet'
    temporary = destination.with_suffix('.parquet.tmp')
    pl.scan_parquet(paths).sink_parquet(temporary)
    temporary.replace(destination)
    for path in paths:
        path.unlink()
    print(f'{args.split}: complete in {time.time()-t:.1f}s', flush=True)


if __name__ == "__main__":
    main()
