"""Stage 1: read raw TSVs, normalize text in parallel, cache as parquet."""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import polars as pl

from .normalize import load_translit, norm_addr, norm_name

FILES = {
    "train": ["train_source1", "train_source2", "train_source3"],
    "test": ["test_source1", "test_source2", "test_source3"],
}


def read_tsv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True,
    ).with_columns(pl.all().fill_null(""))


def _norm_chunk(args):
    names, addrs = args
    full, core, addr = [], [], []
    for n, a in zip(names, addrs):
        f, c = norm_name(n)
        full.append(f)
        core.append(c)
        addr.append(norm_addr(a))
    return full, core, addr


def normalize_frame(df: pl.DataFrame, pool: Pool, chunk: int = 25_000,
                    buffer_rows: int = 200_000) -> pl.DataFrame:
    # Bound Python strings and queued multiprocessing payloads to one window.
    parts = []
    for window in df.iter_slices(buffer_rows):
        jobs = [(part["business_name"].to_list(), part["business_address"].to_list())
                for part in window.iter_slices(chunk)]
        full, core, addr = [], [], []
        for f, c, a in pool.imap(_norm_chunk, jobs):
            full.extend(f)
            core.extend(c)
            addr.extend(a)
        parts.append(window.with_columns(
            pl.Series("name_n", full), pl.Series("core_n", core), pl.Series("addr_n", addr)))
    return pl.concat(parts) if parts else df.with_columns(
        name_n=pl.lit(""), core_n=pl.lit(""), addr_n=pl.lit(""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="student_resource/dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8))
    ap.add_argument("--buffer-rows", type=int, default=200_000)
    ap.add_argument("--translit", default=None, help="learned token map (translit.py)")
    args = ap.parse_args()
    out = Path(args.work) / "norm"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=load_translit, initargs=(args.translit,)) as pool:
        for split in args.splits:
            for name in FILES[split]:
                t = time.time()
                df = read_tsv(Path(args.dataset) / split / f"{name}.tsv")
                df = normalize_frame(df, pool, buffer_rows=args.buffer_rows)
                df = df.with_row_index("idx")
                df.write_parquet(out / f"{name}.parquet")
                print(f"{name}: {len(df):,} rows in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
