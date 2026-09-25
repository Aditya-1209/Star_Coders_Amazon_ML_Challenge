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


def normalize_frame(df: pl.DataFrame, pool: Pool, chunk: int = 50_000) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    full, core, addr = [], [], []
    for f, c, a in pool.imap(_norm_chunk, jobs):
        full += f
        core += c
        addr += a
    return df.with_columns(
        pl.Series("name_n", full), pl.Series("core_n", core), pl.Series("addr_n", addr),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="student_resource/dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--translit", default=None, help="learned token map (translit.py)")
    args = ap.parse_args()
    out = Path(args.work) / "norm"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=load_translit, initargs=(args.translit,)) as pool:
        for split in args.splits:
            for name in FILES[split]:
                t = time.time()
                df = read_tsv(Path(args.dataset) / split / f"{name}.tsv")
                df = normalize_frame(df, pool)
                df = df.with_row_index("idx")
                df.write_parquet(out / f"{name}.parquet")
                print(f"{name}: {len(df):,} rows in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
