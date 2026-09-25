"""Measure a phonetic revision on fold 3 before rebuilding the full candidates."""
import argparse
import gc
import json
from pathlib import Path
import polars as pl
from er_v2.block import PHONETIC_CAPS, generate, make_phonetic_keys
from er_v2.indexing import build_index
from er_v2.run_block import load_split, split_countries, parquet_rows
from er_v2.run_features import ground_truth_pairs
from er_v2.train import fold_expr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()
    total = sum(parquet_rows(args.work / "norm" / f"train_source{i}.parquet") for i in (2, 3))
    results = {}
    for country in split_countries(args.work, "train"):
        s1, tg = load_split(args.work, "train", country)
        truth = ground_truth_pairs(args.dataset, s1, tg).filter(fold_expr() == 3)
        query = s1.filter((pl.col("idx").cast(pl.UInt32).hash(seed=11) % 10) == 3)
        index = build_index(tg, total, args.work, make_phonetic_keys, PHONETIC_CAPS)
        retrieved = generate(make_phonetic_keys(query), index, top_k=8, chunk=2000, verbose=False)
        original = (pl.scan_parquet(args.work / "cands_train.parquet").filter((fold_expr() == 3) & (pl.col("rescue") != 2))
                    .select("sidx", "tidx").collect(engine="streaming"))
        added = retrieved.join(original, on=["sidx", "tidx"], how="anti")
        hits = added.join(truth, on=["sidx", "tidx"])
        results[country] = {"new_pairs": len(added), "new_true_pairs": len(hits), "anchors": len(query)}
        print(country, results[country], flush=True)
        del s1, tg, query, index, original, added, retrieved, hits, truth
        gc.collect()
    (args.work / "phonetic_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
