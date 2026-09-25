"""Measure added phonetic retrieval on fold 3 only, before fitting models."""
import argparse
import json
from pathlib import Path
import polars as pl
from er_v2.metrics import by_country, macro_f05
from er_v2.run_block import load_split
from er_v2.run_features import ground_truth_pairs
from er_v2.train import fold_expr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()
    s1, targets = load_split(args.work, "train", columns=["idx", "entity_id", "country"])
    anchors = s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country").filter(fold_expr() == 3)
    truth = ground_truth_pairs(args.dataset, s1, targets).filter(fold_expr() == 3)
    del s1, targets
    candidates = (pl.scan_parquet(args.work / "cands_train.parquet").select("sidx", "tidx", "rescue")
                  .filter(fold_expr() == 3).collect(engine="streaming"))
    result = {"evaluation_fold": 3}
    for name, frame in (("original_channels", candidates.filter(pl.col("rescue") != 2)),
                        ("with_phonetic", candidates)):
        hits = frame.join(truth, on=["sidx", "tidx"])
        result[name] = {"oracle": macro_f05(hits, truth, anchors["sidx"]),
                        "country_oracles": by_country(hits, truth, anchors),
                        "candidates_per_anchor": len(frame) / len(anchors)}
    result["additional_true_pairs"] = candidates.filter(pl.col("rescue") == 2).join(truth, on=["sidx", "tidx"]).height
    (args.work / "retrieval_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
