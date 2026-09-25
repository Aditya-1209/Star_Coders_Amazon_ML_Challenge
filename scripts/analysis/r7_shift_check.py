"""Compare train/test separability with and without the new pair-local features.

Uses no matching labels. Validation is grouped by Source 1 record. This is a
heuristic shift diagnostic, not a substitute for a labeled unseen-country test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from er_v2.features import feature_names
from er_v2.train import X
from er_v2.runtime import feature_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=100)
    args = ap.parse_args()
    example = pl.read_parquet(feature_parts(args.work / "feats_train")[0], n_rows=1)
    profiles = {p: feature_names(example, p) for p in ("baseline", "enhanced")}
    samples = {}
    for split in ("train", "test"):
        country = pl.read_parquet(args.work / "norm" / f"{split}_source1.parquet", columns=["idx", "country"])
        frame = (pl.scan_parquet(feature_parts(args.work / f"feats_{split}"))
                 .select("sidx", "tidx", *profiles["enhanced"])
                 .filter(pl.struct("sidx", "tidx").hash(seed=91) % 512 == 0)
                 .collect(engine="streaming"))
        samples[split] = frame.join(country.rename({"idx": "sidx"}), on="sidx")
    results = {}
    for country in ("India", "US"):
        chunks = []
        for label, split in enumerate(("train", "test")):
            part = samples[split].filter(pl.col("country") == country)
            if len(part) < 100:
                raise ValueError(f"Too few sampled pairs for {country}/{split}")
            chunks.append(part.sample(min(len(part), 150_000), seed=9).with_columns(_label=pl.lit(label)))
        frame = pl.concat(chunks)
        train = frame.filter(pl.col("sidx").hash(seed=61) % 5 != 0)
        valid = frame.filter(pl.col("sidx").hash(seed=61) % 5 == 0)
        trial = {"training_pairs": len(train), "validation_pairs": len(valid)}
        for profile, features in profiles.items():
            dtrain = xgb.QuantileDMatrix(X(train, features), train["_label"].to_numpy(),
                                         feature_names=features, nthread=args.threads)
            dvalid = xgb.QuantileDMatrix(X(valid, features), valid["_label"].to_numpy(),
                                         feature_names=features, nthread=args.threads, ref=dtrain)
            model = xgb.train({"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
                               "device": "cpu", "max_depth": 5, "eta": 0.15, "seed": 7,
                               "nthread": args.threads}, dtrain, args.rounds, verbose_eval=False)
            trial[profile + "_auc"] = float(model.eval_set([(dvalid, "valid")]).rsplit(":", 1)[1])
            print(f"{country} {profile}: AUC={trial[profile + '_auc']:.5f}", flush=True)
        trial["allow_enhanced"] = trial["enhanced_auc"] <= max(0.75, trial["baseline_auc"] + 0.02)
        results[country] = trial
    report = {"allow_enhanced": all(v["allow_enhanced"] for v in results.values()),
              "countries": results, "sample_seed": 91, "grouped_validation_seed": 61,
              "rule": "enhanced AUC <= max(0.75, baseline AUC + 0.02) in each labeled country",
              "limitation": "France has no matching labels; this diagnostic cannot establish its accuracy"}
    (args.work / "shift_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
