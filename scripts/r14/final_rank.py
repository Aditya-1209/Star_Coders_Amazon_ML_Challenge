"""Final decision layer on r12 scores: rank-aware cutoffs + France calibration.

Macro F0.5 is per business, so a business's first correct match is worth far more
than its third, and a wrong first match on a singleton costs a full point. Hence:
  * US/India (labelled): a business's top remaining candidate needs >= --t1, every
    further candidate >= --t2. Chosen on fold 3, confirmed on fold 4 (r12 scores:
    +0.000066 / +0.000165 over the single 0.719 cutoff).
  * France (unlabelled, 2.3x more ambiguous candidates): one stricter cutoff
    --fr (leaderboard: 0.719 -> 0.983713, 0.85 -> 0.984124, 0.90 -> 0.984248).
Exclusivity first: every target goes to its highest-scoring business (ties -> lowest
index), then the cutoffs apply. Candidates are the full r12 stacker input set.
"""
import argparse
import json
from pathlib import Path

import polars as pl

from er_v2.decision import best_per_target
from er_v2.predict import write_lists

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, default=Path("D:/star_r10b/work"))
ap.add_argument("--scores", type=Path, default=Path("D:/star_r14/r12_test_scores_local.parquet"))
ap.add_argument("--t1", type=float, default=0.60)
ap.add_argument("--t2", type=float, default=0.7190439701080322)
ap.add_argument("--fr", type=float, default=0.90)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
N = args.work / "norm"
country = pl.read_parquet(N / "test_source1.parquet", columns=["idx", "country"]).select(
    sidx=pl.col("idx").cast(pl.UInt32), country="country")
scores = pl.read_parquet(args.scores).select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32),
                                              pl.col("score").cast(pl.Float64))
floor = min(args.t1, args.t2, args.fr)
won = best_per_target(scores.filter(pl.col("score") >= floor), "score").join(country, on="sidx")
won = won.with_columns(rk=pl.col("score").rank("ordinal", descending=True).over("sidx"))
fr = pl.col("country") == "France"
keep = (fr & (pl.col("score") >= args.fr)) | (~fr & (((pl.col("rk") == 1) & (pl.col("score") >= args.t1))
                                                    | ((pl.col("rk") > 1) & (pl.col("score") >= args.t2))))
matches = won.filter(keep).select("sidx", "tidx", "score")
ref = best_per_target(scores.filter(pl.col("score") >= args.t2), "score").join(country, on="sidx")
ref = ref.filter(~fr | (pl.col("score") >= args.fr)).select("sidx", "tidx")
k = ["sidx", "tidx"]
stats = {"matches": len(matches), "vs_r12_fr090_added": matches.select(k).join(ref, on=k, how="anti").height,
         "vs_r12_fr090_removed": ref.join(matches.select(k), on=k, how="anti").height,
         "by_country": matches.join(country, on="sidx").group_by("country").len().sort("country").rows()}
print(json.dumps(stats), flush=True)
args.output.mkdir(parents=True, exist_ok=True)
s1 = pl.read_parquet(N / "test_source1.parquet", columns=["idx", "entity_id"])
targets = pl.concat([pl.read_parquet(N / f"test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
write_lists(args.output / "candidate_pairs.tsv", s1, scores.sort("sidx", "score", descending=[False, True]),
            targets, "candidate_entity_ids")
write_lists(args.output / "matching_results.tsv", s1, matches.sort("sidx", "score", descending=[False, True]),
            targets, "matched_entity_ids")
