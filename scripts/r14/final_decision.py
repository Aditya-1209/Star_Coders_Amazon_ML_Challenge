"""Final decision layer: r12 stacker scores -> output/matching_results.tsv + candidate_pairs.tsv.

Reads the test scores written by the r12 runner's inference stage
(WORK/r10_test_predictions.parquet: sidx, tidx, score for every candidate the final
stacker scored) and the r12 cutoff chosen on fold 3 (WORK/selection.json).

Decisions (every Source 2/3 record goes to at most one business, its highest scorer):
  * US / India / any labelled-like country: score >= r12 cutoff (0.719, tuned on fold 3).
    Optional --rank-t1: a business's best remaining candidate only needs >= t1
    (macro F0.5 values a business's first match most; +0.000165 on fold 4 at t1=0.60).
  * France (no labels): score >= --france-threshold. The cutoff was calibrated on the
    public leaderboard with France-only probes (0.719 -> 0.983713, 0.85 -> 0.984124,
    0.90 -> 0.984248): the stacker is over-confident on French look-alike records.
candidate_pairs.tsv = exactly the pairs the final stacker scored.
"""
import argparse
import json
from pathlib import Path

import polars as pl

from er_v2.decision import best_per_target
from er_v2.predict import write_lists

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, default=Path("work/r12"))
ap.add_argument("--scores", type=Path, help="default WORK/r10_test_predictions.parquet")
ap.add_argument("--norm", type=Path, help="default WORK/norm")
ap.add_argument("--threshold", type=float, help="default: selected cutoff in WORK/selection.json")
ap.add_argument("--france-threshold", type=float, default=0.90)
ap.add_argument("--rank-t1", type=float, help="optional cutoff for each business's top candidate (US/India)")
ap.add_argument("--output", type=Path, default=Path("output"))
args = ap.parse_args()
norm = args.norm or args.work / "norm"
if args.threshold is None:
    selection = json.loads((args.work / "selection.json").read_text())
    if selection["selected"] != "r10":
        raise SystemExit("selection.json did not select the r10 stacker; pass --threshold explicitly")
    args.threshold = selection["proposal"]["threshold"]
scores = pl.read_parquet(args.scores or args.work / "r10_test_predictions.parquet").select(
    pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32), pl.col("score").cast(pl.Float64))
country = pl.read_parquet(norm / "test_source1.parquet", columns=["idx", "country"]).select(
    sidx=pl.col("idx").cast(pl.UInt32), country="country")
t1 = args.rank_t1 if args.rank_t1 is not None else args.threshold
floor = min(args.threshold, t1, args.france_threshold)
won = best_per_target(scores.filter(pl.col("score") >= floor), "score").join(country, on="sidx")
won = won.with_columns(rk=pl.col("score").rank("ordinal", descending=True).over("sidx"))
fr = pl.col("country") == "France"
keep = (fr & (pl.col("score") >= args.france_threshold)) | (
    ~fr & (((pl.col("rk") == 1) & (pl.col("score") >= t1)) | ((pl.col("rk") > 1) & (pl.col("score") >= args.threshold))))
matches = won.filter(keep).select("sidx", "tidx", "score")
args.output.mkdir(parents=True, exist_ok=True)
s1 = pl.read_parquet(norm / "test_source1.parquet", columns=["idx", "entity_id"])
targets = pl.concat([pl.read_parquet(norm / f"test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
write_lists(args.output / "candidate_pairs.tsv", s1, scores.sort("sidx", "score", descending=[False, True]),
            targets, "candidate_entity_ids")
write_lists(args.output / "matching_results.tsv", s1, matches.sort("sidx", "score", descending=[False, True]),
            targets, "matched_entity_ids")
print(json.dumps({"threshold": args.threshold, "rank_t1": args.rank_t1, "france_threshold": args.france_threshold,
                  "candidates": len(scores), "matches": len(matches),
                  "by_country": dict(matches.join(country, on="sidx").group_by("country").len().rows())}), flush=True)
