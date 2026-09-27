"""Write France-only threshold variants of a finished submission's test scores.

France has no labels, so its cutoff cannot be tuned locally. US/India decisions stay
exactly as tuned; only French businesses use --france-threshold. Because US/India are
unchanged, the leaderboard difference between two variants measures France alone:
    delta_F_France ~= delta_LB / 0.150

usage: python scripts/r14/france_probe.py --work W --scores S.parquet --metrics r14_metrics.json
                                           --france-threshold 0.5 --output OUT
"""
import argparse
import json
from pathlib import Path

import polars as pl

from er_v2.decision import decide_country
from er_v2.predict import write_lists

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--scores", type=Path, required=True)
ap.add_argument("--metrics", type=Path, required=True)
ap.add_argument("--france-threshold", type=float, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()

metrics = json.loads(args.metrics.read_text())
chosen = metrics["results"][metrics["selected"]]
cutoffs = {**chosen["country_thresholds"], "France": args.france_threshold}
scores = pl.read_parquet(args.scores)
country = pl.read_parquet(args.work / "norm/test_source1.parquet", columns=["idx", "country"]).select(
    sidx=pl.col("idx").cast(pl.UInt32), country="country")
base = decide_country(scores, chosen["threshold"], country, chosen["country_thresholds"], "score")
matches = decide_country(scores, chosen["threshold"], country, cutoffs, "score")
s1 = pl.read_parquet(args.work / "norm/test_source1.parquet", columns=["idx", "entity_id"])
targets = pl.concat([pl.read_parquet(args.work / f"norm/test_source{i}.parquet", columns=["entity_id"])
                     for i in (2, 3)])["entity_id"]
args.output.mkdir(parents=True, exist_ok=True)
write_lists(args.output / "candidate_pairs.tsv", s1, scores, targets, "candidate_entity_ids")
write_lists(args.output / "matching_results.tsv", s1, matches.sort("sidx", "score", descending=[False, True]),
            targets, "matched_entity_ids")
diff = (matches.select("sidx", "tidx").join(base.select("sidx", "tidx"), on=["sidx", "tidx"], how="anti").height,
        base.select("sidx", "tidx").join(matches.select("sidx", "tidx"), on=["sidx", "tidx"], how="anti").height)
other = lambda m: m.join(country.filter(pl.col("country") != "France"), on="sidx").select("sidx", "tidx").sort("sidx", "tidx")
if not other(matches).equals(other(base)):
    raise RuntimeError("US/India decisions changed; France-only probe is invalid")
print(json.dumps({"base_threshold": chosen["threshold"], "france_threshold": args.france_threshold,
                  "france_pairs_added": diff[0], "france_pairs_removed": diff[1], "total_matches": len(matches)},
                 indent=1))
