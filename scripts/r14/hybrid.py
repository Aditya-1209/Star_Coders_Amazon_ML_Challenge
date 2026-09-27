"""Country hybrid: US/India decisions from a base scorer, France from another scorer.

Targets are country-scoped, so exclusive ownership never crosses countries; non-French
rows are verified identical to the base decision.

usage: python scripts/r14/hybrid.py --work W --base S.parquet --base-thr T --fr S2.parquet --fr-thr T2 --output OUT
"""
import argparse
import json
from pathlib import Path

import polars as pl

from er_v2.decision import decide_country
from er_v2.predict import write_lists

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--base-thr", type=float, required=True)
ap.add_argument("--fr", type=Path, required=True)
ap.add_argument("--fr-thr", type=float, required=True)
ap.add_argument("--output", type=Path)
args = ap.parse_args()
N = args.work / "norm"
country = pl.read_parquet(N / "test_source1.parquet", columns=["idx", "country"]).select(
    sidx=pl.col("idx").cast(pl.UInt32), country="country")
fr_ids = country.filter(pl.col("country") == "France").select("sidx")
key = lambda f: f.select(pl.col("sidx").cast(pl.UInt32), pl.col("tidx").cast(pl.UInt32), pl.col("score").cast(pl.Float64))
base = key(pl.read_parquet(args.base))
alt = key(pl.read_parquet(args.fr))
scores = pl.concat([base.join(fr_ids, on="sidx", how="anti"), alt.join(fr_ids, on="sidx", how="semi")])
matches = pl.concat([decide_country(base.join(fr_ids, on="sidx", how="anti"), args.base_thr, country, {}, "score"),
                     decide_country(alt.join(fr_ids, on="sidx", how="semi"), args.fr_thr, country, {}, "score")])
ref = decide_country(base, args.base_thr, country, {}, "score")
other = lambda m: m.join(fr_ids, on="sidx", how="anti").select("sidx", "tidx").sort("sidx", "tidx")
if not other(matches).equals(other(ref)):
    raise RuntimeError("non-French decisions differ from base")
fr_new = matches.join(fr_ids, on="sidx", how="semi").select("sidx", "tidx")
fr_ref = ref.join(fr_ids, on="sidx", how="semi").select("sidx", "tidx")
stats = {"france_matches": len(fr_new), "france_base": len(fr_ref),
         "added": fr_new.join(fr_ref, on=["sidx", "tidx"], how="anti").height,
         "removed": fr_ref.join(fr_new, on=["sidx", "tidx"], how="anti").height,
         "france_businesses_changed": pl.concat([fr_new.join(fr_ref, on=["sidx", "tidx"], how="anti"),
                                                 fr_ref.join(fr_new, on=["sidx", "tidx"], how="anti")])["sidx"].n_unique()}
print(json.dumps(stats), flush=True)
if args.output:
    args.output.mkdir(parents=True, exist_ok=True)
    s1 = pl.read_parquet(N / "test_source1.parquet", columns=["idx", "entity_id"])
    targets = pl.concat([pl.read_parquet(N / f"test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
    write_lists(args.output / "candidate_pairs.tsv", s1, scores.sort("sidx", "score", descending=[False, True]),
                targets, "candidate_entity_ids")
    write_lists(args.output / "matching_results.tsv", s1, matches.sort("sidx", "score", descending=[False, True]),
                targets, "matched_entity_ids")
