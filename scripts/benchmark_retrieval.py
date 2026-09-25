#!/usr/bin/env python3
"""Measure full-index retrieval on a seeded sample of training anchors only."""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

from er_baseline.data import read_records, read_truth, read_tsv
from er_baseline.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--posting-budget", type=int, default=0)
    args = parser.parse_args()
    splits = dict(read_tsv(args.dev / "splits.tsv", ["source1_entity_id", "split"]))
    truth = read_truth(args.dev / "ground_truth.tsv")
    eligible = [r for r in read_records(args.dev / "anchors.tsv") if splits[r.entity_id] == "train"]
    anchors = random.Random(2026).sample(eligible, min(args.count, len(eligible)))
    retriever = Retriever(args.index, posting_budget=args.posting_budget)
    durations = []
    total = found = pairs = 0
    started = time.monotonic()
    try:
        for i, anchor in enumerate(anchors, 1):
            start = time.monotonic()
            candidates = retriever.query(anchor)
            durations.append(time.monotonic() - start)
            ids = {r.entity_id for r, _ in candidates}
            expected = truth[anchor.entity_id]
            total += len(expected)
            found += len(expected & ids)
            pairs += len(ids)
            if i % 10 == 0:
                print(f"{i}/{len(anchors)} queries; {time.monotonic() - started:.1f}s elapsed", flush=True)
        result = {
            "anchors": len(anchors), "indexed_targets": retriever.metadata["records"],
            "posting_budget": args.posting_budget,
            "elapsed_seconds": time.monotonic() - started,
            "mean_query_seconds": statistics.mean(durations),
            "median_query_seconds": statistics.median(durations),
            "max_query_seconds": max(durations),
            "candidate_pair_recall": found / total if total else None,
            "mean_candidates": pairs / len(anchors),
            "estimated_6000_anchor_retrieval_minutes": statistics.mean(durations) * 6000 / 60,
            "estimated_full_test_single_process_hours": statistics.mean(durations) * 1732544 / 3600,
            "scope": "60 or fewer seeded training anchors; timings are a rough extrapolation, not a completion promise.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
