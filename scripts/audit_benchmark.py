#!/usr/bin/env python3
"""Audit benchmark exports, independently rescore them, and reload the model."""

import argparse
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile

from er_baseline.data import SOURCE_HEADER, read_records, read_truth, read_tsv, write_tsv


def load_lists(path, column):
    result = {}
    for source, text in read_tsv(path, ["source1_entity_id", column]):
        if source in result:
            raise ValueError(f"Duplicate Source 1 ID: {source}")
        ids = text.split(",") if text else []
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate target in list for {source}")
        result[source] = set(ids)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--validator", type=Path, default=Path("student_resource/utils/validate_submission.py"))
    args = parser.parse_args()
    truth = read_truth(args.dev / "ground_truth.tsv")
    splits = dict(read_tsv(args.dev / "splits.tsv", ["source1_entity_id", "split"]))
    anchors = [r for r in read_records(args.dev / "anchors.tsv") if splits[r.entity_id] == "holdout"]
    matched = load_lists(args.run / "holdout_matching_results.tsv", "matched_entity_ids")
    candidates = load_lists(args.run / "holdout_candidate_pairs.tsv", "candidate_entity_ids")
    expected = {r.entity_id for r in anchors}
    if set(matched) != expected or set(candidates) != expected:
        raise ValueError("Holdout export rows do not match the held-out anchor set")
    all_candidates = set()
    independent_scores = []
    for source in expected:
        if not matched[source] <= candidates[source]:
            raise ValueError("A predicted match is absent from its candidate set")
        all_candidates.update(candidates[source])
        actual, predicted = truth[source], matched[source]
        tp = len(actual & predicted)
        if not actual and not predicted:
            score = 1.0
        elif not tp:
            score = 0.0
        else:
            precision, recall = tp / len(predicted), tp / len(actual)
            score = 1.25 * precision * recall / (0.25 * precision + recall)
        independent_scores.append(score)
    independently_scored = sum(independent_scores) / len(independent_scores)
    reported = json.loads((args.run / "metrics.json").read_text())["partitions"]["holdout"]["macro_f05"]
    if not math.isclose(independently_scored, reported, abs_tol=1e-12):
        raise ValueError("Independent metric calculation disagrees")
    with sqlite3.connect(f"file:{args.index.resolve()}?mode=ro", uri=True) as db:
        source_paths = json.loads(db.execute("SELECT value FROM metadata WHERE key='source_paths'").fetchone()[0])
        ids = sorted(all_candidates)
        for start in range(0, len(ids), 800):
            batch = ids[start:start + 800]
            placeholders = ",".join("?" for _ in batch)
            present = {r[0] for r in db.execute(f"SELECT entity_id FROM records WHERE entity_id IN ({placeholders})", batch)}
            if present != set(batch):
                raise ValueError("Candidate IDs absent from the indexed target corpus")
    test_log = args.test_log.read_text()
    test_count = re.search(r"Ran (\d+) tests", test_log)
    if not test_count or not test_log.rstrip().endswith("OK"):
        raise ValueError("Unit-test log does not show a successful run")
    with tempfile.TemporaryDirectory(prefix="er_benchmark_audit_") as directory:
        fixture = Path(directory)
        write_tsv(fixture / "test_source1.tsv", SOURCE_HEADER, (a.values() for a in anchors))
        for source in (2, 3):
            path = next((Path(p) for p in source_paths if Path(p).name.endswith(f"source{source}.tsv")), None)
            if path is None:
                raise ValueError("Audit expects an index built from separate source2/source3 files")
            (fixture / f"test_source{source}.tsv").symlink_to(path.resolve())
        checked = subprocess.run([
            sys.executable, str(args.validator), "--matching", str(args.run / "holdout_matching_results.tsv"),
            "--candidate", str(args.run / "holdout_candidate_pairs.tsv"), "--test-dir", str(fixture), "--check-ids",
        ], capture_output=True, text=True, check=True)
        (args.run / "official_validation.log").write_text(checked.stdout)
        print(checked.stdout, flush=True)
        selected = {a.entity_id: a for a in anchors[:40] + [a for a in anchors if not truth[a.entity_id]][:10]}
        write_tsv(fixture / "smoke.tsv", SOURCE_HEADER, (a.values() for a in selected.values()))
        from er_baseline.model import predict
        predict(fixture / "smoke.tsv", args.index, args.run, fixture / "predictions")
        for filename, column, reference in (
            ("matching_results.tsv", "matched_entity_ids", matched),
            ("candidate_pairs.tsv", "candidate_entity_ids", candidates),
        ):
            actual = load_lists(fixture / "predictions" / filename, column)
            if set(actual) != set(selected) or any(actual[s] != reference[s] for s in selected):
                raise ValueError("Reloaded-model inference differs from benchmark exports")
    result = {
        "unit_tests_passed": int(test_count.group(1)),
        "official_validator_with_check_ids": f"PASS on all {len(expected)} development holdout rows",
        "strict_candidate_subset_and_valid_id_audit": "PASS",
        "independent_exported_tsv_metric_recalculation": "PASS",
        "independently_scored_macro_f05": independently_scored,
        "saved_model_inference_exact_match_anchors": len(selected),
        "validation_scope": "Sampled holdout businesses against the indexed training corpus; not official test predictions",
    }
    (args.run / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
