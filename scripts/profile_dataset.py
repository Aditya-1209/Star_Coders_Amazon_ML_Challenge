#!/usr/bin/env python3
"""Inspect the supplied TSVs with a streaming, standard-library-only scan."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
import time


def rows(path, expected_header):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        if header != expected_header:
            raise ValueError(f"Unexpected header in {path}: {header}")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Bad row at physical line {reader.line_num} in {path}")
            yield row


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def profile(dataset, output, sample_size):
    started = time.monotonic()
    rng = random.Random(42)
    ground_truth_path = dataset / "train/train_ground_truth.tsv"
    distribution = Counter()
    targets_by_source = Counter()
    sampled = []
    gt_rows = bad_gt_ids = duplicate_lists = 0
    for source_id, target_text in rows(
        ground_truth_path, ["source1_entity_id", "matched_entity_ids"]
    ):
        gt_rows += 1
        targets = target_text.split(",") if target_text else []
        distribution[len(targets)] += 1
        duplicate_lists += len(targets) != len(set(targets))
        bad_gt_ids += not source_id.startswith("S1-")
        for target in targets:
            targets_by_source[target.split("-", 1)[0]] += 1
            bad_gt_ids += not target.startswith(("S2-", "S3-"))
        item = (source_id, targets)
        if len(sampled) < sample_size:
            sampled.append(item)
        else:
            position = rng.randrange(gt_rows)
            if position < sample_size:
                sampled[position] = item
    print(f"Ground truth: {gt_rows:,} rows", flush=True)
    wanted = {source for source, _ in sampled}
    wanted.update(target for _, targets in sampled for target in targets)
    sample_records = {}
    file_profiles = []
    header = ["entity_id", "business_name", "business_address", "country"]
    for split in ("train", "test"):
        for source in (1, 2, 3):
            path = dataset / split / f"{split}_source{source}.tsv"
            countries = Counter()
            empty = Counter({column: 0 for column in header})
            count = wrong_prefix = 0
            for values in rows(path, header):
                entity_id, name, address, country = values
                count += 1
                countries[country] += 1
                for column, value in zip(header, values):
                    if not value.strip():
                        empty[column] += 1
                wrong_prefix += not entity_id.startswith(f"S{source}-")
                if split == "train" and entity_id in wanted:
                    if entity_id in sample_records:
                        raise ValueError(f"Duplicate sampled entity ID: {entity_id}")
                    sample_records[entity_id] = dict(zip(header, values))
            file_profiles.append({
                "file": str(path.relative_to(dataset)), "split": split,
                "source": source, "rows": count, "bytes": path.stat().st_size,
                "countries": dict(countries), "empty_fields": dict(empty),
                "wrong_source_prefix": wrong_prefix,
            })
            print(f"{path.name}: {count:,} rows; {dict(countries)}", flush=True)

    pair_checks = Counter()
    examples = []
    for source_id, targets in sampled:
        source_record = sample_records.get(source_id)
        if source_record is None:
            raise ValueError(f"Sampled S1 ID absent from source data: {source_id}")
        for target_id in targets:
            target_record = sample_records.get(target_id)
            if target_record is None:
                raise ValueError(f"Sampled target ID absent from source data: {target_id}")
            pair_checks["pairs"] += 1
            pair_checks["same_country"] += source_record["country"] == target_record["country"]
            for field in ("business_name", "business_address"):
                pair_checks[f"exact_{field}"] += source_record[field] == target_record[field]
                pair_checks[f"normalized_exact_{field}"] += (
                    normalize(source_record[field]) == normalize(target_record[field])
                )
            if len(examples) < 6 and (
                source_record["business_name"] != target_record["business_name"]
                or source_record["business_address"] != target_record["business_address"]
            ):
                examples.append({"source1": source_record, "match": target_record})

    comparisons = {}
    for split in ("train", "test"):
        counts = {f["source"]: f["rows"] for f in file_profiles if f["split"] == split}
        comparisons[split] = counts[1] * (counts[2] + counts[3])
    train_s1_rows = next(f["rows"] for f in file_profiles if f["split"] == "train" and f["source"] == 1)
    total_pairs = sum(size * count for size, count in distribution.items())
    gt = {
        "rows": gt_rows, "matches": total_pairs,
        "singleton_rows": distribution[0],
        "singleton_fraction": distribution[0] / gt_rows,
        "mean_matches_per_source1": total_pairs / gt_rows,
        "match_count_distribution": dict(sorted(distribution.items())),
        "matches_by_source": dict(targets_by_source),
        "rows_with_duplicate_target_ids": duplicate_lists,
        "invalid_id_prefixes": bad_gt_ids,
        "row_count_equals_train_source1": gt_rows == train_s1_rows,
    }
    result = {
        "files": file_profiles, "ground_truth": gt,
        "all_pairs_comparisons": comparisons,
        "sample": {"seed": 42, "source1_rows": len(sampled),
                   "matched_pairs": dict(pair_checks), "examples": examples},
        "limitations": [
            "Full ID uniqueness and full ground-truth referential integrity were not checked.",
            "True-pair text similarity statistics use a seeded reservoir sample of Source 1 rows.",
            "No model has been trained; no validation or test performance is claimed.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_profile.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    text = ["# Dataset inspection", "", "All six source files and the ground-truth file were scanned completely, using a streaming TSV reader.", "", "| Split | Source | Rows | Country counts |", "| --- | --- | ---: | --- |"]
    for info in file_profiles:
        country_text = "; ".join(f"{country}: {count:,}" for country, count in sorted(info["countries"].items()))
        text.append(f"| {info['split']} | S{info['source']} | {info['rows']:,} | {country_text} |")
    text.extend([
        "", "## Training labels", "",
        f"- Ground-truth rows: {gt_rows:,}.",
        f"- Labeled matching pairs: {total_pairs:,}.",
        f"- Source 1 businesses with no matches: {distribution[0]:,} ({distribution[0] / gt_rows:.2%}).",
        f"- Mean matching records per Source 1 business: {total_pairs / gt_rows:.3f}.",
        f"- Ground-truth and training Source 1 row counts agree: {gt_rows == train_s1_rows}.",
        f"- Duplicate IDs within ground-truth lists: {duplicate_lists:,} lists.",
        f"- Invalid ground-truth source prefixes: {bad_gt_ids:,}.",
        "", "| Matches per Source 1 | Businesses |", "| ---: | ---: |",
    ])
    text.extend(f"| {size} | {count:,} |" for size, count in sorted(distribution.items()))
    text.extend(["", "## Data quality", ""])
    for info in file_profiles:
        missing = ", ".join(f"{column}={count:,}" for column, count in info["empty_fields"].items())
        text.append(f"- `{info['file']}`: blank fields: {missing}; wrong source prefixes: {info['wrong_source_prefix']:,}.")
    text.extend([
        "", "## Sampled true-match difficulty", "",
        f"A reservoir sample of {len(sampled):,} training Source 1 rows (seed 42) contains {pair_checks['pairs']:,} true matching pairs. The percentages below describe this sample, not the full population.", "",
    ])
    for key, value in pair_checks.items():
        if key != "pairs":
            text.append(f"- {key}: {value:,} / {pair_checks['pairs']:,} ({value / pair_checks['pairs']:.2%}).")
    text.extend([
        "", "## Compute implications", "",
        f"- Unrestricted training comparisons: {comparisons['train']:,}.",
        f"- Unrestricted test comparisons: {comparisons['test']:,}.",
        "- Build a retrieval/blocking index and score only candidate pairs; do not materialize the Cartesian product.",
        "- Keep raw TSVs on disk and process them in chunks; start experiments on a representative development subset.",
        "- RAM and GPU needs for training remain to be benchmarked on the chosen pipeline.",
        "", "## Next experiment", "",
        "1. Create a seeded split by Source 1 business, keeping its labeled matches together. Check for shared target IDs before treating groups as independent.",
        "2. Build candidate retrieval from names and addresses, and measure candidate recall before training the matcher. Evaluate with realistic unrelated records in the candidate pool.",
        "   Preserve Unicode text and assess alternate-script names, initials, and missing addresses explicitly; exact-name retrieval alone cannot cover the observed noise.",
        "3. Train a pair classifier with string-similarity features and hard negatives from retrieval.",
        "4. Tune the match threshold against macro F0.5 per Source 1, including singletons and true matches missed during retrieval.",
        "5. Evaluate generalization across US and India; preserve unknown country labels so France is included at inference.",
        "6. Save the exact candidates scored by the model. Generate both required TSV outputs and validate them before submission.",
        "   The supplied validator skips ID-existence checks by default (`--check-ids` enables them), and reports matches outside the candidate set only as warnings. Audit both conditions before submitting.",
        "", "## Limitations", "",
    ])
    text.extend(f"- {limitation}" for limitation in result["limitations"])
    text.append("")
    (output / "dataset_profile.md").write_text("\n".join(text))
    print(f"Reports written to {output}; elapsed {result['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--sample-size", type=int, default=2000)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    profile(args.dataset, args.output, args.sample_size)
