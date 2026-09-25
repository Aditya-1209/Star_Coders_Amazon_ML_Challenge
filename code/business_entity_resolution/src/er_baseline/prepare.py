"""Seeded, grouped development data with labeled positives and random distractors."""

from collections import defaultdict
import json
from pathlib import Path
import random
import time

from .data import SOURCE_HEADER, TRUTH_HEADER, read_records, read_tsv, write_tsv


def group_split(anchors, truth, seed):
    """Keep anchors sharing any labeled target in the same partition."""
    parents = {entity_id: entity_id for entity_id in anchors}

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    target_owner = {}
    for source_id, targets in truth.items():
        for target_id in targets:
            if target_id in target_owner:
                parents[find(source_id)] = find(target_owner[target_id])
            else:
                target_owner[target_id] = source_id
    groups = defaultdict(list)
    for entity_id in sorted(anchors):
        groups[find(entity_id)].append(entity_id)
    strata = defaultdict(list)
    for ids in groups.values():
        first = ids[0]
        strata[(anchors[first].country, not truth[first])].append(ids)
    rng = random.Random(seed)
    partitions = {}
    for key in sorted(strata):
        groups_in_stratum = strata[key]
        rng.shuffle(groups_in_stratum)
        ntrain = int(len(groups_in_stratum) * 0.7)
        ntune = int(len(groups_in_stratum) * 0.15)
        for i, ids in enumerate(groups_in_stratum):
            partition = "train" if i < ntrain else "tune" if i < ntrain + ntune else "holdout"
            for entity_id in ids:
                partitions[entity_id] = partition
    owners = {}
    for source_id, targets in truth.items():
        for target in targets:
            if target in owners and owners[target] != partitions[source_id]:
                raise AssertionError("Shared labeled target crosses partitions")
            owners[target] = partitions[source_id]
    return partitions


def prepare(dataset, output, anchors_count=6000, background_count=200000, seed=42):
    start = time.monotonic()
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite development data: {output}")
    rng = random.Random(seed)
    sample = []
    for n, row in enumerate(read_tsv(dataset / "train/train_ground_truth.tsv", TRUTH_HEADER), 1):
        if len(sample) < anchors_count:
            sample.append(row)
        else:
            index = rng.randrange(n)
            if index < anchors_count:
                sample[index] = row
    truth = {s: set(t.split(",")) if t else set() for s, t in sample}
    if len(truth) != len(sample):
        raise ValueError("Duplicate anchor IDs in sample")
    wanted = set().union(*truth.values())
    anchors = {}
    for record in read_records(dataset / "train/train_source1.tsv"):
        if record.entity_id in truth:
            if record.entity_id in anchors:
                raise ValueError(f"Duplicate anchor: {record.entity_id}")
            anchors[record.entity_id] = record
    if set(anchors) != set(truth):
        raise ValueError("Sampled ground-truth anchor absent from source file")
    print(f"Sampled {len(anchors):,} anchors and {len(wanted):,} unique labeled targets", flush=True)
    positives, background = {}, []
    seen_background = total_targets = 0
    for source in (2, 3):
        for record in read_records(dataset / f"train/train_source{source}.tsv"):
            total_targets += 1
            if record.entity_id in wanted:
                if record.entity_id in positives:
                    raise ValueError(f"Duplicate positive target: {record.entity_id}")
                positives[record.entity_id] = record
                continue
            seen_background += 1
            if len(background) < background_count:
                background.append(record)
            else:
                index = rng.randrange(seen_background)
                if index < background_count:
                    background[index] = record
        print(f"Scanned source {source}; {len(positives):,} positives found", flush=True)
    if set(positives) != wanted:
        raise ValueError(f"Missing labeled targets: {len(wanted - set(positives))}")
    partitions = group_split(anchors, truth, seed)
    output.mkdir(parents=True)
    write_tsv(output / "anchors.tsv", SOURCE_HEADER, (anchors[s].values() for s in sorted(anchors)))
    write_tsv(output / "ground_truth.tsv", TRUTH_HEADER, ((s, ",".join(sorted(truth[s]))) for s in sorted(truth)))
    write_tsv(output / "splits.tsv", ["source1_entity_id", "split"], sorted(partitions.items()))
    targets = list(positives.values()) + background
    if len({r.entity_id for r in targets}) != len(targets):
        raise ValueError("Duplicate IDs in selected target pool")
    rng.shuffle(targets)
    write_tsv(output / "targets.tsv", SOURCE_HEADER, (record.values() for record in targets))
    split_summary = {}
    for split in ("train", "tune", "holdout"):
        ids = [s for s in truth if partitions[s] == split]
        split_summary[split] = {
            "anchors": len(ids), "true_pairs": sum(len(truth[s]) for s in ids),
            "singletons": sum(not truth[s] for s in ids),
            "countries": {c: sum(anchors[s].country == c for s in ids) for c in sorted({a.country for a in anchors.values()})},
        }
    manifest = {
        "seed": seed, "anchors": len(anchors), "target_pool": len(targets),
        "labeled_positive_targets": len(positives), "background_targets": len(background),
        "full_training_target_pool": total_targets, "splits": split_summary,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "limitations": [
            "The reduced target pool includes all sampled labels plus uniformly sampled distractors.",
            "This is easier than full-corpus retrieval and is not a leaderboard score estimate.",
            "Partitions share an unlabeled search index, but no anchor or labeled positive target crosses partitions.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

