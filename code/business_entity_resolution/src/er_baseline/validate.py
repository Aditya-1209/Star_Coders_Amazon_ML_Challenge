"""Strict submission validation without retaining every candidate mapping."""

from collections import Counter
from itertools import zip_longest
from pathlib import Path
import time

from .data import read_records, read_tsv
from .parallel import atomic_json, fingerprint, now


def _id_list(value, source, valid_ids):
    ids = value.split(",") if value else []
    unique = set(ids)
    if len(ids) != len(unique):
        raise ValueError(f"Duplicate target ID for {source}")
    for target in unique:
        if not target.startswith(("S2-", "S3-")) or target not in valid_ids:
            raise ValueError(f"Unknown/invalid target ID for {source}: {target!r}")
    return unique


def validate_submission(anchors, targets, output, receipt=None):
    """Check all rows and IDs; memory scales with source IDs, not candidate pairs.

    Outputs must preserve Source 1 input order, a stricter condition than the
    organizer requires. This allows a complete row-by-row subset check.
    Target existence is checked against the raw supplied TSVs, not the index.
    """
    started = time.monotonic()
    output = Path(output)
    matching, candidates = output / "matching_results.tsv", output / "candidate_pairs.tsv"
    source_paths = [Path(anchors), *map(Path, targets)]
    source_fingerprints = [fingerprint(p) for p in source_paths]
    output_fingerprints = {p.name: fingerprint(p, True) for p in (matching, candidates)}
    valid_ids = set()
    for target_path in targets:
        count = 0
        for record in read_records(target_path):
            if not record.entity_id.startswith(("S2-", "S3-")) or record.entity_id in valid_ids:
                raise ValueError(f"Invalid/duplicate target source ID: {record.entity_id}")
            valid_ids.add(record.entity_id)
            count += 1
        print(f"Loaded {count:,} valid target IDs from {Path(target_path).name}", flush=True)
    seen = set()
    country_rows = Counter()
    counts = Counter()
    rows = zip_longest(
        read_records(anchors),
        read_tsv(matching, ["source1_entity_id", "matched_entity_ids"]),
        read_tsv(candidates, ["source1_entity_id", "candidate_entity_ids"]),
    )
    for number, (record, matched, candidate) in enumerate(rows, 1):
        if record is None or matched is None or candidate is None:
            raise ValueError(f"Output row count mismatch at row {number}")
        source = record.entity_id
        if not source.startswith("S1-") or source in seen:
            raise ValueError(f"Invalid/duplicate Source 1 ID: {source}")
        if matched[0] != source or candidate[0] != source:
            raise ValueError(f"Wrong/missing/duplicate Source 1 output row at row {number}: expected {source}")
        seen.add(source)
        mids = _id_list(matched[1], source, valid_ids)
        cids = _id_list(candidate[1], source, valid_ids)
        if not mids <= cids:
            raise ValueError(f"Matches outside candidate set for {source}")
        country_rows[record.country] += 1
        counts.update(rows=1, matched_pairs=len(mids), candidate_pairs=len(cids),
                      empty_matches=int(not mids), empty_candidates=int(not cids))
        if number % 100000 == 0:
            print(f"Validated {number:,} Source 1 rows", flush=True)
    if not seen:
        raise ValueError("Source 1 is empty")
    if [fingerprint(p) for p in source_paths] != source_fingerprints:
        raise ValueError("Source files changed during validation")
    if {p.name: fingerprint(p, True) for p in (matching, candidates)} != output_fingerprints:
        raise ValueError("Outputs changed during validation")
    result = {
        "status": "passed", "validator": "strict_streaming", "checked_at": now(),
        **counts, "country_rows": dict(country_rows), "valid_target_ids": len(valid_ids),
        "checks": ["exact headers", "all required rows in input order", "unique source IDs",
                   "unique target IDs within each list", "target prefixes and existence in raw inputs",
                   "matches are a subset of candidates", "empty lists retained", "unchanged input/output files"],
        "seconds": round(time.monotonic() - started, 3),
        "source_files": source_fingerprints, "output_files": output_fingerprints,
    }
    if receipt:
        atomic_json(receipt, result)
    print(f"PASS: validated {counts['rows']:,} rows and {counts['candidate_pairs']:,} candidate pairs", flush=True)
    return result
