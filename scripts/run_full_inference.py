#!/usr/bin/env python3
"""Extract data, build the index, predict, validate, and package a submission."""

import argparse
from pathlib import Path
import os
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    raise SystemExit("Run this project inside WSL2 Ubuntu, not native Windows Python. See README.md.")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from er_baseline.parallel import atomic_json, now, output_lock, predict_parallel
from er_baseline.retrieval import INDEX_VERSION, build_index
from er_baseline.submission import package_submission
from er_baseline.validate import validate_submission
from extract_dataset import extract_dataset


def ensure_index(sources, index):
    expected = [str(path.resolve()) for path in sources]
    if index.exists():
        with sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True) as database:
            metadata = {key: json.loads(value) for key, value in database.execute("SELECT key,value FROM metadata")}
        if metadata.get("version") != INDEX_VERSION or metadata.get("source_paths") != expected:
            raise ValueError("Existing index does not match this dataset/version; use a new --work directory")
        if any(path.stat().st_mtime_ns > index.stat().st_mtime_ns for path in sources):
            raise ValueError("Dataset is newer than the existing index; rebuild in a new --work directory")
        return
    temporary = index.with_name(index.stem + ".building.sqlite")
    # These files belong solely to an interrupted build under the pipeline lock.
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(temporary) + suffix)
        if path.exists():
            path.unlink()
    build_index(sources, temporary)
    temporary.replace(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive", type=Path, help="Organizer dataset ZIP; extracted safely when needed")
    parser.add_argument("--dataset", type=Path, default=ROOT / "student_resource/dataset")
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--model", type=Path, help="Defaults to the frozen model included in this repository")
    parser.add_argument("--submission", type=Path, default=ROOT / "submissions/Star_Coders_submission.zip")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("Workers and chunk size must be positive")
    args.work.mkdir(parents=True, exist_ok=True)
    with output_lock(args.work / ".pipeline"):
        run(args)


def run(args):
    status_path = args.work / "full_inference_status.json"
    started = now()

    def status(stage, **fields):
        atomic_json(status_path, {"stage": stage, "pid": os.getpid(), "started_at": started,
                                 "updated_at": now(), **fields})

    data = args.dataset / "test"
    output = args.output
    model = args.model or ROOT / "models/baseline_full_corpus"
    if args.resume and args.model is None and (output / "inference_manifest.json").exists():
        # Preserve the original local model path when resuming the earlier Mac run.
        previous = json.loads((output / "inference_manifest.json").read_text())
        model = Path(previous["identity"]["model"]["path"]).parent
    try:
        if args.archive:
            status("extracting")
            extract_dataset(args.archive, args.dataset)
        sources = [data / "test_source2.tsv", data / "test_source3.tsv"]
        for path in [data / "test_source1.tsv", *sources, model / "model.cbm", model / "config.json"]:
            if not path.is_file():
                raise FileNotFoundError(f"Required file missing: {path}. Supply --archive or --dataset; see README.md.")
        if (output / "inference_manifest.json").exists() and not args.resume:
            raise FileExistsError("An inference run already exists; use --resume after stopping it")
        atomic_json(args.work / "run_arguments.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        status("indexing")
        index = args.work / "test_search.sqlite"
        ensure_index(sources, index)
        status("inference", workers=args.workers)
        inference = predict_parallel(data / "test_source1.tsv", index, model, output,
                                     args.workers, args.chunk_size, args.resume)
        if inference["status"] != "complete":
            raise RuntimeError("Inference did not complete")
        status("validation", rows=inference["completed_rows"])
        validation = validate_submission(data / "test_source1.tsv",
                                         [data / "test_source2.tsv", data / "test_source3.tsv"],
                                         output, output / "validation.json")
        for key in ("candidate_pairs", "matched_pairs", "empty_matches"):
            if validation[key] != inference[key]:
                raise ValueError(f"Inference/validation count disagreement: {key}")
        if validation["rows"] != inference["completed_rows"]:
            raise ValueError("Inference/validation row count disagreement")
        for name, details in validation["output_files"].items():
            if details["sha256"] != inference["output_files"][name]["sha256"]:
                raise ValueError(f"Output changed before validation: {name}")
        status("packaging", rows=validation["rows"])
        package = package_submission(output, model, ROOT / "code/business_entity_resolution",
                                     ROOT / "docs/legacy/Documentation_template_baseline.md", args.submission)
        status("complete", validation=str(output / "validation.json"), rows=validation["rows"],
               output_files=validation["output_files"], submission=package["archive"])
        print("Full test inference, validation, and submission packaging completed successfully.", flush=True)
    except BaseException as error:
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error) or type(error).__name__)
        if isinstance(error, KeyboardInterrupt):
            print("Stopped. Completed checkpoints are preserved. Repeat with --resume to continue.", flush=True)
            raise SystemExit(130)
        raise


if __name__ == "__main__":
    main()
