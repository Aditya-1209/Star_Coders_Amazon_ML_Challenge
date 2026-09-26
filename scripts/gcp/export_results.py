#!/usr/bin/env python3
"""Export progress/results to GCS; no dataset or large feature caches are uploaded."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ["run.json", "result.json", "result.md", "metrics.json", "selection.json", "crossfit.json",
           "final_models.json", "expand_train.json", "expand_test.json", "base/shift_check.json",
           "base/retrieval_audit.json", "base/models/metrics.json", "base/models/stage3_metrics.json"]


def selected_files(root, full):
    work = root / "work/r9"
    items = [(work / name, Path("work") / name) for name in REPORTS]
    items += [(p, Path("work/logs") / p.name) for p in (work / "logs").glob("*.log")]
    items += [(root / "work/service.log", Path("service.log")),
              (Path("/var/log/r9-bootstrap.log"), Path("bootstrap.log"))]
    if full:
        for directory in ("models", "base/models"):
            items += [(p, Path("work") / directory / p.name) for p in (work / directory).glob("*.json")]
        for directory in ("", "baseline", "baseline/stage2"):
            for name in ("matching_results.tsv", "candidate_pairs.tsv"):
                items.append((root / "output/r9" / directory / name, Path("output") / directory / name))
    return {destination: source for source, destination in items if source.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    destination = os.environ["R9_RESULTS"].rstrip("/")
    if not destination.startswith("gs://"):
        raise ValueError("R9_RESULTS must be a Cloud Storage URI")
    with (ROOT / "work/export.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix="r9-export-") as temporary:
            folder = Path(temporary)
            for relative, source in selected_files(ROOT, args.full).items():
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(folder), destination], check=True)
            # No deletion flag: a periodic report export cannot remove completed models/output.


if __name__ == "__main__":
    main()
