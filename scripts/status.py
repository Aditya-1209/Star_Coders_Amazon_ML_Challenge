#!/usr/bin/env python3
"""Show the local pipeline stage and saved inference progress."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, default=ROOT / "artifacts")
parser.add_argument("--output", type=Path, default=ROOT / "output")
args = parser.parse_args()
state = args.work / "full_inference_status.json"
if not state.exists():
    raise SystemExit("No run has been started in this work directory.")
status = json.loads(state.read_text())
print(f"Stage: {status['stage']} (updated {status['updated_at']})")
progress = args.output / "inference_progress.json"
if progress.exists():
    data = json.loads(progress.read_text())
    rows, total = data["completed_rows"], data["total_rows"]
    print(f"Saved progress: {rows:,} / {total:,} businesses ({100 * rows / total:.2f}%)")
    print(f"Last throughput: {data['rows_per_second']:.1f} businesses/second")
    if status["stage"] == "inference" and data.get("estimated_remaining_seconds"):
        print(f"Estimated active processing remaining: {data['estimated_remaining_seconds'] / 3600:.2f} hours")
    print(f"Progress last updated: {data['updated_at']}")
if status.get("error"):
    print(f"Last error: {status['error']}")
if status.get("submission"):
    print(f"Submission: {status['submission']['path']}")
print("Progress files describe the last saved state; a stale timestamp is not proof the process is still running.")
