"""Small, side-effect-free helpers shared by the v2/stage-3 commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def feature_parts(folder: Path) -> list[Path]:
    """Accept legacy shards, but reject incomplete or altered manifested runs."""
    folder = Path(folder)
    if (folder / "_INCOMPLETE").exists():
        raise ValueError(f"Feature generation is incomplete in {folder}; rebuild in a fresh work directory")
    parts = sorted(folder.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No feature shards in {folder}; run er_v2.run_features first")
    manifest = folder / "manifest.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text(encoding="utf-8"))["parts"]
        actual = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in parts]
        if actual != expected:
            raise ValueError(f"Feature shards changed or stale shards were added in {folder}")
    return parts
