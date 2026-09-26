#!/usr/bin/env python3
"""Extract challenge data and its official validator from the organizer ZIP."""
import argparse
from pathlib import Path
import zipfile

from extract_dataset import extract_dataset


def prepare(archive: Path, destination: Path):
    with zipfile.ZipFile(archive) as zipped:
        allowed = {"student_resource/utils/validate_submission.py", "utils/validate_submission.py"}
        matches = [m for m in zipped.infolist() if m.filename in allowed]
        if len(matches) != 1:
            raise ValueError("Expected one official utils/validate_submission.py in the organizer ZIP")
        if matches[0].file_size > 2 * 1024 * 1024:
            raise ValueError("Unexpected validator size")
        validator = zipped.read(matches[0])
    extract_dataset(archive, destination / "dataset")
    path = destination / "utils/validate_submission.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != validator:
        raise FileExistsError(f"Existing official validator differs: {path}")
    path.write_bytes(validator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("student_resource"))
    args = parser.parse_args()
    prepare(args.archive, args.destination)
