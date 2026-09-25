#!/usr/bin/env python3
"""Extract only known dataset files, without trusting archive member paths."""

import argparse
from pathlib import Path
import shutil
import zipfile
import zlib

FILES = [f"{split}/{split}_source{source}.tsv" for split in ("train", "test") for source in (1, 2, 3)] + ["train/train_ground_truth.tsv"]


def extract_dataset(archive, dataset):
    archive, dataset = Path(archive), Path(dataset)
    with zipfile.ZipFile(archive) as zipped:
        selected = {}
        for member in zipped.infolist():
            for relative in FILES:
                if member.filename in ("student_resource/dataset/" + relative, "dataset/" + relative, relative):
                    if relative in selected:
                        raise ValueError(f"Duplicate dataset member: {relative}")
                    selected[relative] = member
        if set(selected) != set(FILES):
            raise ValueError(f"Dataset ZIP is missing: {sorted(set(FILES) - set(selected))}")
        required = sum(info.file_size for relative, info in selected.items() if not (dataset / relative).exists())
        dataset.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(dataset).free < required + 1024 ** 3:
            raise OSError("Not enough disk space to extract the dataset")
        for relative, member in selected.items():
            destination = dataset / relative
            if destination.exists():
                crc = 0
                with destination.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if destination.stat().st_size != member.file_size or crc != member.CRC:
                    raise FileExistsError(f"Existing file differs from the archive: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tsv.partial")
            with zipped.open(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            temporary.replace(destination)
            print(f"Extracted {relative}", flush=True)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("student_resource/dataset"))
    args = parser.parse_args()
    extract_dataset(args.archive, args.dataset)
