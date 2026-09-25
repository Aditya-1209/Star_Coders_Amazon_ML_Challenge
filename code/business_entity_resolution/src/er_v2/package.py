"""Build the final submission ZIP in the layout the challenge requires.

<team>_submission.zip
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src/er_v2, models/v2, README.md, requirements.txt}
  Documentation_template.md
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="output")
    ap.add_argument("--model-dir", default="models/v2")
    ap.add_argument("--code-root", default="code/business_entity_resolution")
    ap.add_argument("--doc", default="Documentation_template.md")
    ap.add_argument("--zip", default="submissions/Star_Coders_submission.zip")
    args = ap.parse_args()
    code = Path(args.code_root)
    dest = Path(args.zip)
    dest.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, str]] = [
        (Path(args.output) / "matching_results.tsv", "output/matching_results.tsv"),
        (Path(args.output) / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        (Path(args.doc), "Documentation_template.md"),
        (code / "README_v2.md", "code/business_entity_resolution/README.md"),
        (code / "requirements_v2.txt", "code/business_entity_resolution/requirements.txt"),
    ]
    for p in sorted((code / "src" / "er_v2").glob("*.py")):
        entries.append((p, f"code/business_entity_resolution/src/er_v2/{p.name}"))
    for p in sorted(Path(args.model_dir).iterdir()):
        if p.is_file():
            entries.append((p, f"code/business_entity_resolution/models/v2/{p.name}"))
    missing = [str(p) for p, _ in entries if not p.is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in entries:
            zf.write(src, arc)
    with zipfile.ZipFile(dest) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"corrupt member {bad}")
        names = zf.namelist()
    print(f"{dest}: {len(names)} files, {dest.stat().st_size / 1e6:.1f} MB")
    for n in names:
        print("  " + n)


if __name__ == "__main__":
    main()
