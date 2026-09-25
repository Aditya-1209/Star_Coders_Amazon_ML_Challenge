"""Package only completed, validated predictions with reproducible source."""

import hashlib
import json
from pathlib import Path
import re
import zipfile

from .parallel import FILES, atomic_json, digest, fingerprint, now


def package_submission(output, model, code, documentation, archive):
    output, model, code, documentation, archive = map(Path, (output, model, code, documentation, archive))
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite submission: {archive}")
    validation = json.loads((output / "validation.json").read_text())
    inference = json.loads((output / "inference_complete.json").read_text())
    manifest = json.loads((output / "inference_manifest.json").read_text())
    if validation["status"] != "passed" or inference["status"] != "complete":
        raise ValueError("Packaging requires complete inference and successful validation")
    if validation["rows"] != inference["total_rows"] or validation["rows"] != inference["completed_rows"]:
        raise ValueError("Validation and inference row counts differ")
    for key in ("candidate_pairs", "matched_pairs", "empty_matches"):
        if validation[key] != inference[key]:
            raise ValueError(f"Validation and inference disagree: {key}")
    for name in FILES:
        actual = digest(output / name)
        if actual != validation["output_files"][name]["sha256"] or actual != inference["output_files"][name]["sha256"]:
            raise ValueError(f"Validated output has changed: {name}")
    for name, key in (("model.cbm", "model"), ("config.json", "config")):
        if digest(model / name) != manifest["identity"][key]["sha256"]:
            raise ValueError(f"Model artifact differs from inference: {name}")
    for name, expected in manifest["identity"]["code"].items():
        if digest(code / "src/er_baseline" / name) != expected:
            raise ValueError(f"Inference code changed: {name}")

    final_results = (
        f"Full test inference and strict streaming validation completed successfully.\n\n"
        f"- Source 1 rows in each output: **{validation['rows']:,}**.\n"
        f"- Candidate pairs scored: **{validation['candidate_pairs']:,}**.\n"
        f"- Final matched pairs: **{validation['matched_pairs']:,}**.\n"
        f"- Empty final-match rows: **{validation['empty_matches']:,}**.\n"
        f"- Valid target IDs checked against raw input: **{validation['valid_target_ids']:,}**.\n"
        f"- Country row counts: {json.dumps(validation['country_rows'], sort_keys=True)}.\n"
        f"- Validation completed at {validation['checked_at']}.\n"
    )
    doc = documentation.read_text(encoding="utf-8")
    doc, replacements = re.subn(r"<!-- FULL_TEST_RESULTS -->.*?<!-- END_FULL_TEST_RESULTS -->",
                                lambda _: final_results, doc, flags=re.S)
    if replacements != 1:
        raise ValueError("Methodology must contain exactly one full-test results block")
    archive.parent.mkdir(parents=True, exist_ok=True)
    prefix = "code/business_entity_resolution/"
    entries = {f"output/{name}": output / name for name in FILES}
    for directory in ("src", "tests"):
        for path in sorted((code / directory).rglob("*.py")):
            entries[prefix + path.relative_to(code).as_posix()] = path
    entries[prefix + "README.md"] = code / "SUBMISSION_README.md"
    entries[prefix + "requirements.txt"] = code / "requirements.txt"
    for name in ("model.cbm", "config.json", "metrics.json"):
        entries[prefix + "model/" + name] = model / name
    before = {name: fingerprint(path, True) for name, path in entries.items()}
    receipt = {key: validation[key] for key in (
        "status", "validator", "checked_at", "rows", "candidate_pairs", "matched_pairs",
        "empty_matches", "empty_candidates", "country_rows", "valid_target_ids", "checks")}
    receipt["output_sha256"] = {name: validation["output_files"][name]["sha256"] for name in FILES}
    contents = {
        "Documentation_template.md": doc.encode(),
        "output/validation.json": (json.dumps(receipt, indent=2) + "\n").encode(),
    }
    temporary = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as zipped:
        for name, path in entries.items():
            zipped.write(path, name)
        for name, data in contents.items():
            zipped.writestr(name, data)
    expected_hashes = {name: value["sha256"] for name, value in before.items()}
    expected_hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in contents.items()})
    with zipfile.ZipFile(temporary) as zipped:
        if len(zipped.namelist()) != len(expected_hashes) or set(zipped.namelist()) != set(expected_hashes):
            raise ValueError("Unexpected archive contents")
        for name, expected in expected_hashes.items():
            value = hashlib.sha256()
            with zipped.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected:
                raise ValueError(f"Archive integrity mismatch: {name}")
    if {name: fingerprint(path, True) for name, path in entries.items()} != before:
        raise ValueError("Package inputs changed during packaging")
    temporary.replace(archive)
    result = {"status": "complete", "created_at": now(), "archive": fingerprint(archive, True),
              "rows": validation["rows"], "files": expected_hashes}
    atomic_json(archive.with_suffix(".manifest.json"), result)
    print(f"Submission created and verified: {archive}", flush=True)
    return result
