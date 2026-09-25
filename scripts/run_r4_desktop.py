"""Sequential desktop workflow. Standard library only; no ML imports here."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "desktop_13900k_3060.json"
STEP_NAMES = ("preflight", "checks", "translit", "prepare_train", "block_train", "features_train",
              "train", "stage3_train", "prepare_test", "block_test", "features_test",
              "predict", "stage3_predict", "validate")
DEPENDENCIES = {
    "prepare_train": {"translit"}, "block_train": {"prepare_train"}, "features_train": {"block_train"},
    "train": {"features_train"}, "stage3_train": {"train"},
    "prepare_test": {"translit"}, "block_test": {"prepare_test"}, "features_test": {"block_test"},
    "predict": {"train", "features_test"}, "stage3_predict": {"stage3_train", "predict"},
    "validate": {"stage3_predict"},
}


def affected_steps(start: str) -> set[str]:
    affected = {start}
    for name in STEP_NAMES:
        if DEPENDENCIES.get(name, set()) & affected:
            affected.add(name)
    return affected


def build_steps(profile, work: Path, output: Path, dataset: Path, validator: Path):
    model = work / "models"
    def module(name, *options):
        return [sys.executable, "-m", "er_v2." + name, *map(str, options)]
    common = ["--work", work]
    train = [*common, "--model-dir", model, "--dataset", dataset, "--device", profile["train_device"],
             "--predict-device", profile["predict_device"], "--threads", profile["train_threads"],
             "--batch-rows", profile["predict_batch_rows"], "--matrix-batch-rows", profile["matrix_batch_rows"],
             "--hist-cache-nodes", profile["hist_cache_nodes"]]
    graph = ["--support-anchors", profile["support_anchors"], "--block-chunk", profile["block_chunk"],
             "--country-partition"]
    steps = {
        "preflight": module("preflight"),
        "checks": [sys.executable, "-m", "unittest", "discover", "-s", "tests_v2", "-v"],
        "translit": module("translit", "--dataset", dataset, "--out", model / "translit.json"),
        "train": module("train", *train),
        "stage3_train": module("stage3", *train, *graph, "--split", "train"),
        "predict": module("predict", *common, "--model-dir", model, "--output", str(output) + "-stage2",
                          "--device", profile["predict_device"], "--threads", profile["predict_threads"],
                          "--batch-rows", profile["predict_batch_rows"]),
        "stage3_predict": module("stage3", *common, "--model-dir", model, "--output", output, "--split", "test",
                                 "--device", profile["predict_device"], "--threads", profile["predict_threads"],
                                 "--batch-rows", profile["predict_batch_rows"], *graph),
        "validate": [sys.executable, str(validator), "--matching", str(output / "matching_results.tsv"),
                     "--candidate", str(output / "candidate_pairs.tsv"), "--test-dir", str(dataset / "test"), "--check-ids"],
    }
    for split in ("train", "test"):
        steps["prepare_" + split] = module("prepare", *common, "--dataset", dataset, "--splits", split,
                                          "--workers", profile["normalize_workers"],
                                          "--buffer-rows", profile["normalize_buffer_rows"],
                                          "--translit", model / "translit.json")
        steps["block_" + split] = module("run_block", *common, "--split", split, "--country-partition",
                                        "--top-k", profile["top_k"], "--name-k", profile["name_k"],
                                        "--address-k", profile["address_k"], "--block-chunk", profile["block_chunk"])
        steps["features_" + split] = module("run_features", *common, "--dataset", dataset,
                                           "--split", split, "--country-partition",
                                           "--shard-pairs", profile["feature_shard_pairs"],
                                           "--workers", profile["feature_workers"])
    return [(name, steps[name]) for name in STEP_NAMES]


def run_signature(dataset: Path, validator: Path, work: Path, output: Path) -> str:
    digest = hashlib.sha256(json.dumps([str(dataset), str(validator), str(work), str(output), sys.executable]).encode())
    code = list((ROOT / "code/business_entity_resolution/src/er_v2").glob("*.py"))
    code += list((ROOT / "tests_v2").glob("*.py"))
    code += [Path(__file__), ROOT / "code/business_entity_resolution/requirements_v2.txt", validator]
    for path in sorted(code):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    for split in ("train", "test"):
        filenames = [f"{split}_source{index}.tsv" for index in (1, 2, 3)]
        if split == "train":
            filenames.append("train_ground_truth.tsv")
        for name in filenames:
            path = dataset / split / name
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def output_snapshot(step: str, work: Path, output: Path) -> list[dict]:
    """Detect missing/replaced outputs before skipping a completed stage."""
    models = work / "models"
    paths = {
        "preflight": [], "checks": [], "translit": [models / "translit.json"],
        "train": [models / name for name in ("stage1.json", "stage1_b.json", "stage2.json", "metrics.json")]
                 + [work / "stage2_train.parquet", work / "eval_preds.parquet"],
        "stage3_train": [models / "stage3.json", models / "stage3_metrics.json",
                         work / "stage3_train.parquet", work / "eval_preds_stage3.parquet"],
        "predict": [work / "test_preds.parquet"] + [Path(str(output) + "-stage2") / name
                    for name in ("matching_results.tsv", "candidate_pairs.tsv")],
        "stage3_predict": [work / "test_preds_stage3.parquet", output / "matching_results.tsv", output / "candidate_pairs.tsv"],
        "validate": [output / "matching_results.tsv", output / "candidate_pairs.tsv"],
    }
    for split in ("train", "test"):
        paths["prepare_" + split] = [work / "norm" / f"{split}_source{i}.parquet" for i in (1, 2, 3)]
        paths["block_" + split] = [work / f"cands_{split}.parquet", work / f"cands_{split}.json"]
        folder = work / f"feats_{split}"
        paths["features_" + split] = [folder / "manifest.json", *sorted(folder.glob("part_*.parquet"))]
        if step == "features_" + split and (folder / "_INCOMPLETE").exists():
            raise ValueError(f"Incomplete feature folder: {folder}")
    return [{"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths[step]]


def verify_completed(name, command, record, work, output):
    if command != record.get("command"):
        raise SystemExit(f"Settings changed for {name}; restart with --from-step {name}.")
    try:
        actual = output_snapshot(name, work, output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cannot resume {name}: {error}. Restore its outputs or use --from-step {name}.") from error
    if actual != record.get("outputs"):
        raise SystemExit(f"Completed outputs changed for {name}; restore them or use --from-step {name}.")


def run_step(command, environment, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--work", type=Path, default=ROOT / "work/desktop_13900k_3060")
    parser.add_argument("--output", type=Path, default=ROOT / "output/desktop_13900k_3060")
    parser.add_argument("--dataset", type=Path, default=ROOT / "student_resource/dataset")
    parser.add_argument("--validator", type=Path, default=ROOT / "student_resource/utils/validate_submission.py")
    parser.add_argument("--plan", action="store_true", help="print commands without running or writing anything")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", action="store_true", help="skip only steps recorded as successfully completed")
    start.add_argument("--from-step", choices=STEP_NAMES)
    parser.add_argument("--to-step", choices=STEP_NAMES, default="validate")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    work, output, dataset, validator = [p.resolve() for p in (args.work, args.output, args.dataset, args.validator)]
    steps = build_steps(profile, work, output, dataset, validator)
    first = STEP_NAMES.index(args.from_step or "preflight")
    last = STEP_NAMES.index(args.to_step)
    if first > last:
        parser.error("--from-step must precede --to-step")
    if args.plan:
        print(profile["name"])
        for name, command in steps[first:last + 1]:
            print(f"{name}: {subprocess.list2cmdline(command)}")
        return
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12 and the project's requirements_v2.txt.")
    try:
        signature = run_signature(dataset, validator, work, output)
    except FileNotFoundError as error:
        raise SystemExit(f"Missing input: {error.filename}. Extract the organizer ZIP first.") from error
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "desktop_run.json"
    state = {"signature": signature, "completed": {}, "profile": profile}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise SystemExit("Code, paths or dataset changed. Use a fresh --work and --output directory.")
        if not args.resume and args.from_step is None:
            raise SystemExit("This workspace has a run record. Use --resume or an explicit --from-step.")
    elif args.resume:
        raise SystemExit("No run record exists; start without --resume.")
    if args.from_step is not None:
        for name, command in steps[:first]:
            if name in state["completed"]:
                verify_completed(name, command, state["completed"][name], work, output)
        for name in affected_steps(args.from_step):
            state["completed"].pop(name, None)
    state["profile"] = profile
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / "code/business_entity_resolution/src"), PYTHONUTF8="1",
                       PYTHONUNBUFFERED="1", POLARS_MAX_THREADS=str(profile["polars_threads"]),
                       OMP_NUM_THREADS=str(profile["train_threads"]), OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    print(f"{profile['name']} — free workspace disk: {shutil.disk_usage(work).free / 2**30:.1f} GiB", flush=True)
    for name, command in steps[first:last + 1]:
        if (args.resume or args.from_step) and name in state["completed"] and name != "preflight":
            verify_completed(name, command, state["completed"][name], work, output)
            print(f"Skipping completed step: {name}", flush=True)
            continue
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = logs / f"{name}-{stamp}.log"
        state.update(status="running", step=name, log=str(log))
        write_state(state_path, state)
        print(f"\nStarting {name}; log: {log}", flush=True)
        begin = time.monotonic()
        try:
            code = run_step(command, environment, log)
        except KeyboardInterrupt:
            state.update(status="interrupted")
            write_state(state_path, state)
            raise SystemExit(130)
        if code:
            state.update(status="failed", exit_code=code)
            write_state(state_path, state)
            raise SystemExit(f"{name} failed (exit {code}); see {log}. Later steps were not started.")
        try:
            snapshot = output_snapshot(name, work, output)
        except (OSError, ValueError) as error:
            state.update(status="failed", error=str(error))
            write_state(state_path, state)
            raise SystemExit(f"{name} returned success but its expected outputs are missing/incomplete: {error}") from error
        state["completed"][name] = {"seconds": round(time.monotonic() - begin, 2), "log": str(log),
                                     "command": command, "outputs": snapshot}
        state.update(status="step_complete")
        state.pop("exit_code", None)
        state.pop("error", None)
        write_state(state_path, state)
    state.update(status="complete" if args.to_step == "validate" and "validate" in state["completed"]
                 else "stopped_after_requested_step")
    write_state(state_path, state)
    print(f"Finished through {args.to_step}. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
