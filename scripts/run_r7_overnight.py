#!/usr/bin/env python3
"""Sequential, logged accuracy experiment. One heavy process at a time.

Run under caffeinate on macOS. This runner never pushes to GitHub or submits to
Amazon. Result reporting distinguishes local holdout from leaderboard scores.
"""
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
STEPS = ["checks", "translit", "prepare", "block_train", "retrieval_audit", "features_train",
         "block_test", "features_test", "shift_check", "train", "stage3_train", "predict",
         "stage3_test", "validate", "report"]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def code_hash():
    files = sorted((ROOT / "code/business_entity_resolution/src/er_v2").glob("*.py"))
    files += sorted((ROOT / "scripts/analysis").glob("*.py"))
    files += [Path(__file__), ROOT / "code/business_entity_resolution/requirements_v2.txt"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot(paths):
    files = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected output missing: {path}")
        if path.is_dir():
            if (path / "_INCOMPLETE").exists():
                raise RuntimeError(f"Incomplete output: {path}")
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            files.append(path)
    return [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]


def commands(args):
    # Keep the venv executable path: resolving its symlink bypasses pyvenv.cfg.
    py = str(Path(sys.executable).absolute())
    work, model, out = args.work, args.work / "models", args.output
    module = lambda name: [py, "-u", "-m", "er_v2." + name, "--work", str(work)]
    data = ["--dataset", str(args.dataset)]
    runtime = ["--model-dir", str(model), "--device", args.device, "--threads", str(args.threads), "--batch-rows", "100000"]
    gate_path = work / "shift_check.json"
    enhanced = not gate_path.exists() or json.loads(gate_path.read_text())["allow_enhanced"]
    accuracy = ["--feature-profile", "enhanced" if enhanced else "baseline", "--rounds", str(args.rounds), "--country-thresholds"]
    if enhanced:
        accuracy += ["--compare-baseline"]
    analysis = lambda name: [py, "-u", str(ROOT / "scripts/analysis" / name), "--work", str(work)]
    plan = {
        "checks": ([py, "-m", "unittest", "discover", "-s", "tests_v2", "-v"], []),
        "translit": ([py, "-u", "-m", "er_v2.translit", *data, "--out", str(work / "translit.json")], [work / "translit.json"]),
        "prepare": (module("prepare") + data + ["--translit", str(work / "translit.json"), "--workers", str(args.threads),
                                                    "--buffer-rows", "150000"], [work / "norm"]),
        "retrieval_audit": (analysis("r7_retrieval_audit.py") + data, [work / "retrieval_audit.json"]),
        "shift_check": (analysis("r7_shift_check.py") + ["--threads", str(args.threads)], [work / "shift_check.json"]),
        "train": (module("train") + data + runtime + accuracy,
                  [model / n for n in ("stage1.json", "stage1_b.json", "stage2.json", "metrics.json")]
                  + [work / "stage2_train.parquet", work / "eval_preds.parquet"]),
        "stage3_train": (module("stage3") + data + runtime + accuracy + ["--rounds", str(args.stage3_rounds), "--split", "train", "--support-anchors", "1000"],
                         [model / "stage3.json", model / "stage3_metrics.json", work / "eval_preds_stage3.parquet"]),
        "predict": (module("predict") + runtime + ["--output", str(out / "stage2")], [work / "test_preds.parquet", out / "stage2"]),
        "stage3_test": (module("stage3") + runtime + ["--split", "test", "--support-anchors", "1000", "--output", str(out)],
                        [out / "matching_results.tsv", out / "candidate_pairs.tsv", work / "test_preds_stage3.parquet"]),
        "validate": ([py, str(ROOT / "student_resource/utils/validate_submission.py"),
                      "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                      "--test-dir", str(args.dataset / "test"), "--check-ids"], []),
        "report": (analysis("r7_report.py") + ["--output", str(out)], [work / "result.json", work / "result.md"]),
    }
    for split in ("train", "test"):
        plan["block_" + split] = (module("run_block") + ["--split", split, "--phonetic-k", "8", "--block-chunk", "2000"],
                                 [work / f"cands_{split}.parquet", work / f"cands_{split}.json"])
        plan["features_" + split] = (module("run_features") + data + ["--split", split, "--enhanced", "--workers", str(args.threads),
                                        "--shard-pairs", "150000", "--overwrite"], [work / f"feats_{split}"])
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("work/r7_overnight"))
    ap.add_argument("--dataset", type=Path, default=Path("student_resource/dataset"))
    ap.add_argument("--output", type=Path, default=Path("output/r7_overnight"))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--stage3-rounds", type=int, default=2000)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--start", choices=STEPS, help="explicitly reuse prerequisite artifacts and start here")
    group.add_argument("--resume", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    for field in ("work", "dataset", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if args.threads < 1 or args.rounds < 1 or args.stage3_rounds < 1:
        ap.error("threads and rounds must be positive")
    if args.plan:
        for step, (cmd, _) in ((s, commands(args)[s]) for s in STEPS):
            print(step, subprocess.list2cmdline(cmd))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "models").mkdir(exist_ok=True)
    (args.work / "logs").mkdir(exist_ok=True)
    state_path = args.work / "overnight.json"
    lock = args.work / "overnight.lock"
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f"Runner {pid} already owns {args.work}")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        dataset_files = sorted(args.dataset.glob("*/*.tsv"))
        if len(dataset_files) < 7:
            raise FileNotFoundError("The six source TSVs and training ground truth are required")
        identity = {"code_sha256": code_hash(), "dataset": snapshot(dataset_files),
                    "device": args.device, "threads": args.threads, "rounds": args.rounds,
                    "stage3_rounds": args.stage3_rounds,
                    "output": str(args.output), "python": sys.executable}
        if args.resume:
            state = json.loads(state_path.read_text())
            if state["identity"] != identity:
                raise RuntimeError("Code, settings or data changed; inspect dependencies before using --start in a compatible workspace")
        else:
            if state_path.exists() and not args.start:
                raise RuntimeError("Existing run: use --resume or explicitly --start after reviewing prerequisites")
            state = {"identity": identity, "started": now(), "completed": {}, "reused_before": args.start}
        state.update(status="running", pid=os.getpid())
        env = {**os.environ, "PYTHONPATH": str(ROOT / "code/business_entity_resolution/src"),
               "POLARS_MAX_THREADS": str(args.threads), "OMP_NUM_THREADS": str(args.threads),
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1",
               "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore::DeprecationWarning"}
        start = STEPS.index(args.start) if args.start else 0
        if args.resume and state.get("reused_before"):
            start = STEPS.index(state["reused_before"])
        for step in STEPS[start:]:
            cmd, outputs = commands(args)[step]
            previous = state["completed"].get(step)
            if previous is not None:
                if previous["command"] != cmd or previous["outputs"] != snapshot(outputs):
                    raise RuntimeError(f"Completed step changed: {step}")
                continue
            if shutil.disk_usage(args.work).free < 3 * 1024 ** 3:
                raise RuntimeError("Less than 3 GB free; stop before starting the next stage")
            log_path = args.work / "logs" / f"{step}.log"
            state.update(step=step, step_started=now(), log=str(log_path), command=cmd)
            save(state_path, state)
            print(f"{now()} {step} -> {log_path}", flush=True)
            before = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = child.pid
                save(state_path, state)
                try:
                    while child.poll() is None:
                        time.sleep(5)
                        state["step_seconds"] = round(time.monotonic() - before, 1)
                        save(state_path, state)
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
            if child.returncode:
                raise RuntimeError(f"{step} exited {child.returncode}; inspect {log_path}")
            state["completed"][step] = {"command": cmd, "outputs": snapshot(outputs),
                                         "seconds": round(time.monotonic() - before, 1), "finished": now()}
            save(state_path, state)
        state.update(status="complete", finished=now(), child_pid=None)
        save(state_path, state)
        print(f"Complete: {args.work / 'result.md'}", flush=True)
    except BaseException as error:
        if "state" in locals():
            state.update(status="failed", error=str(error), finished=now())
            save(state_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
