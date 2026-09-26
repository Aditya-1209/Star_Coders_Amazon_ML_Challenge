#!/usr/bin/env python3
"""Resume a sequential R9 experiment; launch nothing unless explicitly invoked.

Uses the R7 pipeline as a freshly rebuilt reference, followed by a second graph
round. Cached artifacts are accepted only from this runner with identical code,
data and model settings. --plan prints commands without loading ML libraries.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version as installed_version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from run_r7_overnight import commands as baseline_commands, snapshot

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["checks", "translit", "prepare", "block_train", "retrieval_audit", "features_train",
         "block_test", "features_test", "shift_check", "train", "stage3_train",
         "crossfit", "expand_train", "fit", "select", "evaluate", "predict", "stage3_test",
         "expand_test", "inference", "validate"]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def code_hash():
    paths = sorted((ROOT / "code/business_entity_resolution/src/er_v2").glob("*.py"))
    paths += sorted((ROOT / "scripts").rglob("*.py")) + sorted((ROOT / "tests_v2").glob("*.py"))
    paths += [ROOT / "code/business_entity_resolution/requirements_v2.txt"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_versions():
    versions = {}
    for line in (ROOT / "code/business_entity_resolution/requirements_v2.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        package, expected = line.strip().split("==")
        actual = installed_version(package)
        if actual != expected:
            raise RuntimeError(f"Dependency {package}: expected {expected}, installed {actual}; install requirements_v2.txt")
        versions[package] = actual
    return versions


def commands(args):
    base = args.work / "base"
    out_base = args.output / "baseline"
    settings = SimpleNamespace(work=base, output=out_base, dataset=args.dataset,
                               device=args.device, threads=args.threads,
                               rounds=args.base_rounds, stage3_rounds=args.base_graph_rounds)
    plan = baseline_commands(settings)
    plan.pop("report")
    # Keep graph-training labels out of supervised normalization for cross-fitting.
    plan["translit"][0].extend(["--learn-folds", "0", "1", "8", "9"])
    suffix = "".join(f"_no{country}" for country in args.exclude_country)
    if args.exclude_country:
        for step in ("translit", "train", "stage3_train"):
            plan[step][0].extend(["--exclude-country", *args.exclude_country])
        for step in ("train", "stage3_train"):
            cmd, outputs = plan[step]
            outputs = [p.with_name(p.stem + suffix + p.suffix)
                       if p.name in ("stage2_train.parquet", "eval_preds.parquet") else p for p in outputs]
            plan[step] = cmd, outputs
    plan["stage3_train"][1].append(base / f"stage3_train{suffix}.parquet")
    plan["stage3_test"][1].append(base / "stage3_test.parquet")
    py = str(Path(sys.executable).absolute())
    common = ["--work", str(args.work), "--base", str(base), "--dataset", str(args.dataset),
              "--output", str(args.output), "--baseline-output", str(out_base),
              "--device", args.device, "--threads", str(args.threads), "--rounds", str(args.rounds),
              "--crossfit-rounds", str(args.crossfit_rounds), "--anchor-threshold", str(args.anchor_threshold),
              "--hop-k", str(args.hop_k), "--minimum-gain", str(args.minimum_gain)]
    if args.exclude_country:
        common += ["--exclude-country", *args.exclude_country]
    outputs = {
        "crossfit": ["crossfit.json", "models/crossfit_6.json", "models/crossfit_7.json"],
        "expand_train": ["graph_train.parquet", "expand_train.json"],
        "fit": ["final_models.json", "models/pair.json", "models/business.json"],
        "select": ["selection.json"], "evaluate": ["metrics.json"],
        "expand_test": ["expand_test.json"], "inference": [],
    }
    selection_file = args.work / "selection.json"
    selected = json.loads(selection_file.read_text())["selected"] if selection_file.exists() else None
    if selected != "baseline":
        outputs["expand_test"].append("graph_test.parquet")
        outputs["inference"].append("test_predictions.parquet")
    for step, artifacts in outputs.items():
        plan[step] = ([py, "-u", "-m", "er_v2.r9", step, *common], [args.work / p for p in artifacts])
    plan["inference"][1].extend([args.output / "matching_results.tsv", args.output / "candidate_pairs.tsv"])
    plan["validate"] = ([py, str(args.validator), "--matching", str(args.output / "matching_results.tsv"),
                         "--candidate", str(args.output / "candidate_pairs.tsv"),
                         "--test-dir", str(args.dataset / "test"), "--check-ids"], [])
    return plan


def stop_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def interrupt(signum, _frame):
    raise InterruptedError(f"Received signal {signum}; the active stage will restart on resume")


def report(args, state):
    metrics = json.loads((args.work / "metrics.json").read_text())
    complete = "validate" in state["completed"]
    result = {**metrics, "status": "complete" if complete else "evaluation_only",
              "official_validation": "PASS" if complete else "not run",
              "identity": state["identity"], "stage_seconds": {k: v["seconds"] for k, v in state["completed"].items()},
              "amazon_score": None, "outputs": {}}
    if complete:
        for filename in ("matching_results.tsv", "candidate_pairs.tsv"):
            path = args.output / filename
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result["outputs"][filename] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    for name, path in (("shift_diagnostic", args.work / "base/shift_check.json"),
                       ("expansion", args.work / "expand_train.json")):
        result[name] = json.loads(path.read_text())
    save(args.work / "result.json", result)
    chosen, baseline = result["local_fold4"]["macro_f05"], result["baseline_fold4"]["macro_f05"]
    text = (f"# R9 measured result\n\nSelected: **{result['selected']}**.\n\n"
            f"Local fold-4 macro F0.5: **{chosen:.6f}**; rebuilt baseline: **{baseline:.6f}**.\n\n"
            f"Difference: {(chosen - baseline) * 100:+.4f} percentage points.\n\n"
            f"R9 proposal diagnostic (not used for selection): {result['proposal_fold4_diagnostic']['macro_f05']:.6f}.\n\n"
            f"Official format/ID validation: {result['official_validation']}.\n\n"
            "Amazon and France scores have not been measured. Fold 4 was already observed in earlier r7 work; "
            "it is report-only here, not a new blind evaluation.\n\n"
            "See result.json for country metrics, the selection gate, candidate oracle, hashes and runtimes.\n")
    (args.work / "result.md").write_text(text, encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("work/r9"))
    p.add_argument("--output", type=Path, default=Path("output/r9"))
    p.add_argument("--dataset", type=Path, default=Path("student_resource/dataset"))
    p.add_argument("--validator", type=Path, default=Path("student_resource/utils/validate_submission.py"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--base-rounds", type=int, default=1500)
    p.add_argument("--base-graph-rounds", type=int, default=2000)
    p.add_argument("--crossfit-rounds", type=int, default=700)
    p.add_argument("--rounds", type=int, default=1200)
    p.add_argument("--anchor-threshold", type=float, default=0.8)
    p.add_argument("--hop-k", type=int, default=15)
    p.add_argument("--minimum-gain", type=float, default=0.0005)
    p.add_argument("--exclude-country", nargs="*", choices=["India", "US"], default=[])
    p.add_argument("--max-hours", type=float, default=12, help="Wall time for this invocation, including resumed stages")
    p.add_argument("--until", choices=["evaluate", "validate"], default="validate")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.threads, args.base_rounds, args.base_graph_rounds, args.crossfit_rounds, args.rounds) < 1:
        p.error("Thread and round counts must be positive")
    if not 0 < args.max_hours <= 72 or not 0 <= args.anchor_threshold <= 1 or not 1 <= args.hop_k < 65535 or not 0 <= args.minimum_gain <= 1:
        p.error("Invalid wall time, anchor cutoff, hop count or minimum gain")
    args.exclude_country = sorted(set(args.exclude_country))
    if len(args.exclude_country) > 1:
        p.error("Cannot exclude every labeled country")
    for field in ("work", "output", "dataset", "validator"):
        setattr(args, field, getattr(args, field).resolve())
    for left, right in ((args.work, args.dataset), (args.output, args.dataset), (args.work, args.output)):
        if left == right or left in right.parents or right in left.parents:
            p.error("Work, output and dataset directories must be disjoint")
    sequence = STEPS[:STEPS.index(args.until) + 1]
    if args.plan:
        for step in sequence:
            print(step, subprocess.list2cmdline(commands(args)[step][0]))
        return
    args.work.mkdir(parents=True, exist_ok=True)
    # Kernel releases flock even after SIGKILL or instance shutdown; no stale PID guessing.
    with (args.work / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, sequence)


def run(args, sequence):
    for folder in ("logs", "models", "base/models"):
        (args.work / folder).mkdir(parents=True, exist_ok=True)
    data = [args.dataset / split / f"{split}_source{i}.tsv" for split in ("train", "test") for i in (1, 2, 3)]
    data += [args.dataset / "train/train_ground_truth.tsv", args.validator]
    identity = {"code_sha256": code_hash(), "dataset": snapshot(data),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                             if k not in {"resume", "plan", "max_hours", "until"}},
                "python": sys.version, "dependencies": dependency_versions()}
    path = args.work / "run.json"
    if args.resume:
        state = json.loads(path.read_text())
        if state["identity"] != identity:
            raise RuntimeError("Code, data or settings changed. Use a fresh work/output directory; caches cannot be mixed.")
    else:
        if path.exists() or any((args.work / "base").glob("*.parquet")):
            raise FileExistsError("Existing run: use --resume with identical code/settings")
        state = {"identity": identity, "started": now(), "completed": {}}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "code/business_entity_resolution/src"),
           "POLARS_MAX_THREADS": str(args.threads), "OMP_NUM_THREADS": str(args.threads),
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    deadline = time.monotonic() + args.max_hours * 3600
    state.update(status="running", pid=os.getpid(), invocation_started=now(), max_hours=args.max_hours)
    save(path, state)
    try:
        for step in sequence:
            cmd, outputs = commands(args)[step]
            previous = state["completed"].get(step)
            if previous:
                if previous["command"] != cmd or previous["outputs"] != snapshot(outputs):
                    raise RuntimeError(f"Cached stage {step} changed; refuse unsafe resume")
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Invocation wall-time allowance exhausted")
            if shutil.disk_usage(args.work).free < 12 * 1024 ** 3:
                raise OSError("Less than 12 GiB disk free; expand EBS before resuming")
            log_path = args.work / "logs" / f"{step}.log"
            state.update(stage=step, log=str(log_path), stage_started=now())
            save(path, state)
            print(f"{now()} {step}: {log_path}", flush=True)
            started = time.monotonic()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nInvocation at {now()}\n")
                log.flush()
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while child.poll() is None:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Wall-time limit reached during {step}; resume will restart that stage")
                        time.sleep(3)
                finally:
                    stop_child(child)
            if child.returncode:
                raise RuntimeError(f"{step} failed ({child.returncode}); inspect {log_path}")
            state["completed"][step] = {"command": cmd, "outputs": snapshot(outputs),
                                         "seconds": round(time.monotonic() - started, 2), "finished": now()}
            save(path, state)
        report(args, state)
        state.update(status="complete" if "validate" in state["completed"] else "evaluation_only", finished=now())
        save(path, state)
        print(f"Result: {args.work / 'result.md'}", flush=True)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, (InterruptedError, TimeoutError, KeyboardInterrupt)) else "failed",
                     error=str(error), finished=now())
        save(path, state)
        raise


if __name__ == "__main__":
    main()
