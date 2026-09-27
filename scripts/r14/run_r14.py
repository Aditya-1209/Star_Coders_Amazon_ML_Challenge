"""r14 driver: wait for CE training, CE-score uncertain pairs, fuse (two bands), validate, copy."""
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(r"C:\Users\ashma\SC_r14")
REPO = Path(r"C:\Users\ashma\Star_Coders_Amazon_ML_Challenge")
PY = str(REPO / ".venv312/Scripts/python.exe")
WORK = Path(r"D:\star_r10b\work")
R14 = Path(r"D:\star_r14")
DATASET = REPO / "student_resource/dataset"
LOG = R14 / "run_r14.log"
env = dict(os.environ, PYTHONPATH=str(CODE / "code/business_entity_resolution/src"), PYTHONUTF8="1",
           TOKENIZERS_PARALLELISM="false", POLARS_MAX_THREADS="16", OMP_NUM_THREADS="16")


def note(m):
    line = f"=== {datetime.datetime.now():%H:%M:%S} {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(name, cmd):
    note(f"{name} start")
    with open(R14 / f"{name}.log", "w", encoding="utf-8") as fh:
        code = subprocess.call([PY, "-W", "ignore", *cmd], cwd=CODE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if code:
        note(f"!!! {name} FAILED (exit {code})")
        sys.exit(code)
    note(f"{name} done")


done = WORK / "ce_model/training.json"
while not done.exists():
    time.sleep(60)
note("CE training finished: " + json.dumps(json.loads(done.read_text())["history"]))

for split in ("train", "test"):
    out = R14 / f"ce_{split}_uncertain.parquet"
    if not out.exists():
        run(f"ce_score_{split}", ["scripts/r14/ce_score_pairs.py", "--work", str(WORK), "--split", split,
                                  "--pairs", str(R14 / f"pairs_{split}_uncertain.parquet"), "--out", str(out)])

best = None
for tag, band in (("wide", ("0.001", "0.999")), ("narrow", ("0.01", "0.99"))):
    out = R14 / f"fusion_{tag}"
    run(f"fuse_{tag}", ["scripts/r14/fuse.py", "--work", str(WORK), "--dataset", str(DATASET),
                        "--ce-train", str(R14 / "ce_train_uncertain.parquet"),
                        "--ce-test", str(R14 / "ce_test_uncertain.parquet"),
                        "--output", str(out), "--band", *band, "--threads", "16"])
    metrics = json.loads((out / "r14_metrics.json").read_text())
    chosen = metrics["results"][metrics["selected"]]
    note(f"{tag}: fold3 {chosen['fold3']:.5f} fold4 {chosen['fold4']['macro_f05']:.5f} {chosen['fold4_by_country']}")
    if best is None or chosen["fold3"] > best[1]:
        best = (out, chosen["fold3"], chosen["fold4"]["macro_f05"])

out = best[0]
note(f"selected {out.name} (fold3 {best[1]:.5f}, fold4 {best[2]:.5f})")
run("validate", [str(REPO / "student_resource/utils/validate_submission.py"),
                 "--matching", str(out / "matching_results.tsv"), "--candidate", str(out / "candidate_pairs.tsv"),
                 "--test-dir", str(DATASET / "test"), "--check-ids"])
dest = REPO / "submissions" / "r14"
dest.mkdir(parents=True, exist_ok=True)
for f in ("matching_results.tsv", "candidate_pairs.tsv", "r14_metrics.json"):
    shutil.copy2(out / f, dest / f)
note(f"R14 FILE READY: {dest}")
