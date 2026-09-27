#!/usr/bin/env bash
# Reuse completed R12; a separate clone protects the running R15 checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
SR=${1:?usage: bash scripts/vm_r16_fast.sh /path/to/student_resource [--resume]}
shift
for option in "$@"; do
  [ "$option" = --resume ] || { echo 'Only --resume is accepted'; exit 2; }
done
SR=$(realpath "$SR")
BASE=$(realpath "${R16_BASE_WORK:?set R16_BASE_WORK to the original work/r12 directory}")
PY=$(realpath "${R16_PYTHON:?set R16_PYTHON to the existing .venv-r12/bin/python}")
[ -x "$PY" ] || { echo 'Existing R12 Python environment is required'; exit 2; }
# Never contend with the known R15 job on the same single GPU, or change its
# shutdown timer. The original checkout and saved neural artifacts stay intact.
R15_STATE="${R16_R15_WORK:-$(dirname "$BASE")/r15}/run.json"
"$PY" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()) if p.exists() else {}; assert s.get("status") != "running", "R15 is still running: wait for Complete before launching R16-fast"' "$R15_STATE"
mkdir -p work/r16_fast output/r16_fast
exec 9>work/r16_fast/launch.lock
flock -n 9 || { echo 'r16_fast setup/run already active'; exit 2; }
[ ! -f work/r16_fast/run.json ] || [ "${1:-}" = --resume ] || { echo 'Existing run: use --resume'; exit 2; }
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1
export POLARS_MAX_THREADS=${R16_THREADS:-30} OMP_NUM_THREADS=${R16_THREADS:-30}
HOURS=${R16_MAX_HOURS:-3}
ARGS=(--base-work "$BASE" --work work/r16_fast --output output/r16_fast
      --dataset "$SR/dataset" --validator "$SR/utils/validate_submission.py"
      --device cuda --threads "${R16_THREADS:-30}" --max-hours "$HOURS" --reserve-gb 20)
"$PY" -m pip check
"$PY" scripts/run_r16_fast.py "${ARGS[@]}" --preflight
"$PY" -m unittest discover -s tests_v2
"$PY" scripts/run_r16_fast.py "${ARGS[@]}" --plan > work/r16_fast/plan.txt
# Keep the existing known guest shutdown deadline; starting an experiment
# must not silently postpone it past the submission deadline.
sudo shutdown --show
nohup "$PY" -u scripts/run_r16_fast.py "${ARGS[@]}" "$@" >> work/r16_fast/runner.log 2>&1 &
echo "r16_fast started (pid $!). Follow with: tail -f work/r16_fast/runner.log"
