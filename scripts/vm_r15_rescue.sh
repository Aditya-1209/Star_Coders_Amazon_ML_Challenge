#!/usr/bin/env bash
# Separate checkout; do not edit R15 or R16 code under a running process.
set -euo pipefail
cd "$(dirname "$0")/.."
SR=${1:?usage: bash scripts/vm_r15_rescue.sh /path/to/student_resource [--resume]}
shift
for option in "$@"; do
  [ "$option" = --resume ] || { echo 'Only --resume is accepted'; exit 2; }
done
SR=$(realpath "$SR")
BASE=$(realpath "${R15_RESCUE_BASE_WORK:?set R15_RESCUE_BASE_WORK to original work/r12}")
PY=$(realpath "${R15_RESCUE_PYTHON:?set R15_RESCUE_PYTHON to existing .venv-r12/bin/python}")
[ -x "$PY" ] || { echo 'Existing R12 Python environment required'; exit 2; }
R15_STATE="${R15_RESCUE_R15_WORK:-$(dirname "$BASE")/r15}/run.json"
R16_STATE="${R15_RESCUE_R16_WORK:-$HOME/Star_Coders_r16_fast/work/r16_fast}/run.json"
"$PY" -c 'import json,pathlib,sys; states=[json.loads(pathlib.Path(p).read_text()) for p in sys.argv[1:] if pathlib.Path(p).exists()]; assert all(s.get("status") != "running" for s in states), "R15/R16 is still running: wait for Complete before launching rescue"' "$R15_STATE" "$R16_STATE"
mkdir -p work/r15_rescue output/r15_rescue
exec 9>work/r15_rescue/launch.lock
flock -n 9 || { echo 'r15_rescue setup/run already active'; exit 2; }
[ ! -f work/r15_rescue/run.json ] || [ "${1:-}" = --resume ] || { echo 'Existing run: use --resume'; exit 2; }
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1
export POLARS_MAX_THREADS=${R15_RESCUE_THREADS:-30} OMP_NUM_THREADS=${R15_RESCUE_THREADS:-30}
HOURS=${R15_RESCUE_MAX_HOURS:-3}
ARGS=(--base-work "$BASE" --work work/r15_rescue --output output/r15_rescue
      --dataset "$SR/dataset" --validator "$SR/utils/validate_submission.py"
      --device cuda --threads "${R15_RESCUE_THREADS:-30}" --max-hours "$HOURS" --reserve-gb 20)
"$PY" -m pip check
"$PY" scripts/run_r15_rescue.py "${ARGS[@]}" --preflight
"$PY" -m unittest discover -s tests_v2
"$PY" scripts/run_r15_rescue.py "${ARGS[@]}" --plan > work/r15_rescue/plan.txt
# Keep the known shutdown time; never silently extend it past the deadline.
sudo shutdown --show
nohup "$PY" -u scripts/run_r15_rescue.py "${ARGS[@]}" "$@" >> work/r15_rescue/runner.log 2>&1 &
echo "r15_rescue started (pid $!). Follow with: tail -f work/r15_rescue/runner.log"
