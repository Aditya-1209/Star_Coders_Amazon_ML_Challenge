#!/usr/bin/env bash
# Follow-up on the SAME completed R12 VM. Reuse neural artifacts read-only.
set -euo pipefail
cd "$(dirname "$0")/.."
SR=${1:?usage: bash scripts/vm_r15.sh /path/to/student_resource [--resume]}
shift
for option in "$@"; do
  [ "$option" = --resume ] || { echo 'Only --resume is accepted'; exit 2; }
done
SR=$(realpath "$SR")
PY=.venv-r12/bin/python
[ -x "$PY" ] || { echo 'R15 needs the completed R12 environment (.venv-r12).'; exit 2; }
mkdir -p work/r15 output/r15
exec 9>work/r15/launch.lock
flock -n 9 || { echo 'r15 setup/run already active'; exit 2; }
[ ! -f work/r15/run.json ] || [ "${1:-}" = --resume ] || { echo 'Existing R15 run: use --resume'; exit 2; }
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1
export POLARS_MAX_THREADS=${R15_THREADS:-30} OMP_NUM_THREADS=${R15_THREADS:-30}
HOURS=${R15_MAX_HOURS:-6}
ARGS=(--base-work "${R15_BASE_WORK:-work/r12}" --work work/r15 --output output/r15
      --dataset "$SR/dataset" --validator "$SR/utils/validate_submission.py"
      --device cuda --threads "${R15_THREADS:-30}" --max-hours "$HOURS" --reserve-gb 20)
"$PY" -m pip check
"$PY" scripts/run_r15.py "${ARGS[@]}" --preflight
"$PY" -m unittest discover -s tests_v2
"$PY" scripts/run_r15.py "${ARGS[@]}" --plan > work/r15/plan.txt
MINUTES=$("$PY" -c 'import math,sys; print(math.ceil(float(sys.argv[1])*60)+60)' "$HOURS")
sudo shutdown -c
sudo shutdown -h "+$MINUTES"
nohup "$PY" -u scripts/run_r15.py "${ARGS[@]}" "$@" >> work/r15/runner.log 2>&1 &
echo "r15 started (pid $!). Follow with: tail -f work/r15/runner.log"
