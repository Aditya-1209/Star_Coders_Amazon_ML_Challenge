#!/usr/bin/env bash
# Cached update for g2-standard-32 / L4; leaves completed work/r12 intact.
set -euo pipefail
cd "$(dirname "$0")/.."
SR=${1:?usage: bash scripts/vm_r12_france.sh /path/to/student_resource [--resume]}
shift
for option in "$@"; do
  [ "$option" = --resume ] || { echo 'Only --resume is accepted'; exit 2; }
done
SR=$(realpath "$SR")
BASE=$(realpath "${R12_BASE_WORK:-work/r12}")
PY=${R12_PYTHON:-.venv-r12/bin/python}
# Preserve the venv executable symlink; resolving it would select system Python.
PY="$(cd -- "$(dirname -- "$PY")" && pwd -P)/$(basename -- "$PY")"
[ -x "$PY" ] || { echo 'Keep the completed R12 Python environment; no reinstall is needed'; exit 2; }
mkdir -p work/r12_france output/r12_france
exec 9>work/r12_france/launch.lock
flock -n 9 || { echo 'r12_france setup/run already active'; exit 2; }
[ ! -f work/r12_france/run.json ] || [ "${1:-}" = --resume ] || { echo 'Existing run: use --resume'; exit 2; }
# A single L4 cannot meet this profile while another training job uses it.
ACTIVE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$ACTIVE" ] || { echo "GPU already in use by PID(s): $ACTIVE. Wait for that job."; exit 2; }
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1
export POLARS_MAX_THREADS=${R12_THREADS:-30} OMP_NUM_THREADS=${R12_THREADS:-30}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=(--base-work "$BASE" --work work/r12_france --output output/r12_france
      --dataset "$SR/dataset" --validator "$SR/utils/validate_submission.py"
      --device cuda --threads "${R12_THREADS:-30}" --max-hours "${R12_FR_MAX_HOURS:-3.5}"
      --rounds 900 --reserve-gb 20)
"$PY" -m pip check
"$PY" scripts/run_r12_france.py "${ARGS[@]}" --preflight
"$PY" -m unittest discover -s tests_v2 -p 'test_r12_france.py'
"$PY" scripts/run_r12_france.py "${ARGS[@]}" --plan > work/r12_france/plan.txt
# This launcher does not postpone any existing guest/cloud shutdown deadline.
echo 'Job cap: 3.5 hours by default (at most 4). VM billing continues until stopped.'
nohup "$PY" -u scripts/run_r12_france.py "${ARGS[@]}" "$@" >> work/r12_france/runner.log 2>&1 &
echo "r12_france started (pid $!). Follow: tail -f work/r12_france/runner.log"
