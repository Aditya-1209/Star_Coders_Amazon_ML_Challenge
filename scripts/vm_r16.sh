#!/usr/bin/env bash
# r16 on Mumbai asia-south1-c: g2-standard-32, L4 24 GB, 128 GB RAM,
# 200 GB balanced Persistent Disk, Ubuntu 24.04 / Python 3.12.
#
# One command, from the repository root on branch codex/r16-l4:
#   bash scripts/vm_r16.sh /path/to/student_resource          # first run
#   bash scripts/vm_r16.sh /path/to/student_resource --resume # after any interruption
#
# <student_resource> must contain dataset/{train,test}/*.tsv and utils/validate_submission.py.
# Runs detached (nohup), so an SSH disconnect does not stop it. Progress:
#   tail -f work/r16/runner.log        and        cat work/r16/run.json
#
# R16 keeps the R12 CE sample recipe and gates sibling/presence models
# against a rebuilt R12-style CE ensemble. Default job/VM limits: 16/17 hours.
# Stops the VM ten minutes after the job exits (success or failure). Set
# R16_SHUTDOWN_ON_EXIT=0 to retain only the overall shutdown deadline.
set -euo pipefail
cd "$(dirname "$0")/.."
SR=${1:?usage: bash scripts/vm_r16.sh /path/to/student_resource [--resume]}
shift || true
for option in "$@"; do
  [ "$option" = --resume ] || { echo "Only --resume is accepted; tune via R16_* environment variables."; exit 2; }
done
case "${R16_SHUTDOWN_ON_EXIT:-1}" in 0|1) ;; *) echo 'R16_SHUTDOWN_ON_EXIT must be 0 or 1'; exit 2 ;; esac
SR=$(realpath "$SR")
for split in train test; do
  for side in 1 2 3; do
    [ -f "$SR/dataset/$split/${split}_source${side}.tsv" ] || { echo "missing $split source$side"; exit 2; }
  done
done
[ -f "$SR/dataset/train/train_ground_truth.tsv" ] || { echo "missing train ground truth"; exit 2; }
[ -f "$SR/utils/validate_submission.py" ] || { echo "missing official validator"; exit 2; }

# Guard setup + launch, without overwriting a running session's log.
mkdir -p work/r16 output/r16
exec 9>work/r16/launch.lock
flock -n 9 || { echo "r16 setup/run already active"; exit 2; }
[ ! -f work/r16/run.json ] || [ "${1:-}" = --resume ] || { echo "Existing r16 run: use --resume"; exit 2; }

if [ ! -x .venv-r16/bin/python ]; then
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev >/dev/null
  python3.12 -m venv .venv-r16
fi
# A Python executable alone does not prove installation completed. On failure,
# the marker is absent and the next invocation repairs the same environment.
DEPS=$(sha256sum code/business_entity_resolution/requirements_{v2,r10}.txt)
if [ "$(cat .venv-r16/.ready 2>/dev/null || true)" != "$DEPS" ]; then
  # Avoid retaining a second copy of multi-GB CUDA wheels on the 200 GB disk.
  # Existing caches from other experiments are left intact.
  .venv-r16/bin/python -m pip install --no-cache-dir -q --upgrade pip
  # CUDA 12.6 wheel (works with the L4's driver); pinned to the tested version
  .venv-r16/bin/python -m pip install --no-cache-dir -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
  .venv-r16/bin/python -m pip install --no-cache-dir -q -r code/business_entity_resolution/requirements_r10.txt
  .venv-r16/bin/python -m pip check
  printf '%s\n' "$DEPS" > .venv-r16/.ready
fi
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export POLARS_MAX_THREADS=${R16_THREADS:-30} OMP_NUM_THREADS=${R16_THREADS:-30}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-r16/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; assert torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3, 'r16 profile needs 24 GB VRAM'; assert torch.cuda.is_bf16_supported(including_emulation=False), 'L4 profile needs native BF16'; print('GPU', torch.cuda.get_device_name(0))"

HOURS=${R16_MAX_HOURS:-16}
ARGS=(--profile vm --work work/r16 --output output/r16 --dataset "$SR/dataset" --validator "$SR/utils/validate_submission.py"
      --threads "${R16_THREADS:-30}" --max-hours "$HOURS"
      --ce-token-cache-gb "${R16_CE_TOKEN_CACHE_GB:-12}"
      --encoder-pairs "${R16_ENCODER_PAIRS:-1000000}" --ce-train-businesses "${R16_CE_TRAIN_BUSINESSES:-250000}")
.venv-r16/bin/python scripts/run_r16.py "${ARGS[@]}" --preflight
.venv-r16/bin/python -m unittest discover -s tests_v2
.venv-r16/bin/python scripts/run_r16.py "${ARGS[@]}" --plan > work/r16/plan.txt
# The Python deadline only stops the job. Arm VM shutdown one hour later too.
# Re-arm after any reboot; stopped disk/NAT resources still have storage/network charges.
MINUTES=$(.venv-r16/bin/python -c 'import math, sys; print(math.ceil(float(sys.argv[1]) * 60) + 60)' "$HOURS")
sudo shutdown -c
sudo shutdown -h "+$MINUTES"
export R16_STOP_EPOCH=$(( $(date +%s) + MINUTES * 60 ))
export R16_SHUTDOWN_ON_EXIT=${R16_SHUTDOWN_ON_EXIT:-1}
nohup bash scripts/vm_r16_job.sh "${ARGS[@]}" "$@" >> work/r16/runner.log 2>&1 &
echo "r16 started (pid $!). Follow with: tail -f work/r16/runner.log"
