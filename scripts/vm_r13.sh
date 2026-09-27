#!/usr/bin/env bash
# r13 on Mumbai asia-south1-c: g2-standard-32, L4 24 GB, 128 GB RAM,
# 200 GB balanced Persistent Disk, Ubuntu 24.04 / Python 3.12.
#
# One command, from the repository root on branch r13:
#   bash scripts/vm_r13.sh /path/to/student_resource          # first run
#   bash scripts/vm_r13.sh /path/to/student_resource --resume # after any interruption
#
# <student_resource> must contain dataset/{train,test}/*.tsv and utils/validate_submission.py.
# Runs detached (nohup), so an SSH disconnect does not stop it. Progress:
#   tail -f work/r13/runner.log        and        cat work/r13/run.json
#
# R13 adds stage-2-mined CE negatives, bounded neural sibling evidence,
# and final features that exclude incomplete target competition context.
# Stops the VM ten minutes after the job exits (success or failure). Set
# R13_SHUTDOWN_ON_EXIT=0 to retain only the overall shutdown deadline.
set -euo pipefail
cd "$(dirname "$0")/.."
SR=${1:?usage: bash scripts/vm_r13.sh /path/to/student_resource [--resume]}
shift || true
for option in "$@"; do
  [ "$option" = --resume ] || { echo "Only --resume is accepted; tune via R13_* environment variables."; exit 2; }
done
case "${R13_SHUTDOWN_ON_EXIT:-1}" in 0|1) ;; *) echo 'R13_SHUTDOWN_ON_EXIT must be 0 or 1'; exit 2 ;; esac
SR=$(realpath "$SR")
for split in train test; do
  for side in 1 2 3; do
    [ -f "$SR/dataset/$split/${split}_source${side}.tsv" ] || { echo "missing $split source$side"; exit 2; }
  done
done
[ -f "$SR/dataset/train/train_ground_truth.tsv" ] || { echo "missing train ground truth"; exit 2; }
[ -f "$SR/utils/validate_submission.py" ] || { echo "missing official validator"; exit 2; }

# Guard setup + launch, without overwriting a running session's log.
mkdir -p work/r13 output/r13
exec 9>work/r13/launch.lock
flock -n 9 || { echo "r13 setup/run already active"; exit 2; }
[ ! -f work/r13/run.json ] || [ "${1:-}" = --resume ] || { echo "Existing r13 run: use --resume"; exit 2; }

if [ ! -x .venv-r13/bin/python ]; then
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev >/dev/null
  python3.12 -m venv .venv-r13
fi
# A Python executable alone does not prove installation completed. On failure,
# the marker is absent and the next invocation repairs the same environment.
DEPS=$(sha256sum code/business_entity_resolution/requirements_{v2,r10}.txt)
if [ "$(cat .venv-r13/.ready 2>/dev/null || true)" != "$DEPS" ]; then
  # Avoid retaining a second copy of multi-GB CUDA wheels on the 200 GB disk.
  # Existing caches from other experiments are left intact.
  .venv-r13/bin/python -m pip install --no-cache-dir -q --upgrade pip
  # CUDA 12.6 wheel (works with the L4's driver); pinned to the tested version
  .venv-r13/bin/python -m pip install --no-cache-dir -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
  .venv-r13/bin/python -m pip install --no-cache-dir -q -r code/business_entity_resolution/requirements_r10.txt
  .venv-r13/bin/python -m pip check
  printf '%s\n' "$DEPS" > .venv-r13/.ready
fi
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export POLARS_MAX_THREADS=${R13_THREADS:-30} OMP_NUM_THREADS=${R13_THREADS:-30}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-r13/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; assert torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3, 'r13 profile needs 24 GB VRAM'; assert torch.cuda.is_bf16_supported(including_emulation=False), 'L4 profile needs native BF16'; print('GPU', torch.cuda.get_device_name(0))"

HOURS=${R13_MAX_HOURS:-11}
ARGS=(--profile vm --work work/r13 --output output/r13 --dataset "$SR/dataset" --validator "$SR/utils/validate_submission.py"
      --threads "${R13_THREADS:-30}" --max-hours "$HOURS"
      --ce-token-cache-gb "${R13_CE_TOKEN_CACHE_GB:-12}"
      --encoder-pairs "${R13_ENCODER_PAIRS:-1000000}" --ce-train-businesses "${R13_CE_TRAIN_BUSINESSES:-250000}")
.venv-r13/bin/python scripts/run_r13.py "${ARGS[@]}" --preflight
.venv-r13/bin/python -m unittest discover -s tests_v2
.venv-r13/bin/python scripts/run_r13.py "${ARGS[@]}" --plan > work/r13/plan.txt
# The Python deadline only stops the job. Arm VM shutdown one hour later too.
# Re-arm after any reboot; stopped disk/NAT resources still have storage/network charges.
MINUTES=$(.venv-r13/bin/python -c 'import math, sys; print(math.ceil(float(sys.argv[1]) * 60) + 60)' "$HOURS")
sudo shutdown -c
sudo shutdown -h "+$MINUTES"
export R13_STOP_EPOCH=$(( $(date +%s) + MINUTES * 60 ))
export R13_SHUTDOWN_ON_EXIT=${R13_SHUTDOWN_ON_EXIT:-1}
nohup bash scripts/vm_r13_job.sh "${ARGS[@]}" "$@" >> work/r13/runner.log 2>&1 &
echo "r13 started (pid $!). Follow with: tail -f work/r13/runner.log"
