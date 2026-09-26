#!/usr/bin/env bash
# r12 on the Google Cloud VM (g2-standard-32: 32 vCPU, 128 GB RAM, NVIDIA L4 24 GB, Ubuntu 24.04).
#
# One command, from the repository root on branch r12:
#   bash scripts/vm_r12.sh /path/to/student_resource          # first run
#   bash scripts/vm_r12.sh /path/to/student_resource --resume # after any interruption
#
# <student_resource> must contain dataset/{train,test}/*.tsv and utils/validate_submission.py.
# Runs detached (nohup), so an SSH disconnect does not stop it. Progress:
#   tail -f work/r12/runner.log        and        cat work/r12/run.json
#
# What r12 is: the r10 pipeline (lexical + phonetic blocking, fine-tuned multilingual
# encoder, neural candidates, enhanced features, stage 2, graph stage 3, cross-encoder,
# fusion with a strict holdout gate) plus everything learned on the desktop:
#   * exact GPU neighbour search instead of CPU FAISS (was ~2 h per split)
#   * r11 stage-2 features: record-side name competition (72% of remaining misses have
#     no address) and look-alike competition (France same-name collisions)
#   * cross-encoder without gradient checkpointing, batch 64 (was 16 x 4 for 12 GB)
#   * crash-safe progress file (fsync before rename), bounded GPU search memory
#   * 30 threads, 6M-pair feature shards, 1024-row encode/score batches for 128 GB / 24 GB
set -euo pipefail
cd "$(dirname "$0")/.."
SR=${1:?usage: bash scripts/vm_r12.sh /path/to/student_resource [--resume]}
shift || true
for option in "$@"; do
  [ "$option" = --resume ] || { echo "Only --resume is accepted; tune via R12_* environment variables."; exit 2; }
done
SR=$(realpath "$SR")
for split in train test; do
  for side in 1 2 3; do
    [ -f "$SR/dataset/$split/${split}_source${side}.tsv" ] || { echo "missing $split source$side"; exit 2; }
  done
done
[ -f "$SR/dataset/train/train_ground_truth.tsv" ] || { echo "missing train ground truth"; exit 2; }
[ -f "$SR/utils/validate_submission.py" ] || { echo "missing official validator"; exit 2; }

# Guard setup + launch, without overwriting a running session's log.
mkdir -p work/r12 output/r12
exec 9>work/r12/launch.lock
flock -n 9 || { echo "r12 setup/run already active"; exit 2; }
[ ! -f work/r12/run.json ] || [ "${1:-}" = --resume ] || { echo "Existing r12 run: use --resume"; exit 2; }

if [ ! -x .venv-r12/bin/python ]; then
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev >/dev/null
  python3.12 -m venv .venv-r12
fi
# A Python executable alone does not prove installation completed. On failure,
# the marker is absent and the next invocation repairs the same environment.
DEPS=$(sha256sum code/business_entity_resolution/requirements_{v2,r10}.txt)
if [ "$(cat .venv-r12/.ready 2>/dev/null || true)" != "$DEPS" ]; then
  .venv-r12/bin/python -m pip install -q --upgrade pip
  # CUDA 12.6 wheel (works with the L4's driver); pinned to the tested version
  .venv-r12/bin/python -m pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
  .venv-r12/bin/python -m pip install -q -r code/business_entity_resolution/requirements_r10.txt
  .venv-r12/bin/python -m pip check
  printf '%s\n' "$DEPS" > .venv-r12/.ready
fi
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export POLARS_MAX_THREADS=30 OMP_NUM_THREADS=30
.venv-r12/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; assert torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3, 'r12 profile needs 24 GB VRAM'; print('GPU', torch.cuda.get_device_name(0))"

HOURS=${R12_MAX_HOURS:-11}
ARGS=(--work work/r12 --output output/r12 --dataset "$SR/dataset" --validator "$SR/utils/validate_submission.py"
      --device cuda --threads "${R12_THREADS:-30}" --max-hours "$HOURS" --reserve-gb 20
      --encoder-pairs "${R12_ENCODER_PAIRS:-2000000}" --ce-train-businesses "${R12_CE_TRAIN_BUSINESSES:-250000}"
      --ann gpu-exact --neural-k 24 --rescue-k 8 --r11-features
      --shard-pairs 6000000 --encode-batch 1024
      --ce-batch 64 --ce-accumulation 1 --no-ce-checkpointing --ce-score-batch 1024 --ce-epochs 3)
.venv-r12/bin/python scripts/run_r10.py "${ARGS[@]}" --preflight
.venv-r12/bin/python -m unittest discover -s tests_v2
.venv-r12/bin/python scripts/run_r10.py "${ARGS[@]}" --plan > work/r12/plan.txt
# The Python deadline only stops the job. Arm VM shutdown one hour later too.
# Re-arm after any reboot; stopped disk/NAT resources still have storage/network charges.
MINUTES=$(.venv-r12/bin/python -c 'import math, sys; print(math.ceil(float(sys.argv[1]) * 60) + 60)' "$HOURS")
sudo shutdown -c
sudo shutdown -h "+$MINUTES"
nohup .venv-r12/bin/python -u scripts/run_r10.py "${ARGS[@]}" "$@" >> work/r12/runner.log 2>&1 &
echo "r12 started (pid $!). Follow with: tail -f work/r12/runner.log"
