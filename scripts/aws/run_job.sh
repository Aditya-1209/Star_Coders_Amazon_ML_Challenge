#!/bin/bash
set -Eeuo pipefail
cd /opt/r9/repo
export PYTHONPATH="$PWD/code/business_entity_resolution/src"
export POLARS_MAX_THREADS=12 OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi
# Fail before the full pipeline if CUDA silently falls back to CPU.
.venv/bin/python scripts/check_gpu.py
args=(--work work/r9 --output output/r9 --device cuda --threads 12 --max-hours "$R9_MAX_HOURS")
if [[ "$R9_PROXY_COUNTRY" != none ]]; then
  args+=(--exclude-country "$R9_PROXY_COUNTRY" --until evaluate)
fi
if [[ -f work/r9/run.json ]]; then
  args+=(--resume)
fi
exec .venv/bin/python -u scripts/run_r9.py "${args[@]}"
