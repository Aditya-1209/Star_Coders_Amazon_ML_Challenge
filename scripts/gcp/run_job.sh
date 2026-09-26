#!/bin/bash
set -Eeuo pipefail
cd /opt/r9/repo
export PYTHONPATH="$PWD/code/business_entity_resolution/src"
export POLARS_MAX_THREADS="$R9_THREADS" OMP_NUM_THREADS="$R9_THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
# DLVM driver initialization may still be finishing on the first boot.
timeout 300 /bin/bash -c 'until nvidia-smi >/dev/null 2>&1; do sleep 5; done'
nvidia-smi
.venv/bin/python scripts/check_gpu.py
args=(--work work/r9 --output output/r9 --device cuda --threads "$R9_THREADS" --max-hours "$R9_MAX_HOURS")
if [[ "$R9_PROXY_COUNTRY" != none ]]; then
  args+=(--exclude-country "$R9_PROXY_COUNTRY" --until evaluate)
fi
if [[ -f work/r9/run.json ]]; then
  args+=(--resume)
fi
exec .venv/bin/python -u scripts/run_r9.py "${args[@]}"
