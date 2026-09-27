#!/usr/bin/env bash
# Existing R12 environment only. No installer, shutdown change, or neural launch.
set -euo pipefail
cd "$(dirname "$0")/.."
resource="${1:?Usage: R16_R14_PROVENANCE=/path/inputs.json R16_SHUTDOWN_AT=... bash scripts/vm_r16_fast_fusion.sh /absolute/student_resource [--preflight|--resume]}"
shift
: "${R16_SHUTDOWN_AT:?Set the existing VM shutdown time, e.g. 2026-09-27T17:44:11Z}"
: "${R16_R14_PROVENANCE:?Set the manifest for EXISTING PRE-fusion R14 CE scores, not fusion_ab final scores}"
base="${R16_BASE_WORK:-$HOME/Star_Coders_Amazon_ML_Challenge/work/r12}"
py="${R16_PYTHON:-$HOME/Star_Coders_Amazon_ML_Challenge/.venv-r12/bin/python}"
export_dir="${R16_R12_EXPORT:-$HOME/r12_export}"
[[ -x "$py" ]] || { echo "Existing .venv-r12 Python missing: $py" >&2; exit 2; }
# Keep the venv launcher path: resolving its symlink would escape the environment.
export POLARS_MAX_THREADS="${R16_THREADS:-24}" OMP_NUM_THREADS="${R16_THREADS:-24}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/code/business_entity_resolution/src${PYTHONPATH:+:$PYTHONPATH}"
args=(--base-work "$base" --r12-export "$export_dir" --r14-provenance "$R16_R14_PROVENANCE"
      --dataset "$resource/dataset" --validator "$resource/utils/validate_submission.py"
      --threads "${R16_THREADS:-24}" --shutdown-at "$R16_SHUTDOWN_AT"
      --max-minutes "${R16_MAX_MINUTES:-45}" "$@")
# Foreground: use tmux/screen or the documented nohup command to disconnect safely.
exec "$py" -u scripts/run_r16_fast_fusion.py "${args[@]}"
