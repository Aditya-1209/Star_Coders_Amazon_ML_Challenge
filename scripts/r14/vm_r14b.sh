#!/usr/bin/env bash
# r14 CE-B on a GCP L4 VM. Run from the unpacked bundle directory:  nohup bash vm_r14b.sh > run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD/src PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10
stamp() { echo "=== $(date -u +%H:%M:%S) UTC $*"; }
if [ ! -f .deps_ok ]; then
  stamp setup
  python3 -m pip install -q --user --break-system-packages "transformers==5.17.0" "polars==1.44.2"       "polars-runtime-32==1.44.2" "pyarrow>=20" "safetensors" "accelerate" "anyascii==0.3.3" "RapidFuzz==3.14.6" "numpy<3"
  touch .deps_ok
fi
PY=python3
$PY -c "import torch, transformers; assert torch.cuda.is_available(); print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))"
[ -f work/ce_tokens/train_tg.npy ] || { stamp unpack; $PY cloud_ce.py unpack; }
[ -f work/ce_model/training.json ] || { stamp train; $PY -u cloud_ce.py train; }
for s in train test; do
  [ -f ce_${s}_uncertain_b.parquet ] || { stamp score $s; $PY -u ce_score_pairs.py --work work --split $s \
      --pairs pairs_${s}_uncertain.parquet --out ce_${s}_uncertain_b.parquet --batch 1024; }
done
stamp DONE
