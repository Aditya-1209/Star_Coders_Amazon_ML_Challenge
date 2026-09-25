#!/usr/bin/env bash
# Sequential v2 pipeline runner (one heavy step at a time; logs in $WORK/logs).
#
#   bash scripts/run_v2_pipeline.sh [steps...]
#
# Default steps (normalized records in $WORK/norm are reused; add translit/prepare
# only if normalization changed):
#   block feats train predict s3train s3test validate
# Optional steps:
#   translit prepare         redo transliteration map + normalization
#   loco                     retrain stages 1-3 without India (unseen-country proxy)
#   stproxy                  self-training check on India (needs loco)
#   selftrain stvalidate     France self-training on test -> $OUT_ST, then validate
# Environment overrides: WORK (work), MODEL (work/model_r6), OUT (output_r6),
#   OUT_ST (output_r6_selftrain), PY (./.venv312/Scripts/python).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1
WORK=${WORK:-work}
MODEL=${MODEL:-$WORK/model_r6}
OUT=${OUT:-output_r6}
OUT_ST=${OUT_ST:-output_r6_selftrain}
PY=${PY:-./.venv312/Scripts/python}
LOG=$WORK/logs
mkdir -p "$LOG" "$MODEL"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(block feats train predict s3train s3test validate)

run() {  # run <name> <command...>
  local name=$1; shift
  echo "=== $(date +%T) $name" | tee -a "$LOG/pipeline.log"
  "$@" > "$LOG/$name.log" 2>&1 || { echo "!!! $name failed, see $LOG/$name.log" | tee -a "$LOG/pipeline.log"; tail -20 "$LOG/$name.log"; exit 1; }
  tail -3 "$LOG/$name.log" | tee -a "$LOG/pipeline.log"
}

for step in "${STEPS[@]}"; do
  case $step in
    translit) run translit $PY -u -m er_v2.translit --out "$MODEL/translit.json" ;;
    prepare)  run prepare $PY -u -m er_v2.prepare --work "$WORK" --translit "$MODEL/translit.json" ;;
    block)    run block_train $PY -u -m er_v2.run_block --work "$WORK" --split train
              run block_test  $PY -u -m er_v2.run_block --work "$WORK" --split test ;;
    feats)    run feats_train $PY -u -m er_v2.run_features --work "$WORK" --split train --overwrite
              run feats_test  $PY -u -m er_v2.run_features --work "$WORK" --split test --overwrite ;;
    train)    run train $PY -u -m er_v2.train --work "$WORK" --model-dir "$MODEL" ;;
    predict)  run predict $PY -u -m er_v2.predict --work "$WORK" --model-dir "$MODEL" --output "$OUT" ;;
    s3train)  run s3train $PY -u -m er_v2.stage3 --work "$WORK" --split train --model-dir "$MODEL" ;;
    s3test)   run s3test $PY -u -m er_v2.stage3 --work "$WORK" --split test --model-dir "$MODEL" --output "$OUT" ;;
    validate) run validate $PY student_resource/utils/validate_submission.py \
                --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" \
                --test-dir student_resource/dataset/test --check-ids ;;
    loco)     run loco_train $PY -u -m er_v2.train --work "$WORK" --model-dir "$MODEL/loco_noIndia" --exclude-country India
              run loco_s3    $PY -u -m er_v2.stage3 --work "$WORK" --split train --model-dir "$MODEL/loco_noIndia" --exclude-country India ;;
    stproxy)  run stproxy $PY -u -m er_v2.selftrain proxy --work "$WORK" --model-dir "$MODEL/loco_noIndia" --country India ;;
    selftrain) run selftrain $PY -u -m er_v2.selftrain test --work "$WORK" --model-dir "$MODEL" --output "$OUT_ST" ;;
    stvalidate) run stvalidate $PY student_resource/utils/validate_submission.py                 --matching "$OUT_ST/matching_results.tsv" --candidate "$OUT_ST/candidate_pairs.tsv"                 --test-dir student_resource/dataset/test --check-ids ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
echo "=== $(date +%T) done" | tee -a "$LOG/pipeline.log"
