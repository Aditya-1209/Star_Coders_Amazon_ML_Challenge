# r8: fine-tuned multilingual neural encoder

Built on `r7-improvements` (includes the r7 phonetic channel and enhanced features).
The rules allow MIT/Apache-2.0 models up to 8B parameters. We use
`intfloat/multilingual-e5-small` (MIT, 118M parameters), which reads Latin plus the
Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi
scripts natively. No external business data is used; the model is fine-tuned only
on the provided training labels.

## Results (holdout fold 4, 220,507 businesses; train-vs-test AUC gate passed)

| Model | Holdout F0.5 | Precision | Recall | US | India | Leaderboard |
|---|---:|---:|---:|---:|---:|---:|
| r6.1 stage 2 | 0.9624 | 99.37% | 91.6% | 0.967 | 0.956 | 0.949 |
| r6.1 stage 3 | 0.9658 | 99.43% | 92.9% | 0.970 | 0.960 | n/a |
| r7 stage 3 (Mac run) | 0.9674 | 99.51% | 93.1% | 0.971 | 0.962 | n/a |
| **r8 fast path: stage 2 + neural features** | **0.9694** | **99.71%** | 92.6% | **0.973** | **0.965** | pending |

- Quick check (small model over existing stage-2 scores): +0.42 points from neural
  features alone (0.9624 → 0.9666).
- Neural-feature train-vs-test AUC is 0.57-0.59, cleaner than the original features (~0.72).
- The fine-tuned encoder separates look-alike businesses. Example cosines:
  "Tir Club" vs "Roubaix Club" 0.85 → 0.16; the same name at a different address
  0.89 → 0.55. True matches, including Hindi and Tamil ↔ English, stay at 0.69-0.92.

## Leakage rule

The encoder (and the cross-encoder) are fine-tuned **only on stage-1 folds 0/1/8/9**.
Stage 1 never uses neural features. Stage 2 (folds 2/5), stage 3 (6/7), tuning (3)
and holdout (4) see only out-of-sample neural scores.

## Pipeline (Windows native, CUDA works; `.venv312`)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r code/business_entity_resolution/requirements_v2.txt
# fine-tune (35 min on RTX 3060), embed all 24M records (~55 min)
bash scripts/run_v2_pipeline.sh finetune encode nquick
# fast path: neural features in stage 2 on existing r6.1 features (~25 min)
MODEL=work/model_r8fast OUT=output_r8fast TRAIN_ARGS="--neural --country-thresholds" \
  bash scripts/run_v2_pipeline.sh train predict validate
# full r8: neural top-16 candidates + r7 phonetic channel + enhanced features + stage 3
MODEL=work/model_r8 OUT=output_r8 BLOCK_ARGS="--phonetic-k 8 --augment-phonetic" \
  FEAT_ARGS="--shard-pairs 4000000 --enhanced" \
  TRAIN_ARGS="--neural --feature-profile enhanced --compare-baseline --country-thresholds" \
  bash scripts/run_v2_pipeline.sh nblock nprobe block nmerge feats train predict s3train s3test validate
```

The fine-tuned weights (`work/neural_e5`, ~470 MB) and embeddings (`work/emb`, ~18 GB)
are Git-ignored; rerun `finetune` + `encode` to regenerate them.

## Next

1. Full r8 (in progress): neural retrieval to raise the candidate ceiling (oracle 0.978).
2. Cross-encoder re-ranker (`neural.py ce_finetune` / `ce_score`), coded and not yet run.
3. Test-density ("ghost") training, stage-2/3 cross-fitting on more folds, and
   per-country thresholds (see `SUGGESTED_IMPROVEMENTS.md`).
