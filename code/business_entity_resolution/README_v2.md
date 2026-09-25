# Star Coders: Business Entity Resolution (v2 pipeline)

For the r4 branch, follow `docs/README_r4.md` from the repository root. Stage 3
requires retraining after r4's missing-address support fix; old stage-3 models
are rejected. No new accuracy or performance measurements are claimed for r4.

Matches every test Source 1 business to its Source 2 / Source 3 records using
only the challenge data. Pipeline: text normalization, then weighted
inverted-key blocking (top 64 candidates per business), then 43 string and
context features, then a two-stage gradient-boosted classifier (XGBoost),
then threshold selection and one-owner-per-target assignment.

No external data, APIs, geocoders or pretrained language models are used.
All libraries are MIT / BSD / Apache-2.0 / ISC licensed; the model is a
boosted-tree classifier trained from scratch on the provided labels.

## Environment

* Python 3.12 (tested on Windows 11 native; Linux works identically)
* From the repository root: `python -m pip install -r code/business_entity_resolution/requirements_v2.txt`
* Hardware used: i9-13900K (32 threads), 32 GB RAM, RTX 3060 12 GB.
  Peak RAM is about 22 GB during feature generation. The GPU is used for
  XGBoost training and inference. In r4, pass `--device cpu` to `er_v2.train`,
  `er_v2.predict`, and `er_v2.stage3` when CUDA is unavailable. No code edits needed.

## Layout expected

```
student_resource/dataset/{train,test}/*.tsv   # organizer data (not shipped)
code/business_entity_resolution/src/er_v2/    # this code
```

Run everything from the repository root with
`PYTHONPATH=code/business_entity_resolution/src` and `PYTHONUTF8=1`.
Intermediate files go to `work/`.

## Reproduce end to end (about 1 hour on the hardware above)

```bash
export PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1
# 1. normalize all six source files -> work/norm/*.parquet
python -m er_v2.prepare
# 2. blocking -> work/cands_{train,test}.parquet
python -m er_v2.run_block --split train
python -m er_v2.run_block --split test
# 3. pair features -> work/feats_{train,test}/part_*.parquet
python -m er_v2.run_features --split train
python -m er_v2.run_features --split test
# 4. train stage-1/stage-2 models, tune threshold -> models/v2/
python -m er_v2.train --model-dir models/v2
# 5. score test, write output/matching_results.tsv + output/candidate_pairs.tsv
python -m er_v2.predict --model-dir models/v2 --output output
# 6. official format check
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test --check-ids
```

To skip training, use the shipped `models/v2/` and run steps 1, 2, 3 (test
split only) and 5. `models/v2/metrics.json` stores the features, the
threshold and the validation scores.

## Modules

| file | role |
| --- | --- |
| `normalize.py` | transliteration (anyascii), OCR-typo repair, canonical abbreviations, legal-suffix stripping |
| `translit.py` | learns the Indic-script token to Latin token map from aligned training pairs |
| `prepare.py` | parallel normalization of the raw TSVs |
| `block.py`, `run_block.py` | country-scoped IDF-weighted key blocking and top-K candidates |
| `features.py`, `run_features.py` | RapidFuzz similarities, token-set overlaps, blocking context |
| `train.py` | fold split, stage-1 and stage-2 XGBoost, threshold tuning, validation metrics |
| `metrics.py` | macro F0.5 exactly as the challenge defines it (singletons included) |
| `predict.py` | test inference and TSV writing, with exactly one row per Source 1 in input order |
| `package.py` | builds the final submission ZIP |
