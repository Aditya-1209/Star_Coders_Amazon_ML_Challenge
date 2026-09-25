# Suggested improvements (as of r7-improvements, 26 Sep 2026, ~01:00)

This branch is **r6.1 + stage 3**, the current best pipeline, plus an unrun
"ghost" option for test-density training. It also includes analysis scripts and
this plan. Everything below is ordered by expected leaderboard gain per hour on
the i9-13900K / 32 GB / RTX 3060 desktop.

## 1. Where we are

| Submission | Pipeline | Holdout F0.5 (fold 4) | Leaderboard |
|---|---|---:|---:|
| r1 | v2 blocking + 2-stage XGBoost | 0.9494 | 0.933 |
| r2 | + learned transliteration, Ltd fix, stage-1 pruning | 0.9586 | 0.94x |
| r3 | + stage 3 two-hop | 0.9631 | 0.944 |
| r6 stage 2 | r4 features + **split-dependent count/IDF features** | 0.9670 | **0.936**: regression |
| r6.1 stage 2 | r6 minus the 16 split-dependent features | 0.9624 | **0.949** |
| r6.1 stage 3 | + two-hop (`submissions/r61_stage3`) | 0.9658 | pending (estimate ~0.950-0.953) |

r6.1 stage 3 per country: **US 0.9698, India 0.9599**. The candidate-set oracle is
0.9803, and there are **6.8 candidates per Source 1** on test.

### Lessons

1. **The holdout cannot detect train/test distribution shift.** r6 improved the
   holdout by 0.4 and lost 0.8 on the leaderboard. The cause was features whose
   *scale* depends on split size: absolute counts, and IDF = log(N/df) with a
   different N per split. US has 1.32M train businesses but only 0.66M test.
   An adversarial classifier separated train from test with AUC 0.93-0.97. With
   only the original features, AUC was ~0.72.
   **Gate every new feature with `scripts/analysis/adversarial_check.py`.**
2. **Leaderboard estimate:** `metrics.leaderboard_estimate` weights US/India
   holdout scores by the test mix (38% US, 47% India, 15% France) and assumes
   France is 0.87. It predicted 0.947 for r6.1 stage 2, and the leaderboard gave
   0.949. France is currently about 0.88.

## 2. Where the remaining points are (r6.1, fold 4)

Loss is measured in "Source 1 equivalents" (1 - F per business), 8,285 in total:

| Bucket | Share of loss |
|---|---:|
| Has matches, all missed (predicted empty) | 30% |
| Has matches, some missed, no wrong ones | 55% |
| Some wrong matches | 12% |
| Singleton given a match | 3% |

- Missed true pairs: 64.3k. **Two thirds were never retrieved by blocking**
  (42.8k); 21.5k were retrieved but rejected by the model.
- India's never-retrieved misses: 30% have non-Latin names.
- Rejected misses: 32% (India) and 44% (US) have **no address**.
- Wrong matches: 4,414, and **99% go to records owned by no business**.

## 3. Plan

Every block ends in an upload candidate. The upload gates are: fold-4 holdout,
fold-4 holdout in ghost mode, adversarial AUC ≤ 0.75 per country, and the
official validator with `--check-ids`.

### Block 2: test-density ("ghost") training and validation. ~1h15. Code is in this branch.
- **Finding:** test has **5.75 Source 2/3 records per business** in every country;
  train has **4.68**. We predict about as many matches per business on test as on
  train, so the extra records are ownerless: roughly **2.3 per business on test vs
  1.2 in train**. Stage 2 learned "reject when another business claims this record
  strongly", and ownerless records have no competing claim. That fits the fact
  that 99% of our wrong matches go to ownerless records.
- **Fix:** `python -m er_v2.train --ghost-frac 0.19` removes a fixed 19% of training
  businesses, turning their records into ownerless ones and matching test density
  (derivation: (2.68M + 7.64M·q) / (2.21M·(1-q)) = 2.3 gives q ≈ 0.19). It
  recomputes the blocking-competition features (`b_rel_t`, `t_rank`, `t_nc`)
  without them, and excludes ghosts from scoring. Outputs are tagged `_ghost19`.
- **To do:** add the same `--ghost-frac` to `stage3.py`: apply `ghostify` to the
  stored direct-pair features in `build()`, and filter anchors/truth. Then compare
  normal vs ghost holdout, and the ghost threshold vs the normal one.

### Block 3: France. ~45 min
1. India-as-unseen proxy, already coded:
   `bash scripts/run_v2_pipeline.sh loco stproxy`. It reports unseen-India F0.5
   before/after self-training.
2. French normalization rules in `normalize.py`: region vs département names,
   `CEDEX`, `BP`, `bis`/`ter`, `St`/`Ste`/`Saint`, `Chem`/`Imp`/`Bd`/`Av`.
3. Only if the proxy improves: `bash scripts/run_v2_pipeline.sh selftrain stvalidate`,
   which adds pseudo-labels at score ≥ 0.97 / ≤ 0.02 for the unseen test countries.

### Block 4: recall and features. ~1h15. Needs new blocking + features.
1. **The 16 dropped features, made split-invariant:** IDF weights divided by
   log(N_country); counts turned into rates per 1M records of that country and
   side. They were worth +0.46 on the holdout. Re-check the adversarial AUC.
2. **Token alignment (Monge-Elkan):** mean over name tokens of the best
   Jaro-Winkler match on the other side. Robust to single-word typos.
3. **Phonetic blocking keys:** a consonant-class code per name token (e.g. b/p,
   d/t, g/k/q/c, s/z, v/w/f merged), aimed at India's non-Latin misses.
4. **Blank-address feature group:** name-strength features that only activate
   when one side has no address (30-44% of rejected true pairs).
5. **Second two-hop round:** re-anchor on stage-3 scores and expand again. The
   oracle is 0.980 and we score 0.966.

### Block 5: tuning sweep. ~1h, automated.
Stage 2 trains on stage-1 survivors only (~2.5M rows), so one fit takes about a
minute on the GPU. Tune on fold 3 and confirm once on fold 4.

| Knob | Now | Sweep |
|---|---|---|
| eta / rounds | 0.08 / early stop | 0.03, 0.05 |
| max_depth | 10 | 6, 8, 12 |
| min_child_weight | 5 | 1, 20 |
| colsample_bytree | 0.8 | 0.5, 0.65 |
| stage-1 negative keep rate (`--neg-frac`) | 0.5 | 0.3, 1.0 |
| stage-1 prune floor (`PRUNE`) | 0.001 | 0.0005, 0.003 (report candidates per S1) |
| two-hop anchor cutoff (`ANCHOR`) | 0.5 | 0.4, 0.6 |
| two-hop neighbours (`HOP_K`) | 10 | 5, 15 |
| two-hop support bars (`HOP_MIN_SUPPORT` / `HOP_NAME_ONLY`) | 50 / 90 | 40-70 / 85-95 |
| thresholds | one global | per country for US/India, global fallback for others |

Then train the final models on all non-holdout folds as a 3-seed ensemble.

### Block 6: final package. ~20 min
Update `Documentation_template.md` with the r6.x method and the numbers above.
Point README/requirements at `requirements_v2.txt` (it now needs `xgboost`, not
`lightgbm`). Build the zip with `er_v2.package`, verify it, and check that the
zip includes `output/candidate_pairs.tsv` from the same run as `matching_results.tsv`.

## 4. How to run (Windows, repo root, `.venv312`)

```bash
# full r6.1-style run (normalized records in work/norm are reused)
MODEL=work/model_r7 OUT=output_r7 bash scripts/run_v2_pipeline.sh block feats train predict s3train s3test validate
# test-density variant of stages 1+2 (then stage 3 once Block 2 is finished)
PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1 ./.venv312/Scripts/python -m er_v2.train --model-dir work/model_r7g --ghost-frac 0.19
# gates
PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1 ./.venv312/Scripts/python scripts/analysis/adversarial_check.py India work/model_r7/metrics.json
PYTHONPATH=code/business_entity_resolution/src PYTHONUTF8=1 ./.venv312/Scripts/python scripts/analysis/error_breakdown.py work/model_r7
```

Timings on the desktop:

| Step | Time |
|---|---|
| Blocking | train 20 min, test 13 min |
| Features | train 26 min, test 22 min |
| Stages 1+2 | 11 min |
| Prediction | 4 min |
| Stage 3 | train 25 min, test 38 min |
| **Full run** | **about 2h40** |
| Model-only rerun | about 1h15 |

Run one heavy step at a time: peak RAM is 13-20 GB.
