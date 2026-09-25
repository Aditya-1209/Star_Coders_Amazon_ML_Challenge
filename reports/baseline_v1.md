# First CPU baseline

This is a development experiment on a reduced candidate corpus. It is not a leaderboard score or a full-test submission.

The model uses 6,000 sampled Source 1 businesses and a search pool of 220,838 Source 2/3 records (20,838 labeled matches plus 200,000 random distractors). The full training target pool contains 10,320,219 records.

## Results

| Partition | Source 1 businesses | Macro F0.5 | Pair precision | Pair recall | Candidate recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 4,198 | 0.9755 | 98.53% | 96.89% | 98.01% |
| tune | 898 | 0.9661 | 97.91% | 96.04% | 97.79% |
| holdout | 904 | 0.9741 | 98.27% | 96.48% | 98.49% |

The threshold **0.435** was selected on the tuning partition. The holdout businesses were not used to fit the classifier or choose its threshold.

- Holdout candidate oracle macro F0.5: **0.9962**. This is the score a perfect classifier could achieve using only the retrieved candidates.
- Always predicting no matches would score **0.0586** on this holdout.
- Holdout singleton accuracy: **94.34%**, across 53 businesses with no matches.
- Mean candidates per holdout business: **47.3**.

| Holdout country | Businesses | Macro F0.5 | Pair precision | Pair recall |
| --- | ---: | ---: | ---: | ---: |
| India | 363 | 0.9662 | 97.73% | 94.39% |
| US | 541 | 0.9795 | 98.62% | 97.83% |

## Error breakdown

- False positive pairs: 53.
- True pairs missed by retrieval: 47.
- Retrieved true pairs rejected by the classifier: 63.

## CPU and memory

Measured on an Apple M4 with 16 GB RAM; CatBoost used six CPU threads and no GPU.

- Development sample preparation: 14.7 seconds.
- Search index construction: 6.8 seconds.
- Candidate retrieval and feature generation for all 6,000 businesses: 69.6 seconds.
- Classifier fitting: 4.7 seconds.
- Training command total through scoring: 75.5 seconds.
- Peak training-process resident memory: 527.7 MiB (not whole-system memory).
- Candidate pairs scored: 284,024.

## Most influential features

- `address_token_set`: 21.97
- `name_core_ratio`: 16.76
- `address_token_containment`: 10.59
- `name_partial`: 6.38
- `address_token_sort`: 4.91
- `address_ratio`: 4.46
- `name_rank_inverse`: 3.66
- `name_token_sort`: 3.51

## Interpretation and next step

The CPU pipeline is operational. The next experiment should increase the candidate corpus toward the full training pool, then reevaluate retrieval recall, false positives, speed, and threshold choice. Do this before full test inference; extra distractors can change both rankings and accuracy.

## Limits

- The reduced target pool includes all sampled labels plus uniformly sampled distractors.
- This is easier than full-corpus retrieval and is not a leaderboard score estimate.
- Partitions share an unlabeled search index, but no anchor or labeled positive target crosses partitions.
- France has no labels, so this run does not establish French matching accuracy.
- Country-restricted lexical retrieval can miss alternate-script names when addresses provide no shared text.
- Holdout exports are development predictions only. No full-test predictions have been generated or uploaded.

## Verification

- 8 unit tests passed, covering scoring, grouping, Unicode, missing fields, and search ranking.
- Official validator with ID-existence checks: PASS on all 904 development holdout rows.
- Strict candidate subset and valid-ID audit: PASS.
- Independent score calculation from exported TSV files: PASS.
- Reloaded model reproduced both exports exactly on 47 smoke-test businesses, including singletons.

## Reproduction

See `code/business_entity_resolution/README.md` for commands, pinned dependencies, and model details. Machine-readable metrics, model weights, pair features, predictions, and errors are in `artifacts/baseline_v1/`.
