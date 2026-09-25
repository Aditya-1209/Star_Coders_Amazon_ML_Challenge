# CPU baseline: full training candidate corpus

> Historical development report, written before parallel test inference. For the current stopped-run state and desktop workflow, see the project-root README.md.

This is a development experiment using sampled Source 1 businesses. It is not a leaderboard score or a full-test submission.

The model uses 6,000 sampled Source 1 businesses and searches 10,320,219 Source 2/3 records. The index contains the entire training target pool.

## Results

| Partition | Source 1 businesses | Macro F0.5 | Pair precision | Pair recall | Candidate recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 4,198 | 0.9244 | 98.42% | 84.25% | 92.69% |
| tune | 898 | 0.8962 | 97.53% | 80.79% | 91.43% |
| holdout | 904 | 0.9255 | 97.17% | 84.59% | 93.82% |

The threshold **0.605** was selected on the tuning partition. The holdout businesses were not used to fit the classifier or choose its threshold.

- Holdout candidate oracle macro F0.5: **0.9750**. This is the score a perfect classifier could achieve using only the retrieved candidates.
- Always predicting no matches would score **0.0586** on this holdout.
- Holdout singleton accuracy: **92.45%**, across 53 businesses with no matches.
- Mean candidates per holdout business: **46.2**.

| Holdout country | Businesses | Macro F0.5 | Pair precision | Pair recall |
| --- | ---: | ---: | ---: | ---: |
| India | 363 | 0.8843 | 95.12% | 77.71% |
| US | 541 | 0.9532 | 98.37% | 89.06% |

## Error breakdown

- False positive pairs: 77.
- True pairs missed by retrieval: 193.
- Retrieved true pairs rejected by the classifier: 288.

## CPU and memory

Measured on an Apple M4 with 16 GB RAM; CatBoost used six CPU threads and no GPU.

- Development sample preparation: 14.7 seconds.
- Search index construction: 475.8 seconds.
- Candidate retrieval and feature generation for all 6,000 businesses: 309.4 seconds.
- Classifier fitting: 4.6 seconds.
- Training command total through scoring: 315.2 seconds.
- Peak training-process resident memory: 533.0 MiB (not whole-system memory).
- Candidate pairs scored: 276,844.

## Most influential features

- `name_core_ratio`: 15.66
- `address_token_set`: 13.93
- `address_token_containment`: 9.67
- `name_partial`: 7.84
- `first_number_conflict`: 5.97
- `address_token_jaccard`: 5.11
- `number_containment`: 3.91
- `name_token_sort`: 3.63

## Interpretation and next step

The full training target corpus has been evaluated. Review retrieval recall, error rates, and query throughput before full test inference. Country-level results do not establish performance on unseen France.

Full test inference has not been started. At the measured serial retrieval/feature speed, processing all 1,732,544 test anchors would take roughly **24.8 hours**, before additional overhead and changes in country distribution. Parallel inference needs its own benchmark.

## Limits

- The classifier is trained and evaluated on sampled Source 1 businesses, not every labeled business.
- The partitions share an unlabeled search index, but no anchor or labeled positive target crosses partitions.
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

See `code/business_entity_resolution/README.md` for commands, pinned dependencies, and model details. Machine-readable metrics, model weights, pair features, predictions, and errors are in `artifacts/baseline_full_corpus`.
