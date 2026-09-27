# R15 review and experiment rationale

Base: `r12` at `7ba5312`; neural/training implementation at `06656f9`.
Reviewed parallel experiment: `origin/r13` at `bfb2ef3`.
No remote `r14` branch was available when fetched and checked.

## What the measurements establish

| Run | Local fold-4 macro F0.5 | Website score |
| --- | ---: | ---: |
| R10 stage 2 | 0.987001382 | 0.975, user-reported |
| R11 stage 2 | 0.987234331 | 0.977, user-reported |
| R12 graph reference | 0.988420989 | Not submitted separately |
| R12 CE fusion | 0.990387533 | 0.984, user-reported |

R12 CE validation loss fell 0.054947678 -> 0.043568462 -> 0.042298641.
Final pair precision/recall are 0.998661580 / 0.973913283. The candidate
oracle F0.5 is 0.999081939. These support improving decisions on existing
candidates before expanding retrieval or repeating twelve hours of preprocessing.

Going from 0.984 to 0.99 requires removing 37.5% of the remaining leaderboard
score deficit. This is a substantial unmeasured improvement, not something
established by the local score. France is unlabeled; the observed gap cannot
be attributed entirely to France without additional evidence. The earlier
missing-address breakdown was from older models, not the completed R12 CE.

## Implemented experiment

- Measure aggregate R12 errors on 3A before fitting. Report retrieved-but-rejected
  true pairs, absent candidates, false positives, country and missing-address
  breakdowns. Fold-4 breakdowns are generated only after final selection.
- Add eleven label-free features: four signed alternative features within a
  business/source, and seven features comparing a candidate with the strongest
  CE candidate in the other target source. They distinguish corroborating names,
  conflicting addresses and missing evidence. They never force propagation.
- Exclude incomplete target-population CE/neural ranks and gaps from the new
  final model, as R13 also does. Preserve the pre-existing count/IDF exclusion.
- Fit pair-weighted and business-balanced final XGBoost classifiers on 6/7;
  early stopping and model/blend/threshold choice on 3A. Retain the original
  .0005 gain, positive lower bound and country-regression gate on 3B.
- Compare with the actual frozen R12 CE fusion selection. If the gate fails,
  emit that model's decision, not the older graph reference. Holdout reporting
  cannot change the saved choice.
- Reuse R12's encoder, retrieval, graph and CE scores read-only. Process one
  country at a time, build record lookup joins once per country, and perform
  string similarities in bounded batches. Log exact processed/total counts.
- Supply an isolated runner and GCP launcher with provenance and dependency
  checks, fingerprints for resume, dataset/validator identity checks, lock
  protection, deadline, process-group cleanup, disk reserve and official TSV
  validation. The default is a six-hour limit, with VM shutdown one hour later.

R13 tests hard-negative CE retraining and embedding-based siblings selected by
stage 2. R15 tests CE-selected lexical/address siblings using the already
completed model. No claim is made that one is superior until measured.

## Validation

Completed before commit:

- `python -m unittest discover -s tests_v2`: **64 tests in 235.422 seconds,
  OK (four Linux-only skips)** on Windows/Python 3.12.
- All four skipped launcher tests passed on Ubuntu in 8.752 seconds.
- R15's eight focused tests also passed after the country-join optimization.
- `bash -n scripts/vm_r15.sh` passed on Ubuntu.
- Both `scripts/run_r15.py --plan` and the original GPU-exact R12
  `scripts/run_r10.py ... --plan` passed on Windows and Ubuntu.

The focused R15 tests cover signed margins and tied alternatives; no self
corroboration; missing-address semantics; row-order/label invariance; absent
source handling; fold isolation; excluded features; actual XGBoost training
with missing features; and exact R12 fallback behavior. Preflight tests reject
changed neural folds, data and dependencies.

The existing offline pipeline test now runs actual R15 feature construction,
fitting, selection, reporting and inference after a tiny BERT/graph run. Both
R15 and fallback outputs pass the official validator. Fallback TSV bytes equal
the R12 output. Evaluation leaves selection unchanged, and parent artifacts
remain unchanged.

Linux launcher tests use fake Python/shutdown executables to test failure
before shutdown, detached launch, quoted dataset paths, and duplicate locks.
No VM is started and no real shutdown is scheduled by the tests.

Full-size R15 runtime, GPU peak memory, holdout gain and website gain are
unmeasured. The cloud run is required to evaluate this experiment.
