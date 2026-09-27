# Evidence and change rationale

User-supplied completed R15 selection/audit, 27 September 2026:

- R12 3A macro F0.5 .990422563, precision .998707706, recall .974039422.
- Best R15 3A trial: pair model, .75 blend, cutoff .70960015, F0.5 .990505699.
- Gate gain .0000494283, one-sided 95% lower bound -.0000383699,
  minimum .0005; country guard passed. R15 fell back for inadequate/uncertain
  overall improvement, not a crash or a country-guard failure.
- India 3B recall rose .976447762 -> .976835805 but precision dropped
  .998782334 -> .998648503, and macro F0.5 declined slightly.
- US recall rose .972304998 -> .972771156 with nearly unchanged precision;
  its macro gain remained small. A global new pair model changed many scores
  without fixing enough errors to beat the established reference.
- On 3A, false negatives 9,928, retrieved rejected 8,743, missing-address
  retrieved rejected 7,206 (82.42%), false positives 482, missing-address false
  positives 182, never retrieved 1,185. The earlier 72% figure is superseded
  for this measured R12 3A audit; it is not a France/public error estimate.

Implemented response: specialize fitting/scoring to missing target addresses,
rather than another whole-population refit. Learn trusted sibling signals only
when the other-source donor is high-scoring and unambiguous within its source.
Preserve all other R12 candidate scores; tune the decision only on 3A and gate
one chosen proposal on 3B. Do not add split-size-sensitive counts/IDF, change
folds, reduce the original .0005 acceptance gate, or tune on fold 4. Full neural
training/retrieval/scoring is reused, not repeated. Two small trees are fitted.

This targets a different hypothesis from R16-fast's whole-business rejection.
Neither hypothesis has a full-scale measured improvement yet. Repeated
experiments already exposed 3B/4, so the inherited split is not pristine and
local improvements cannot guarantee .99 on the website. The 1,185 unretrieved
pairs cannot be corrected by this classifier.

Validation: Python 3.12 `python -m unittest discover -s tests_v2` passed 72
tests in 237.985 seconds (six Linux-only skips on Windows). Two new Linux
launcher tests passed separately on Ubuntu in 2.249 seconds. Six focused
specialist tests passed, including trusted-donor ambiguity, label/order
invariance, scoring scope, actual tree fitting and fold/feature exclusions.
The real offline tiny-BERT pipeline exercised all rescue stages and both
selection paths; outputs passed the official ID-aware validator, preserving
France/empty rows. R12 fallback TSV was byte-identical, fold-4 evaluation left
the saved selection unchanged, and parent artifacts were unchanged.
Both fast plans (Windows/Ubuntu), the original `run_r10.py --plan`, shell
syntax and `git diff --check` passed. No full-data GPU or public score is
claimed; test artifacts are synthetic and do not measure challenge accuracy.
