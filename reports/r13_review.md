# R13 implementation and validation record

Base: `origin/r12`, commit `06656f97e5fef60f3e976349228fa7b8f3df59a6`.
Objective: website macro F0.5 >= **0.985**. Status: **not yet measured**.

## Evidence behind the changes

The checked-in R12 review identifies classification, missing addresses and
competition context as remaining issues. The published R11 logs on
[`r11-logs`](https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge/tree/r11-logs/reports/r10_r11_run_logs)
report stage-2 fold-4 macro F0.5 `0.9872343313262878`, precision about `0.9974`,
recall about `0.9678`, and a pre-pruning candidate-set oracle about `0.99874`.
The R12 review records the user-reported R11 website score as `0.977`.
These are earlier experiments, not R13 measurements.

The R11 error audit reports that roughly 89% of missed pairs had been retrieved
and rejected, and about 72.5% of those had a missing target address. This motivates
better classification and support from other records rather than a blanket
increase to candidate counts. No completed R12 CE/website result was available
for this review, so R13 does not claim a measured improvement over it.

## Implemented experiment

1. Mine CE negatives from stage-2 predictions on CE folds 0/1/8/9, which stage 2
   did not fit. Retain all positives, explicitly remove them before mining,
   preserve lexical/ANN/random diversity, and cap negatives at five/business.
2. Reuse the trained multilingual target embeddings to measure similarity to
   at most three confidently owned records of a business. Exclude self support,
   reject ambiguous ownership and provide zero evidence for unsupported pairs.
   Read embeddings through memory maps with 8,192-pair gathers.
3. Exclude graph/CE target rank/gap features from the final model because their
   training candidate population contains only four source folds. R12's global
   CE join fixes inconsistent fold loading but does not fix that population
   difference. Within-business CE features and full-pool lexical competition
   remain available.
4. Carry the new artifacts into checked resume, use the inherited validation
   gate/reference fallback, and add report-only missing-address error analysis.
   Record the website target separately, with no fabricated achievement flag.

Defaults keep R12's L4 job deadline and provide the previous desktop's smaller
batches and feature shards. The new sample strategy can retain more distinct
negatives than R12 after deduplication; its per-business ceiling is unchanged.
Neither GPU throughput nor full-data peak RAM has been measured for R13.

## Validation on 2026-09-27

Python 3.12 on macOS ARM, PyTorch 2.14.0, Transformers 5.17.0, FAISS 1.15.1,
XGBoost 3.4.1, Polars 1.44.2 and NumPy 2.5.3. Neural packages were installed
in a temporary test directory; the existing project environment was unchanged.

```text
python -m unittest discover -s tests_v2 -v
Ran 64 tests in 66.251s
OK (skipped=4)
```

60 passed, including the existing R10/R12 regressions and a new R13 end-to-end
fixture with real tiny-BERT training/save/reload, FAISS retrieval, XGBoost,
graph expansion, mined negatives, streamed sibling features, final model
fitting/selection and official submission-format checks. Both selected-model
and reference-fallback outputs were validated, including France and unmatched
business rows. The tiny model/dataset test checks execution, not challenge
accuracy. Fold-4 evaluation leaves the selection file unchanged.

Additional tests verify that mining ignores other-fold predictions/labels,
never turns a positive into a negative, retains diverse negatives within its
cap, and keeps business weighting. Sibling tests cover self exclusion, ties,
competing ownership, no support, negative cosines and batch-size invariance.
Feature attachment rejects missing evidence; final fitting excludes incomplete
target-context features. Runner tests exercise both hardware profiles and
unchanged R10 defaults. `bash -n scripts/vm_r13.sh` and desktop/VM command plans
also passed.

The four skipped tests are R12/R13 Linux launcher contracts requiring `flock`
and `sha256sum`, unavailable on this Mac. They run with fake installation,
CUDA and shutdown commands on Linux; no paid VM or real shutdown was launched
for this review. CUDA kernels, actual GPU memory limits and full GPU training
remain untested here. The pinned PyTorch 2.14.0 Python 3.12 Linux/Windows CUDA
12.6 wheels were checked against the [official wheel index](https://download.pytorch.org/whl/cu126/torch/).

No full R13 submission is included. A completed GPU run, its per-country and
missing-address diagnostics, and an actual website score are required before
accepting the 98.5% objective as achieved. The reference is a rebuilt R12 graph
model, not a previously submitted R12 artifact. Earlier stages use fold 3,
past runs have inspected fold 4, and France lacks labeled validation examples.
