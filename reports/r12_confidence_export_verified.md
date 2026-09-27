# R12 confidence export verified

The existing `r12_export.tar` supplied on 27 September 2026 was inspected
locally, without another VM export, model fit, threshold search or cloud run.
It is sufficient as the R12 side of an R12–R14 comparison when used with the
shared organizer dataset. **No additional R12 export is needed.**

The archive is 413,388,800 bytes. Its SHA-256 is
`d27e504b931b1cc46a17142599b89e71977ab226df4a298aaa9344405154d05c`.
It was received directly, not from the GitHub `r12` branch. The smaller gzip
on that branch is a different artifact containing only final submission rows.

| Archive table | Candidate pairs | Source 1 businesses |
| --- | ---: | ---: |
| `r12_fold3.parquet` | 2,657,314 | 220,699 |
| `r12_fold4.parquet` | 2,657,751 | 220,507 |
| `r12_test.parquet` | 24,284,276 | 1,732,544 |

Every table has `sidx: UInt32`, `tidx: UInt32`, `score: Float32`, `sid: String`
and `tid: String`. Every one of the **29,599,341** pairs was checked:

- Scores are finite, non-null and between 0 and 1; candidate keys are unique.
- Both entity IDs match their numeric indices in the organizer data.
- Source 3's index offset matches the full Source 2 row count for its split.
- Each pair is within one country. Target IDs are globally unique across
  Source 2 and Source 3 within each split.
- Fold membership matches the original UInt32 index hash. Every Source 1
  business in each exported partition has candidates, including businesses
  whose eventual match list is empty.

The archive's `selection.json` and `metrics.json` match the published R12
files at `7ba5312` byte-for-byte. Applying the saved threshold
**0.7190439701080322**, country rules and exclusive target ownership reproduces:

| Partition | Recomputed macro F0.5 | Verification |
| --- | ---: | --- |
| 3A | 0.9904225628407863 | Matches saved selected proposal within floating-point tolerance |
| 3B | 0.9905680734167244 | Both countries match the saved baseline gate metrics |
| Fold 4 | 0.9903875328365094 | Overall, both countries and candidate oracle match published metrics |

The recomputed test TSV is **byte-identical** to the published submission:
97,709,701 bytes; 1,732,544 Source 1 rows; 99,787 empty rows; 5,840,600 matched
pairs. SHA-256:
`418fce53733fc3c01dc4da00efe0409e2a3eb14faa17910cb6cb4b32169e5953`.
The temporary reproduced TSV was removed after verification; it was not
submitted or presented as a new model result.

## Use with the R14 teammate's predictions

Join on **`(split, sid, tid)`**, never independent runs' numeric row indices.
No new mapping export is required: entity IDs are already in every score row,
and the shared organizer data supplies source membership, countries and full
truth, including unretrieved true pairs and singleton businesses.

Within `r12_fold3.parquet`, use the original R12 `sidx` as **UInt32** with
Polars **1.44.2**: `sidx.hash(seed=1010) % 2 == 0` is 3A, and `== 1` is 3B.
Apply those business-ID partitions to R14; do not re-hash R14's own row order.
Resolve equal-score ownership using the original R12 Source 1 indices.

The next experiment is to compare complementary errors on 3A, then freeze one
small score-blend proposal for the unchanged 3B improvement gate. Preserve
the candidate union and handle absent model scores explicitly. R14 must supply
entity-keyed predictions on these same businesses and test records, plus its
training-fold provenance; an in-sample prediction cannot be treated as a
held-out score. Fold 4 remains report-only and was inspected in previous work.

The original producing script, checkpoints and full run manifest are not in
this archive. This audit verifies the supplied rows and reproduces recorded
results and decisions; it does not retrain R12 or establish a new public score.
R16-fast and R15-rescue retained R12, so their fallback submissions add no new
evidence for blending. R14 predictions remain the missing input here.

Full numeric results, file hashes and scope are in
[`r12_confidence_export_verified.json`](r12_confidence_export_verified.json).

## Reproduce the audit locally

The read-only [audit script](../scripts/audit_existing_r12_export.py) uses the
already-extracted archive and shared organizer data. It runs no exporter,
optimizer or threshold search. Use the existing R12 Python environment, with
Polars 1.44.2, and an existing directory for the report. From an `r16` checkout:

```bash
POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /path/to/.venv-r12/bin/python scripts/audit_existing_r12_export.py \
  --export-dir /path/to/extracted/r12_export \
  --dataset /path/to/student_resource/dataset \
  --report /path/to/existing/audit-directory/r12-verification.json
```

Rerunning is optional: the full-data audit reported above already passed.
