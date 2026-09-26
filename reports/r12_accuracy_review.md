# R12 accuracy review — 27 September 2026 (IST)

Objective: improve upon the user-reported r11 leaderboard score of **0.977**,
with **0.99** as the target. Hardware: GCP g2-standard-32, 32 vCPUs, 128 GB RAM,
NVIDIA L4 24 GB, Ubuntu 24.04 / Python 3.12.

## Evidence from completed runs

Reviewed the files in
[r11-logs commit 6bdf4ef](https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge/tree/6bdf4ef/reports/r10_r11_run_logs).
The log README still says r11 leaderboard "pending"; the user subsequently
reported 0.977. No completed CE result is present.

| Artifact | Fold-4 macro F0.5 | Leaderboard | Interpretation |
|---|---:|---:|---|
| r10 stage 2 | 0.9870013821 | 0.975, user-reported | Neural retrieval produced the large improvement |
| r11 stage 2 | 0.9872343313 | 0.977, user-reported | Competition features helped online more than on holdout |
| r10 graph stage 3 | 0.9876994717 | Not submitted in shared logs | Sibling-record evidence helped holdout |
| r11 retrieval oracle | 0.9987379748 | Not a leaderboard measurement | Little room remains for enlarging retrieval blindly |
| r11 post-pruning oracle | 0.9987012196 | Not a leaderboard measurement | Rescue k=8 already retains almost all oracle value |

The shared analysis reports 89% of missed true pairs were retrieved but
rejected, and 72.5% of those lacked a target address. The CE log stops at
80,016 / 1,433,164 examples in its first epoch; neither an epoch validation
score nor a completed cross-encoder submission was recorded. The classification
stage, not a larger neighbour list, is the main untested opportunity.

The shift report gives enhanced-feature AUC 0.64783 (India) and 0.69182 (US).
This audit covers the earlier pair features; it does not establish that the
new graph/CE competition features or France have no distribution shift.
France has no labels. Its inferred score is an assumption, not a measured
country result.

## Changes and why

1. **Retain all CE positives for each sampled business.** The old sampler sorted
   target IDs and kept four positives. Source 2's global IDs precede Source 3,
   so this systematically discarded later Source 3 matches and valid variants.
   Business sampling and business-balanced weights already bound and balance
   training. All positives now remain; negative sampling is unchanged.
2. **Match actual missing-address inputs.** Previous augmentation dropped both
   addresses for an entire batch and removed the internal field separator.
   The supplied Source 1 data has no blank addresses. Independent per-pair
   target-address dropout now preserves Source 1 and the exact token template
   produced for a naturally blank target. Pair-order augmentation remains.
3. **Fix fold-dependent CE context.** Previously `frame_for` filtered folds
   before CE target ranks/gaps were computed. A pair therefore saw a different
   competitive context when fitting, selecting, and reporting. Ranks/gaps now
   use the complete available CE score file before selecting feature rows.
   Context uses predictions only; it does not read other folds' labels.
4. **Keep graph name competitors outside graph folds.** Graph fitting still
   uses its original folds 6/7, with 3 for tuning and 4 for reporting. Its
   record-name/address context now also includes direct candidates from the
   other businesses, using only pair keys and lexical similarities. Otherwise
   a record could appear unique simply because its owner was not in the graph
   fold subset. Generated graph candidates are unioned without duplicate keys.
   This does not generate graph hops for the excluded folds or promise identical
   train/test population density.
5. **Use L4 memory more efficiently during CE inference.** Within each bounded
   window, score similarly sized token pairs together and restore scores to
   their original pair keys. On CUDA OOM, halve the batch and retry those same
   pairs; preserve the reduced ceiling for following windows. No text is
   truncated, no pair is dropped, and there is no silent CPU fallback.
6. **Report the requested target accurately.** The VM profile sets the local
   reporting objective to 0.99. This flag has no effect on model fitting,
   threshold tuning, or the selection gate; leaderboard remains unmeasured.

The existing split-size-dependent count/IDF exclusion and selection gate are
unchanged. Encoder folds remain 0/1/8/9; CE optimization folds remain 0/1/8,
with fold 9 early stopping. No fold-4 labels enter fitting or selection.

## Audit of the supplied student data

Re-read the actual training Source 1 IDs and ground-truth TSV, using the same
Polars fold hash and CE business-sampling seed as the pipeline:

| Quantity | Measured count |
|---|---:|
| Source 1 training businesses | 2,206,821 |
| Encoder-eligible businesses, folds 0/1/8/9 | 882,552 |
| Eligible businesses with a positive match, before hard-negative filtering | 833,203 |
| Full-data labelled pairs | 7,638,365 |
| Pairs a four-positive cap would omit across the full data | 942,805 (12.34%) |
| CE training business sample, folds 0/1/8 | 250,000 |
| Positive pairs retained by the old cap in that sample | 757,711 |
| Positive pairs retained by the revised sampler | 863,736 |
| Additional true training pairs | **106,025** |

The 2M encoder cap cannot create 2M unique-business examples from these data.
The old 1M cap was already non-binding. Keep one epoch rather than claiming
that a larger cap makes the encoder more accurate. Keep neural k=24 / rescue
k=8 because the measured retrieval and post-pruning ceilings are already high.
The CE sample remains 250k businesses, batch 64, up to three epochs selected by
fold-9 loss. Retaining all positives adds training work; runtime is not measured
on the L4. The 16-hour allowance is a limit, not an ETA.

## Validation and remaining measurement

Completed before committing:

- `python -m unittest discover -s tests_v2`: 54 tests in 158.820 seconds;
  52 passed on Windows/Python 3.12, with the two Linux-only launcher tests skipped.
- Both Linux launcher tests passed on Ubuntu 24.04 / Python 3.12 in WSL
  (4.326 seconds), including dependency-failure recovery and duplicate-run locks.
- The full suite includes real offline tiny-BERT optimization, serialization,
  neural scoring, graph/tree fitting, frozen selection and submission validation.
- Five new regression tests cover positive coverage/fold isolation, target-only
  dropout token equivalence, CE fold-invariant features, graph competitors from
  outside its folds, and scoring order plus simulated CUDA OOM retries.
- Linux `bash -n scripts/vm_r12.sh` and the GPU-exact r12 runner `--plan`
  passed with the 16-hour allowance and 0.99 reporting objective.
- `git diff --check` passed.

CUDA OOM handling was tested by simulation. No L4 throughput, peak memory,
full Linux neural run or numerical GPU/CPU equivalence measurement is claimed.

No full-data model was trained in this local review, and no new holdout or
leaderboard score is claimed. The real VM run must measure stage-2, graph and
selected final performance. Keep the known 0.977 submission available. A better
holdout alone does not prove a better France or leaderboard score.
