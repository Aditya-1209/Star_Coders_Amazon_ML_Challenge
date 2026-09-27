# Export completed R12 evidence for comparison

The supplied 27 September handoff reports that R16-fast retained R12: its 3B
gain was 0.00003255, below the unchanged 0.0005 gate, with a negative lower
confidence bound. That result is a successful fallback, not a new measured
improvement. The address-rescue experiment already exists on
`codex/r15-address-rescue` at `78ee707`; its full-data result was not supplied.
R14 code and scores are not present in this repository's remote branches.

The published R12 selection is the mean of its final pair-weighted and
business-balanced classifiers, weight 1, threshold 0.7190439701080322.
`eval_preds_stage3.parquet` and `test_preds_stage3.parquet` are earlier graph
scores, **not the selected R12 model**. R12's completed test cache is
`r10_test_predictions.parquet`; the `r10` name is historical.

`scripts/export_r12_evidence.py` reconstructs frozen final scores for 3A, 3B
and fold 4, then exports the saved final test scores. It does not train, tune,
generate a new submission, change any gate, or alter the VM shutdown timer.
The full R16 training recipe remains a separate experiment; this command does
not launch it.

## Run on the existing VM

First check the actual clock and shutdown schedule. The timestamp in the
handoff is historical; this tool cannot check a VM that it cannot access.
Allow time to download the output before shutdown and the challenge deadline.

```bash
date -u
sudo shutdown --show

# Keep the original R12 checkout/environment/artifacts in place.
# Use this separate checkout, or an existing checkout of branch r16.
cd ~
git clone --branch r16 --single-branch https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git Star_Coders_r16_export
cd ~/Star_Coders_r16_export

"$HOME/Star_Coders_Amazon_ML_Challenge/.venv-r12/bin/python" \
  scripts/export_r12_evidence.py \
  --base-work "$HOME/Star_Coders_Amazon_ML_Challenge/work/r12" \
  --dataset "$HOME/Star_Coders_Amazon_ML_Challenge/student_resource/dataset" \
  --output "$HOME/r12_evidence_export" --threads 30 --device cpu
```

Use a new, separate output directory. The default 30 CPU threads target the
32-vCPU, 128-GB GCP VM from the screenshot; no GPU or new neural scoring is
needed. Full-data export runtime and peak memory have not been measured.
This loads one scoring partition at a time and reuses the completed test
cache; it still reads large feature tables for the labeled partitions.
Avoid running another heavy job concurrently. No packages are installed.

The command checks the original dependency versions, dataset and artifact
fingerprints, completed status and official validation result. It verifies
the recorded 3A score, 3B country metrics and fold-4 metrics, then requires
**byte-identical SHA-256 hashes for both saved official test TSVs**. It checks
parent files again at the end. Changed evidence, invalid IDs or scores, and
reproduction failures stop the export. It refuses to overwrite an export.
A failed output directory retains `_INCOMPLETE`; only a successful export
has `manifest.json` and no `_INCOMPLETE` marker.

## Files and alignment rules

- `3A_scores.parquet`, `3B_scores.parquet`, `fold4_scores.parquet`,
  `test_scores.parquet`: every scored pair, its frozen final confidence, the
  frozen decision flag, source IDs, entity IDs and split/subset membership.
- `3A_truth.parquet`, `3B_truth.parquet`, `fold4_truth.parquet`: full truth for
  each subset, including true pairs that were never retrieved.
- `train_source1_ids.parquet`, `test_source1_ids.parquet`: all businesses,
  including businesses with zero candidates. Training fold/subset membership
  is preserved from the original normalized indices and Polars version.
- `train_target_ids.parquet`, `test_target_ids.parquet`: Source 2 and Source 3
  IDs, original local indices and combined target indices.
- `metadata/`: original run, selection, metrics, result and model/training
  metadata. `manifest.json` records reproduction checks, provenance and output
  checksums. Weights, feature tables and raw business text are not copied.

Across independent runs, join on
`(split, source1_entity_id, target_source, target_entity_id)`. Never join on
`sidx`/`tidx` alone. R14 must be evaluated on the **R12 business-ID partition**,
not a newly hashed row order. Its scores for those businesses must also have
compatible out-of-fold training provenance; identity alignment cannot repair
training leakage.

Use the union of candidate keys with explicit missing-score indicators. Do not
silently inner-join candidates or substitute zero for absent evidence. Compare
score scales and complementary errors on 3A; freeze one proposal there,
retain the 0.0005/positive-lower-bound/country guard on 3B, and report fold 4
only after selection. The earlier experiments inspected 3B/fold 4, so these
are not pristine holdouts. France has no labeled evaluation here. This export
does not establish a 99% public score.

Also preserve the completed R16-fast `selection.json`, `metrics.json`,
`baseline_audit.json`, `run.json`, `result.json`, `result.md` and `logs/` from
`~/Star_Coders_r16_fast/work/r16_fast`. Record `git rev-parse HEAD` and
`git diff --binary HEAD` in that checkout. Those actual run artifacts and the
R14 prediction evidence are still needed to complete the proposed comparison.
