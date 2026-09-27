# R16 fast CE fusion — completed scores only

**Entry point: `scripts/vm_r16_fast_fusion.sh` on `codex/r16-fast-ce-fusion`.**
This is a new, small XGBoost fusion model using R12 and separate R14 neural
predictions. It reuses `.venv-r12`, installs nothing, and has no encoder,
retrieval, feature-generation, CE-training, or CE-scoring stages. The older
`r16` branch at `fcb5a53` remains an audit/full-training branch; its
`vm_r16.sh` is not this launcher.

**Current blocker:** the supplied `D:\star_r14\fusion_ab\r14_fold3.parquet`,
`r14_fold4.parquet`, and `r14_test_scores.parquet` are final, cross-fitted
fusion outputs. They are **not valid inputs to this experiment**. The original
CE-A, swapped CE-A, CE-B and graph predictions are needed. Their locations
have not yet been supplied or inspected. No new R12 export is needed.
There is no trained full-data R16 fusion checkpoint or measured accuracy gain yet.
The VM has no external IP: use the [IAP/Drive handoff](r14_iap_transfer.txt)
for the existing raw files. [Transfer readiness](../reports/r14_prefusion_transfer_status.md)
is tracked separately from the published model code.

## Why this experiment differs

The teammate already tried the final R12/R14 score blend: 0.65/0.35 selected on
fold 3, reporting about 0.99051 on fold 4 versus 0.99044 R12 (+0.00007), with
no France change. Those are teammate-reported figures, not a reproduction of
this repository's frozen R12 0.990387533. That blend does not establish the
required 0.0005 improvement on the frozen 3B gate.

R14's final fusion is cross-fitted on fold 3. Even when each business's score
is out of fold, scores on 3A can depend on a fusion model trained with 3B
labels. Choosing a new proposal using those 3A scores would let 3B influence
proposal selection. This branch therefore uses **pre-fusion predictions**.

The new tree learns disagreements between R12, the donor graph, CE-A,
swapped CE-A and CE-B, including neural agreement/spread, availability, and
within-business rank/gap. It fits a depth-4, strongly regularized,
business-balanced model on **R12 3A only**. An internal business split within
3A chooses up to 400 rounds; the chosen round count is refit on all 3A.
No entity ID, country, fold or label is a model feature.

Selection tests fusion weights 0.1, 0.25, 0.5 and 1 on 3A, with R12 retained
on ties. Candidate sets are united by entity ID. R12-only candidates retain
their R12 score; donor-only candidates use fusion; shared candidates blend.
Missing CE evidence stays missing, never a zero score or a negative label.
One selected proposal is tested by the **unchanged `r10.gate`**:

- gain at least **0.0005 macro F0.5**;
- positive approximate one-sided 95% lower bound;
- neither labeled country regresses by more than **0.002**.

There is no command-line option to relax these thresholds. Selection is saved
before fold 4 is evaluated. Rejected proposals produce the **original R12
submission and candidate TSVs**, with both SHA-256 hashes checked against
`work/r12/result.json`. Do not resubmit that fallback.

The inherited graph was tuned on fold 3; R12 also has inherited validation
reuse, and multiple earlier experiments inspected these partitions. This
preserves the existing gate protocol, not a new untouched holdout. France has
no validation labels. No 99% website score is claimed.

## Required existing artifacts

1. The completed original `~/Star_Coders_Amazon_ML_Challenge/work/r12`:
   `run.json`, `result.json` (official validation PASS and both output hashes),
   `selection.json`, `metrics.json`, and six `norm/{train,test}_source{1,2,3}.parquet`
   files. If the original selection was `reference`, also
   `models/stage3_metrics.json`. These files remain read-only.
2. The **existing** R12 archive extracted to `~/r12_export`, containing
   `r12_fold3.parquet`, `r12_fold4.parquet`, `r12_test.parquet`, `selection.json`
   and `metrics.json`. The verified archive is 413,388,800 bytes, SHA-256
   `d27e504b931b1cc46a17142599b89e71977ab226df4a298aaa9344405154d05c`.
   It contains 29,599,341 scored rows. Reuse it; do not rerun an exporter.
3. The original organizer `student_resource/dataset` and
   `student_resource/utils/validate_submission.py`, at the paths recorded by
   R12. The runner verifies their fingerprints and the original Polars version.
4. The existing `.venv-r12/bin/python` with its working CUDA XGBoost, NumPy,
   Polars, PyArrow and filelock. No new environment is created.
5. R14's **pre-fusion** graph, CE-A, swapped CE-A and CE-B predictions for
   fold 3, fold 4 and test, plus `inputs.json` describing their true paths,
   column names, probability/logit scale and provenance. Use
   [the manifest template](r16_prefusion_inputs.example.json).
   This is a manifest for existing predictions, not a request to train again.

Each component can have `sid`, `tid`, `score` (or a declared score column).
Index-only files are also supported, but require the **original R14** six norm
Parquets in the manifest's `norm_dir`. Those are from `D:\star_r10b\work\norm`,
not R12. The loader gathers IDs in the original row order (Source 2 followed
by Source 3), then joins IDs into R12's mapping. It never equates the two runs'
indices. The original R14 train Source1 ID row order is required even for
ID-keyed score files (set `train_source1_ids` if omitting `norm_dir`). The runner
verifies that donor training-fold membership matches R12 after ID alignment;
otherwise the numeric provenance could conceal training on R12 gate labels.
Unknown/duplicate IDs, wrong R12 folds, cross-country pairs,
non-finite scores and CE candidates outside the donor graph fail the run.
CE band omissions are allowed. Files must contain only their declared split/fold.

Provenance is a producer declaration, not proof from score values: CE-A and
CE-B fit folds 0/1/8 and select/early-stop on 9; graph fits 6/7 and selects on
3. The final cross-fitted `fusion_a.json`/`fusion_b.json` models are not used.
Do not relabel final fusion scores as a raw component.

## Runtime on the existing Mumbai VM

Target: **g2-standard-32, 32 vCPU / 128 GB RAM, L4 24 GB, Ubuntu 24.04,
Python 3.12, asia-south1-c**. Default: 24 CPU threads, CUDA histogram training,
400-round maximum, 500,000-pair inference batches, one stage at a time, and
20 GiB free-disk reserve. Full RAM/VRAM peaks have not been measured.

Budget **10–30 minutes after all inputs are present**, excluding file transfer.
This is an estimate from the limited work performed, **not a measured full-data
L4 runtime**. Selection failure avoids reading donor test scores, which can
shorten the run. Default hard wall cap is **45 minutes** (at most 60 allowed).
The runner also stops **10 minutes before the supplied VM shutdown**.
For `2026-09-27T17:44:11Z`, its latest deadline is **17:34:11 UTC**. It refuses
an expired deadline or a start with less than ten working minutes left after
that buffer. Completion is not guaranteed within the cap; interrupted stages
can be resumed with the same checked inputs on a later VM session.

No shutdown command is issued; the existing 17:44:11 UTC timer stays intact.
The shell launcher runs in the foreground; the `nohup` command below detaches
it. Do not run alongside another heavy model job.

## Launch after the pre-fusion files are available

These commands assume the required R14 files/manifest have actually been
transferred to `~/r14_prefusion/inputs.json`. They are **blocked with only the
currently described `fusion_ab` final files**. Inspect/update the manifest with
real filenames; do not invent missing component paths or score provenance.

```bash
git clone --branch codex/r16-fast-ce-fusion --single-branch \
  https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git \
  "$HOME/Star_Coders_r16_fast_ce_fusion"
cd "$HOME/Star_Coders_r16_fast_ce_fusion"
export R16_BASE_WORK="$HOME/Star_Coders_Amazon_ML_Challenge/work/r12"
export R16_PYTHON="$HOME/Star_Coders_Amazon_ML_Challenge/.venv-r12/bin/python"
export R16_R12_EXPORT="$HOME/r12_export"
export R16_R14_PROVENANCE="$HOME/r14_prefusion/inputs.json"
export R16_SHUTDOWN_AT="2026-09-27T17:44:11Z"
RESOURCE="$HOME/Star_Coders_Amazon_ML_Challenge/student_resource"

# Exits before fitting if artifacts/provenance/environment/time are incompatible.
bash scripts/vm_r16_fast_fusion.sh "$RESOURCE" --preflight

# Run this only if preflight passed. The real run repeats preflight itself.
nohup bash scripts/vm_r16_fast_fusion.sh "$RESOURCE" \
  > r16_fast_ce_fusion.log 2>&1 < /dev/null &
tail -f r16_fast_ce_fusion.log
```

Use R12's actual organizer resource path if its `run.json` records a different
one. Do not copy/renormalize R12 caches or reset its work directory. If only the
existing tar is present, extract it once into a separate empty directory and
point `R16_R12_EXPORT` at the directory holding the five files; this does not
regenerate predictions. Avoid overwriting an already extracted copy.

For the four-stage plan without touching data, add `--plan`. For checked
resume, use the same command with `--resume`. The deadline must reflect the
new session if the original VM has already stopped. The runner verifies
code/settings/input identities and completed-output fingerprints before reuse.
It will reject unrelated work directories and paths that overlap source artifacts.

Outputs: `work/r16_fast_ce_fusion/{fusion,selection,metrics,run,result}.json`,
`result.md`, stage logs, and
`output/r16_fast_ce_fusion/{matching_results,candidate_pairs}.tsv`.
Official validation includes `--check-ids`. Wait for `run.json` status
`complete` and `result.json` official validation `PASS` before using outputs.

Preserve the small audit records plus outputs for transfer:

```bash
tar -czf "$HOME/r16_fast_ce_fusion_results.tar.gz" \
  work/r16_fast_ce_fusion output/r16_fast_ce_fusion r16_fast_ce_fusion.log
```

From a machine authenticated to the GCP project, download through IAP using
the existing matching private-key file locally (replace only its file path):

```bash
gcloud compute scp --tunnel-through-iap \
  --project=amazon-ml-r9 --zone=asia-south1-c \
  --ssh-key-file=/path/to/existing/matching_private_key \
  'akshayvijaygudur_gmail_com@amazon-r9-train:~/r16_fast_ce_fusion_results.tar.gz' .
```

The VM has no external IP; a direct Internet `scp` connection to its internal
hostname is not a working route from this Mac.

No new R12 export, full-model run, or website submission is performed here.
