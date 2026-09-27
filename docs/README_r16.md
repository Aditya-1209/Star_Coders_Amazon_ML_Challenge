# R16: neural matching with a learned no-match decision

R16 is a new experiment for your existing **Mumbai `asia-south1-c` VM:
`g2-standard-32`, one NVIDIA L4 24 GB, 32 vCPUs, 128 GB RAM, 200 GB balanced
Persistent Disk, Ubuntu 24.04 and Python 3.12.3**.

R12 achieved **0.990387533 local macro F0.5** in a measured 12.5-hour L4 run.
Its website score was not included in the published results. **R16 has no
full-data GPU or website score yet.** The 99% website goal is a reporting
objective, not a score this branch claims to have achieved.

## What changes

R16 retains the multilingual E5 encoder, exact GPU retrieval, graph features,
and one cross-encoder. It uses R12's CE sample recipe with all positive records,
lexical/neural/random negatives, and target-address dropout. R13's extra mined
negatives are disabled for this experiment so the final-layer changes are
tested with the established recipe. There is no second neural training pass.

The final layer fits five small tree models sequentially:

1. An **R12-style reference ensemble**, averaging a pair-weighted model and a
   business-weighted model. This is a newly trained reference, not the saved
   model that produced the historical 99.04% result.
2. An **enhanced pair ensemble** with the same two weighting schemes. It adds
   R13's evidence from other confidently matched records and excludes the four
   neural/CE target rank/gap features whose training population is incomplete.
3. A **business no-match classifier**. It predicts whether a business has any
   true match, using fixed aggregates of CE, stage-2, name, address and sibling
   evidence. It sees every business, including those with zero candidates.
   Its labels come from full organizer truth, so missing a true pair during
   retrieval does not incorrectly label that business as a singleton.

The no-match classifier never uses predictions from final models fitted on its
own labels. Its inputs come from earlier models trained on different source
folds. It uses neither entity IDs nor country names as model features. A chosen
business cutoff is applied before exclusive target ownership, allowing a
rejected business's target to remain eligible for a competing business.

## Selection and reporting

Final models train on **6/7**. They early-stop and select one proposal on **3A**.
The search includes keeping the reference, adding the business classifier,
using enhanced pair models, and blending enhanced/reference scores. A cutoff
of zero keeps the business classifier inactive. Ties keep the reference.

Only the best 3A proposal is tested against the reference on **3B**. It must
gain at least **0.0002 macro F0.5**, have a positive approximate one-sided
95% lower bound on paired business-level gain, and lose no more than **0.0005**
in either labeled country. A failed gate selects the R12-style CE reference,
rather than falling back to the weaker graph-only reference used in R13.

Fold **4** is report-only after selection is saved. Evaluation includes
per-country results, false matches on singleton businesses, missing-address
errors and the candidate oracle. Earlier pipeline layers use fold 3, and
previous experiments have inspected fold 4; this is not a fully nested,
untouched validation experiment. France has no labels. Unseen countries use
the learned global pair threshold; no France-specific score is invented.

## Run on the existing VM

Wait for any R9/R12/R13 training to finish. Use a separate clone and the existing
organizer resource path; do not switch the code under a running process.

**Before launch, check the VM's Compute Engine maximum runtime.** An old
12-hour cloud limit can stop this job regardless of its Python setting. For
the default below, set the independent cloud limit to **18 hours / STOP** to
include setup. The new launcher has a **16-hour job limit** and arms a
**17-hour guest shutdown deadline** after setup. These replace the old
11/12-hour settings for R16 only. No cloud setting is changed automatically
by these repository scripts.

```bash
git clone --branch r16 --single-branch https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git Star_Coders_r16
cd Star_Coders_r16
nvidia-smi
bash scripts/vm_r16.sh /absolute/path/to/student_resource
tail -f work/r16/runner.log
```

The resource directory must include `dataset/{train,test}/*.tsv` and
`utils/validate_submission.py`. Keep the existing working GPU driver. The
launcher creates `.venv-r16`, installs the pinned CUDA wheel/dependencies,
checks CUDA/native BF16 and free space, runs the tests and launches detached.
SSH disconnection does not stop training. Setup creates no new cloud resources.

After training exits, including failure, the supervisor brings guest shutdown
forward to ten minutes after exit unless an earlier guest deadline applies.
Persistent Disk retains the outputs. Set `R16_SHUTDOWN_ON_EXIT=0` to disable
the early stop while keeping the overall guest deadline. External cloud
runtime limits are independent and remain in effect.

```bash
cat work/r16/run.json
cat work/r16/result.md
# After interruption, same checkout/data/settings; completed stages are checked:
bash scripts/vm_r16.sh /absolute/path/to/student_resource --resume
```

Resume reuses completed stages; an interrupted active stage starts again. It
does not import R12/R13 work directories or checkpoints. Changes to code/data/
model settings require fresh work. The runtime limit alone may change on resume.

## Resource settings

| Setting | Default |
| --- | --- |
| CPU workers / normalization window | 30 / 750,000 rows |
| Feature shard ceiling | 6,000,000 pairs; whole-business boundaries retained |
| CE precision / optimizer | Native BF16 / fused CUDA AdamW |
| CE train batch / accumulation | 64 / 1; no activation checkpointing |
| CE sample / epochs | 250,000 businesses / up to 3, fold-9 early stopping |
| Encoding / CE score batch ceiling | 1,024 / 1,024; CE scoring reduces on CUDA OOM |
| Active CE token RAM budget | 12 GiB, otherwise memory maps |
| Free-disk reserve | 20 GiB, checked at preflight and during stages |
| Execution | One heavy stage at a time; one GPU |

Compressed Parquet, memory-mapped embeddings and the RAM token cache limit
disk traffic. Pip does not retain duplicate wheel downloads. Existing caches
and previous results are preserved. The reserve is a stop threshold, not an
estimate of total space needed; full R16 storage/peak RAM remain unmeasured.

Environment overrides: `R16_MAX_HOURS`, `R16_THREADS`, `R16_CE_TOKEN_CACHE_GB`,
`R16_ENCODER_PAIRS`, `R16_CE_TRAIN_BUSINESSES`. For a stricter 11-hour job cap:

```bash
R16_MAX_HOURS=11 bash scripts/vm_r16.sh /absolute/path/to/student_resource
```

That cap may interrupt a fresh full run: the measured R12 baseline required
12.5 hours. Budget roughly **12–16 hours** for planning, pending an actual R16
run. This is an estimate based on the R12 timings and extra final tree models,
not a benchmark or completion guarantee. VM time, disk and networking are billed
according to your existing Mumbai configuration; stopped disks may still bill.

To inspect the commands without starting work:

```bash
python3 scripts/run_r16.py --plan
```

## Outputs

`work/r16/result.json` must show `status: complete` and official validation
`PASS` before treating the output as finished. Review `metrics.json` and
`selection.json`; the latter records the reference, proposal and gate result.
Model weights are saved in `work/r16/models` and `work/r16/ce_model` by your run.
This Git branch contains training/inference code, not full trained weights.

Upload **`output/r16/matching_results.tsv`** to measure the website score.
`candidate_pairs.tsv` contains every pair scored by the final matcher, including
pairs later rejected by the business classifier. Every test Source 1 record,
including France and empty-match businesses, remains in both files.

See [the implementation and validation record](../reports/r16_review.md).
## Completed R12 / R16-fast handoff

To compare existing runs without retraining, see
[the frozen R12 evidence exporter](README_r12_evidence_export.md). It exports
final confidence scores with entity/source IDs and verifies saved results.
R16-fast's supplied results retained R12; the full R16 recipe below has no
measured public result from this workspace.
