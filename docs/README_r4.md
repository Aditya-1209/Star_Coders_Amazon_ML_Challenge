# r4: accuracy-focused entity resolution

**Status: code changes only. No training, inference, benchmarks, or tests were
executed for this branch.** Regression tests are included for the teammate who
will run it. Existing `models/v2/metrics.json` describes the earlier model and
does not measure r4.

The older root `scripts/setup.sh` and `scripts/run_full_inference.py` run the
CatBoost baseline. Use the commands below for the current XGBoost pipeline.

## Changes worth knowing before running

- **Broader retrieval:** union the original combined top 64 with the top 16 by
  name evidence and top 8 by address evidence. This keeps every original hit
  and offers up to 88 candidates before model pruning. It targets good matches
  pushed down by common names or shared addresses; it cannot recover a pair
  that shares no retained blocking key. Both train and test use the same rule.
- **More discriminative features:** 63 pair features (previously 43), including
  original-script name similarity, numeric business-name conflicts, house-number
  and postcode evidence, explicit missing-address flags, and corpus name
  frequency. Raw addresses preserve leading-zero postcodes. These are learned
  features, not hard rejection rules. Missing text never scores as a perfect
  string match. Name frequencies use unlabelled records only.
- **Additional training data and an ensemble:** stage 1 fits one model on folds
  0/8 and another on 1/9. Each predicts the other group's rows; unseen rows use
  the mean of both. Folds 8/9 were previously unused by the classifiers. Both
  models are saved and used at inference. Early stopping uses tuning fold 3.
- **Metric-specific decisions:** find the best probability cutoff directly for
  exclusive macro F0.5, including singleton businesses and records with no
  retrieved candidates. Equal scores enter together. This replaces the old
  0.025 threshold grid. A saved threshold just above 1 means no matches.
- **Validate the graph's contribution:** tune stage-2/stage-3 mixtures with
  stage-3 weights 0, 0.25, 0.5, 0.75, and 1 on fold 3. Stage 2 alone is included
  and wins ties. The selected rule is frozen before fold 4 is reported; there
  is no guarantee that a tuning-fold improvement transfers to holdout/test.
- Empty blocking results, zero graph anchors, and candidates without supporting
  peers retain typed schemas rather than failing in concatenation or scoring.
- Equal blocking scores and equal target-ownership scores have explicit tie
  breakers, so hash-group iteration order does not choose winners.
- Training and inference accept `--device cpu|cuda`, `--threads`, and
  `--batch-rows`; dense prediction matrices default to at most 250,000 rows.
- Stage 3 filters stored features before collecting them, computes new-pair
  features in batches, and groups support comparisons by 5,000 businesses
  instead of 300,000. Full normalized tables, candidate graphs, and training
  matrices still occupy memory: this is not an out-of-core training rewrite.
- **Regenerate candidates/features and retrain all three stages.** Model
  metadata must have `feature_version: r4-accuracy-1`. Older stage-3 models and
  the shipped stage-2 weights do not implement these accuracy changes.
- Two-hop expansion keeps at most ten neighbors even when the anchor itself
  was not retrieved. Model-directory selection now also selects the stage-2
  reference threshold; no unrelated-directory fallback remains.
- Small validation folds no longer attempt to sample two million rows. A best
  iteration of zero correctly means one tree. Duplicate evaluation rows are
  treated as sets rather than inflating true-positive counts.
- Newly generated feature folders have a completion marker and shard manifest.
  Interrupted or changed shard sets are rejected; generation refuses to mix
  files with an existing directory. Legacy folders without a manifest can still
  be read, but a fresh r4 workspace is recommended.

These changes target candidate recall, pair discrimination, model variance, and
the challenge's scoring objective. They are not a measured accuracy improvement:
the teammate must run the workflow and compare holdout results before claiming
one. More candidates/features and larger training folds increase compute and
memory requirements; the earlier v2 runtime/RAM measurements do not describe
this version.

## 1. Check out r4 and install the correct dependencies

From an existing clone:

```text
git fetch origin
git switch r4
git pull --ff-only origin r4
```

For a fresh clone:

```text
git clone --branch r4 https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git
cd Star_Coders_Amazon_ML_Challenge
```

Run all following commands at the repository root. Python 3.12 is the intended
version. Native Windows is supported by the v2 modules; they do not import the
legacy POSIX checkpoint runner. A CUDA-capable XGBoost installation and NVIDIA
driver are needed for `--device cuda`. Use `--device cpu` on the Mac or when a
working CUDA environment is unavailable.

PowerShell:

```powershell
py -3.12 -m venv .venv-r4
.\.venv-r4\Scripts\Activate.ps1
python -m pip install -r code/business_entity_resolution/requirements_v2.txt
$env:PYTHONPATH = "code/business_entity_resolution/src"
$env:PYTHONUTF8 = "1"
$env:POLARS_MAX_THREADS = "12"
```

Linux/macOS:

```bash
python3.12 -m venv .venv-r4
source .venv-r4/bin/activate
python -m pip install -r code/business_entity_resolution/requirements_v2.txt
export PYTHONPATH=code/business_entity_resolution/src
export PYTHONUTF8=1 POLARS_MAX_THREADS=12
```

The commands below use one line each and work in either shell after setup.

## 2. Dataset and quick regression checks

Use the organizer ZIP, supplied separately from GitHub. If it has not already
been extracted, this also extracts the official validator bundled in the ZIP:

```text
python -m zipfile -e amazon_ml_dataset.zip .
```

Required inputs are under `student_resource/dataset/train/` and
`student_resource/dataset/test/`. The official validator is
`student_resource/utils/validate_submission.py`.

Run the small new regression suite before a long job. This command is provided
for your PC; it has **not** been run as part of preparing r4:

```text
python -m unittest discover -s tests_v2 -v
```

It uses synthetic records, CPU operations, and a fake predictor. It does not
train a model or require the challenge data or GPU.

## 3. Fresh r4 training workflow

Use **`work/r4_accuracy`** so the previous run and shipped weights stay intact. Keep this
workspace on an SSD and leave ample space for normalized data and feature shards;
reserve roughly 60 GB if possible. This is a planning allowance, not a measured
r4 disk requirement. The work directory and outputs are ignored by Git.

```text
python -m er_v2.translit --out work/r4_accuracy/models/translit.json
python -m er_v2.prepare --work work/r4_accuracy --workers 12 --translit work/r4_accuracy/models/translit.json
python -m er_v2.run_block --work work/r4_accuracy --split train --top-k 64 --name-k 16 --address-k 8
python -m er_v2.run_block --work work/r4_accuracy --split test --top-k 64 --name-k 16 --address-k 8
python -m er_v2.run_features --work work/r4_accuracy --split train --shard-pairs 2000000 --workers 12
python -m er_v2.run_features --work work/r4_accuracy --split test --shard-pairs 2000000 --workers 12
python -m er_v2.train --work work/r4_accuracy --model-dir work/r4_accuracy/models --device cuda --threads 12 --batch-rows 250000
python -m er_v2.stage3 --work work/r4_accuracy --model-dir work/r4_accuracy/models --split train --device cuda --threads 12 --batch-rows 250000 --support-anchors 5000
```

For a different dataset root, pass `--dataset /path/to/dataset` to `translit`,
`prepare`, `run_features`, `train`, and training-mode `stage3` consistently.
Use the same learned transliteration map for all train and test normalization.
Do not combine freshly normalized row indices with candidates from another run.

Stage 1 fits groups 0/8 and 1/9; stage 2 fits folds 2/5; stage 3 fits folds 6/7.
Fold 3 is used for early stopping, threshold and blend tuning; fold 4 is used
only for reporting. The learned transliteration map
excludes folds 3/4. `stage2_train.parquet` includes scores for all folds, including
in-sample scores for stage-2 fitting folds; stage 3 selects only folds 3/4/6/7.
Graph construction for training uses this selected subgraph, while test uses all
test businesses. That distribution difference remains a limitation of the
existing modeling protocol, not something r4 has measured away.

Inspect `work/r4_accuracy/models/metrics.json` and `stage3_metrics.json`:

- `fold4_retrieval_oracle` vs `fold4_oracle` in stage-2 metrics measures the
  ceiling before vs after stage-1 pruning. The `pair_recall` fields expose
  candidates lost at each step.
- `fold4_stage2_excl` is the new stage-2 result. The historical 0.9494 describes
  the old model, not this run.
- `fold4_selected`, `fold4_stage3`, and `fold4_stage2_ref` in stage-3 metrics
  compare the selected mixture, pure graph model, and stage-2 reference on the
  same evaluated businesses. `tuning_trials` records every candidate mixture.
- `eval_preds.parquet` and `eval_preds_stage3.parquet` retain evaluation scores
  in the workspace for later analysis; selection uses only fold 3.

For a retrieval ablation, use a separate workspace with `--name-k 0 --address-k 0`
on **both** blocking commands, keeping other settings the same. Use fold 3 to
choose settings before inspecting fold 4; repeated optimization on fold 4 turns
it into another tuning set. There are no labelled France examples, so neither
validation score establishes French test accuracy.

## 4. Predict and validate

Stage 2 creates `work/r4_accuracy/test_preds.parquet`, which stage 3 needs:

```text
python -m er_v2.predict --work work/r4_accuracy --model-dir work/r4_accuracy/models --output output/r4_accuracy-stage2 --device cuda --threads 12 --batch-rows 250000
python -m er_v2.stage3 --work work/r4_accuracy --model-dir work/r4_accuracy/models --split test --output output/r4_accuracy --device cuda --threads 12 --batch-rows 250000 --support-anchors 5000
python student_resource/utils/validate_submission.py --matching output/r4_accuracy/matching_results.tsv --candidate output/r4_accuracy/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

The final selected-model files are under **`output/r4_accuracy/`**. The candidate file is exactly
the union scored by stage 3, not the earlier 64-candidate blocking output. Empty
Source 1 rows remain present. Each TSV is written through a temporary file and
renamed after completion; the two files are not a single atomic transaction, so
wait for successful command exit and validation before using either file.

Treat any validator warning about matches outside the candidate set as a bug.
The official validator holds ID mappings in memory; run it after inference exits
so the GPU models and graph frames have been released.

## Memory pressure, interruptions, and packaging

For lower scoring-memory use, reduce `--batch-rows` to 100000 and
`--support-anchors` to 1000. Reduce feature shards to 1000000 pairs in a fresh
workspace if feature generation exhausts RAM. If stage-1 training does not fit,
add `--no-extra-stage1-folds` to `er_v2.train`: it keeps both models but uses
only folds 0 and 1, trading the additional training data for lower peak memory.
This switch does not reduce stage-2 or graph memory. `--threads` controls XGBoost and
stage-3 RapidFuzz threads; `POLARS_MAX_THREADS` must be set before Python starts.
None of these limits eliminate the memory used by the global candidate graph or
the full training matrices.

The v2 pipeline does not have the CatBoost runner's automatic resume. Do not
resume a feature folder containing `_INCOMPLETE` or mix new files with old
shards. Preserve the old folder for diagnosis and generate in a fresh workspace,
or deliberately remove only the failed feature folder after stopping its process.
Training and prediction can then be restarted from completed inputs.

Before packaging an r4 submission, update `Documentation_template.md` with the
actual stage-3 method, new measured scores, candidate counts, and validation
result. The existing write-up describes earlier stage-2 results. The packager
checks ZIP integrity, not prediction validity; validate first:

```text
python -m er_v2.package --output output/r4_accuracy --model-dir work/r4_accuracy/models --zip submissions/Star_Coders_r4_submission.zip
```

Keep stage-2 results separately when comparing models. Do not report an r4 score
or submit an archive merely because the code or packaging command completed.
