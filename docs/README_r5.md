# r5-recall-speed

Based on `r2-translit` at `427aa32`. The reported **94.3% public leaderboard**
result is the team's r2 result, not a result reproduced here. **97.5% has not
been demonstrated.** This branch is ready for a full desktop experiment.

## What changed

- Retain every original top-64 candidate and add up to 12 candidates from capped
  name prefixes, suffixes and consonant skeletons. This targets spelling and
  transliteration damage. Country remains an open string label, including France.
- Train the matcher with an explicit indicator for candidates from the new
  channels. A fresh training run is required; do not use the historical shipped
  weights as evidence of r5 quality.
- Compare the ordinary threshold with the existing expected-F0.5 rule on fold 3.
  Persist the selected policy for inference. Fold 4 remains the reporting holdout.
  The development experiment selected the ordinary threshold.
- Sort tokens once per record and reuse Python string conversions across fuzzy
  scorers. The 43 original feature columns were verified exactly equal to r2.
- Partition blocking by country and use 10,000-query joins. IDF still uses the
  global target count, so country partitioning preserves the base scores.
- Default to one-million-pair feature shards in the runner. After training,
  compute test features, immediately score stage 1, and store full features only
  for survivors. Retain **all** stage-1 scores for identical global context.
- Avoid duplicate out-of-fold model predictions and discard stage-1 rejects
  before the stage-2 feature join. Use 250,000-row dense prediction batches.
- Carry over r4's deterministic tie handling, empty-frame support, small-fold
  fixes, duplicate-safe metric, atomic TSV writes and feature manifests. r3/r4's
  two-hop stage is not enabled in this branch; its extra runtime is unmeasured.
- Add a runner with completed-stage checkpoints, input/code fingerprints,
  per-stage timing logs and the official validator with ID checks.

No external business data, lookup APIs, geocoders or pretrained weights are used.
The PDF and student README were read; scoring is **macro F0.5 with singletons**,
and the candidate output is exactly the set scored by the final stage-2 matcher.

## Measured results and their limits

Development data were streamed from the supplied organizer files using seed
20260925: 6,000 uniformly reservoir-sampled S1 records, all 20,701 of their
labelled matches, plus randomly sampled distractors. Total S2/S3 pool: 41,448.
The smaller distractor pool makes this **much easier than the full corpus**.
It is not evidence for a 99% leaderboard score or statistical significance.

| Measurement | r2 feature reference | r5 |
| --- | ---: | ---: |
| Holdout macro F0.5, 601 S1 records | 0.990832 | 0.991830 |
| Tuning macro F0.5, 624 S1 records | 0.993409 | 0.992868 |
| Holdout pair precision | 0.995538 | 0.996020 |
| Holdout pair recall | 0.989163 | 0.986207 |
| Candidate pairs across 6,000 S1 | 376,239 | 400,563 |
| Raw blocking recall across 6,000 S1 | 0.996860 | 0.997198 |

The reference shares the bounded runtime/correctness fixes and sampled
normalization with r5, but uses the original r2 feature set, original candidates
and recomputed original blocking context. Both used CPU, two threads and up to
400 boosting rounds. This is a controlled feature comparison, not a full
reproduction of the team's leaderboard model. The transliteration map excludes
folds 3/4; models use the same fold assignment as r2.

Feature-kernel benchmark: 100,000 actual candidate pairs, two workers, warm-up
and three alternating timed runs. Median r2 **3.135s**, r5 **2.612s**, or **1.20x**
the throughput. All original feature values are exactly equal. This excludes
per-record setup and is not an end-to-end speed estimate.

Cached versus ordinary inference on the development engineering fixture:

- Full feature bytes: 15,985,500 -> 1,316,356; score cache adds 2,290,604 bytes.
- Total stored feature/cache data falls by **77.4%**.
- All probabilities and both TSV files are exactly equal; 27,993 final candidates.
- Total feature-generation-plus-inference time was **13.52s ordinary vs 14.30s
  cached**. It was slightly slower on this small corpus. Full-scale disk/memory
  savings must be timed on the i9/RTX machine.
- Seven unit regressions passed. The complete CLI workflow passed the official
  validator, including ID existence checks, on a 6,000-row engineering fixture.
  Training records were reused as unlabelled inputs solely for that I/O check;
  its predictions are **not a test submission or a generalization score**.

This machine had 16 GB RAM and eight logical CPUs, unlike the target desktop.
No full-corpus runtime, GPU benchmark, France accuracy, or new leaderboard score
is claimed. Raw results: `reports/r5_development_results.json` and
`reports/r5_feature_benchmark.json`.

## Run on the i9-13900K / 32 GB / RTX 3060 desktop

From a clone of the repository, in PowerShell:

```powershell
git fetch origin
git switch r5-recall-speed
py -3.12 -m venv .venv-r5
.\.venv-r5\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements_v2.txt
.\.venv-r5\Scripts\python.exe scripts/run_r5.py --dataset "C:\path\student_resource\dataset" --work work/r5 --output output/r5 --device cuda --threads 12
```

Replace the dataset path with the extracted organizer folder containing `train/`
and `test/`. Its sibling `utils/validate_submission.py` is required for validation.
The runner sets PYTHONPATH and thread limits itself and stops on a failed stage.
The GPU handles XGBoost; text processing and blocking use CPU. Use `--device cpu`
on a machine without a working CUDA installation. Full-data CUDA execution has
not been tested in this task.

In the workspace used to develop this branch the dataset argument is:

```text
../6ab10eb3b23ba_student_resource/student_resource/dataset
```

Outputs and logs:

- `work/r5/models/metrics.json`: selected policy, fold 3/4 scores, per-country
  results, raw-blocking oracle and post-stage-1-pruning oracle.
- `work/r5/checkpoints/*.log` and `*.json`: per-stage logs, times and fingerprints.
- `output/r5/matching_results.tsv`: upload this to the challenge portal only after
  the full run and official validation finish.
- `output/r5/candidate_pairs.tsv`: exact final candidate set for the package.

Keep the workspace on an SSD. Global normalized records, candidate context and
training matrices still consume RAM; this is not a fully disk-backed trainer.
If memory is tight, use `--shard-pairs 500000 --block-chunk 2000` in a fresh run.

Add `--resume` to the **same command** to reuse completed stages. It refuses
changed commands, code, inputs or recorded outputs. Interrupted feature shards
are intentionally rejected: use a fresh work directory after such a failure.
`--phase train` stops after validation metrics; use `--phase predict --resume`
with the same work directory to generate test outputs from those models.

For a blocking ablation, run in a different work/output directory with
`--rescue-k 0`. Do not change retrieval settings between training and prediction.
Check the **raw-blocking oracle first**: if its macro F0.5 is below 0.975, matching
threshold changes alone cannot reach the target on that validation population.
Compare both country scores and the singleton-sensitive macro score; do not
choose thresholds using fold 4 or repeatedly tune to a public leaderboard score.

## Reproduce the development checks

After setting `PYTHONPATH=code/business_entity_resolution/src` (PowerShell:
`$env:PYTHONPATH = "code/business_entity_resolution/src"`):

```text
python -m unittest discover -s tests_v2 -v
python scripts/sample_r5.py --dataset PATH_TO_DATASET --out work/sample/dataset
python scripts/run_r5.py --dataset work/sample/dataset --work work/sample/run --phase train --device cpu --threads 2 --rounds 400 --shard-pairs 100000 --block-chunk 500
python scripts/compare_r5.py --work work/sample/run --reference work/sample/reference --dataset work/sample/dataset
python -m er_v2.train --dataset work/sample/dataset --work work/sample/reference --model-dir work/sample/reference/models --device cpu --threads 2 --rounds 400
python scripts/benchmark_r5_features.py --work work/sample/run --out work/sample/feature_benchmark.json
python scripts/verify_r5_cache.py --work work/sample/run --out work/sample/cache-check
```

The original r2 commit must be present for the kernel benchmark. Each sampling,
reference and cache-check output directory must be fresh.

Before final competition packaging, update `Documentation_template.md` with the
actual full-run methodology and metrics. The historical root document and shipped
weights describe earlier work and must not be presented as r5 results.
