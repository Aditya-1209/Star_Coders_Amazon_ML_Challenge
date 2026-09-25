# r7 accuracy experiment

This builds on `r7-improvements` at `4c2c603`. The team's Amazon score is about
95; the goal is 97–98. **No new Amazon score is claimed.** The local metric is
macro F0.5 per Source 1 business, not ordinary classification accuracy. France
has no labeled training businesses, so US/India holdout alone cannot establish
leaderboard performance. Historical scores and hypotheses are in
[SUGGESTED_IMPROVEMENTS.md](SUGGESTED_IMPROVEMENTS.md).

## Accuracy changes

* An optional eight-candidate consonant-class retrieval channel supplements
  every existing combined/name/address/typo candidate. It targets transliteration
  variants using token, adjacent-token-pair and full-name codes. Keys remain
  country-scoped, capped at 100 targets for token/full-name codes and 200 for
  adjacent-token pairs.
* Thirteen optional pair-local features measure directional mean-best token
  Jaro-Winkler alignment, unmatched tokens, phonetic overlap, missing-address
  name evidence and relative name lengths. Alignment considers up to 12 tokens
  per name. They do not use corpus counts, split sizes or labels.
* Stage 1 keeps the original feature set and complementary training folds.
  Stage 2 compares original versus enhanced features using the same candidates
  and stage-1 ensemble. Stage 3 makes the same feature comparison on its graph.
  Each comparison, cutoff and stage-2/3 blend is selected on fold 3. Ties keep
  the original feature set. Fold 4 is evaluated only after selection.
* Country-specific cutoffs are learned only for labeled tuning countries with
  at least 1,000 businesses and a tuning gain greater than 0.0001. France and
  excluded countries retain the global cutoff. Global-cutoff reference metrics
  remain in the results so the effect can be checked.
* The runner first trains small train-versus-test classifiers using original
  and enhanced features. Validation is grouped by Source 1. It permits enhanced
  features only when AUC is at most `max(0.75, original AUC + 0.02)` for both
  US and India. This is a heuristic diagnostic, not proof of generalization.

The ablation baseline uses the expanded candidate set; it is **not** a rerun of
the original Amazon submission. `retrieval_audit.json` separately measures the
added phonetic candidates against original channels on fold 3. The pre-existing
absolute-frequency/IDF features remain excluded. The experimental ghost mode
is not enabled by this runner, and stage 3 explicitly rejects ghost models.

The first token-only probe recovered just one additional true pair on fold 3.
The revised multi-word probe recovered **1,678** (498 India, 1,180 US), at the
cost of 423,279 additional candidates across 220,699 tuning businesses. This is
retrieval evidence, not a trained-model score. The chosen revision is tested on
the same full target corpus used by the original channels.

## Run on this Mac

Python 3.12 and the extracted organizer files are required. The existing
`.venv` can be used:

```bash
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements_v2.txt
caffeinate -i .venv/bin/python -u scripts/run_r7_overnight.py
```

Keep the charger connected and the lid open. `caffeinate -i` prevents idle sleep;
it does not keep processing through lid-close sleep or shutdown. CPU training is
the default. No model downloads or external business data are used.

On the i9-13900K / 32 GB / RTX 3060 desktop, create a Python 3.12 venv, install
the same requirements, and run:

```powershell
python -u scripts/run_r7_overnight.py --device cuda --threads 12
```

The runner performs checks, transliteration, normalization, blocking, features,
retrieval audit, shift diagnostic, training, graph training, test prediction,
official validation with ID checks, and a final report. It runs one heavy process
at a time. **No fixed overnight completion time or RAM ceiling is guaranteed.**

## Memory and execution

Normalization uses 150k-row windows. Target keys are built in disk-backed
50k-record batches; global key counts and IDF stay consistent across batches.
Countries are loaded separately with global Source 2/3 indices preserved.
Blocking joins use 2k queries, feature shards about 150k pairs, and prediction
and quantization batches 100k rows. Stage-2 joins only materialize surviving
pairs. Graph support uses 1k-anchor batches. Six threads are the Mac default.

Source frames, candidate/context tables and quantized training matrices still
occupy RAM. There is no claim of fully out-of-core training. The iterator follows
the [XGBoost DataIter API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DataIter);
disk-backed key construction uses
[Polars streaming parquet output](https://docs.pola.rs/api/python/stable/reference/api/polars.LazyFrame.sink_parquet.html).

## Progress and resume

`work/r7_overnight/overnight.json` records the current stage, process IDs, elapsed
time, completed stages, exact commands and errors. Each stage has a log in
`work/r7_overnight/logs/`. `--plan` prints commands without starting work.

```bash
.venv/bin/python scripts/run_r7_overnight.py --plan
caffeinate -i .venv/bin/python -u scripts/run_r7_overnight.py --resume
```

Resume verifies the code, dataset metadata, settings and completed output
metadata. It resumes **between stages**, not midway through training. A failed
feature stage regenerates its own output directory. Do not run another pipeline
against this work folder while it is active.

`--start STEP` explicitly reuses prerequisite artifacts. Use it only after
checking their compatibility; code or data changes can require regeneration.
For this initial Mac run, transliteration, preparation and train blocking were
started interactively before the detached runner takes over at `retrieval_audit`.

## Results

After successful completion:

```text
work/r7_overnight/result.md                 readable local result
work/r7_overnight/result.json               metrics, ablations, diagnostics, hashes
work/r7_overnight/models/                   newly trained models and metrics
output/r7_overnight/matching_results.tsv    Amazon upload candidate
output/r7_overnight/candidate_pairs.tsv     exact candidates scored by final model
output/r7_overnight/stage2/                 stage-2 comparison output
```

Only `overnight.json` with `status: complete` confirms the full workflow finished.
The runner does not upload to Amazon, push files, or infer an Amazon score from
the local result. Raw data, features and generated submissions stay Git-ignored.

## Verification before the full run

The 32 tests in `tests_v2` pass on CPU, including a small real XGBoost fit and an
end-to-end synthetic run through both training stages, graph expansion, test
inference and the official validator. The test covers the new feature ablations,
country partition indices, blank names, weighted quantization and chunked index
equivalence. Those synthetic tests do not measure challenge accuracy.
