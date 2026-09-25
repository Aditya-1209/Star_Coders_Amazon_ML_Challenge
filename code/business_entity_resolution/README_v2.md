# Star Coders: r5 entity resolution

This branch extends r2 with typo-tolerant candidate channels, bounded processing,
exact feature optimizations and cached stage-1 test inference. Its score is macro
F0.5 per S1 record, including singletons. Only organizer data are used.

**Fresh training is required.** Historical `models/v2` files do not measure r5.
Neither 97.5% leaderboard performance nor a full GPU runtime has been demonstrated.
See `docs/README_r5.md` at the repository root for measured results and the runner.

## Environment and desktop runner

Use Python 3.12 and the pinned `requirements_v2.txt` (named `requirements.txt` in
a packaged submission). XGBoost uses the requested CUDA device; use `--device cpu`
when CUDA is unavailable. Text processing uses CPU. The full-data target is 32 GB
RAM and an SSD; only bounded CPU development experiments have been run here.

From the repository root:

```text
python -m pip install -r code/business_entity_resolution/requirements_v2.txt
python scripts/run_r5.py --dataset PATH_TO_STUDENT_RESOURCE/dataset --work work/r5 --output output/r5 --device cuda --threads 12
```

The runner sets PYTHONPATH, trains, predicts and runs the official validator.
Add `--resume` to the same command to reuse completed stages when inputs and
code are unchanged. Use a fresh workspace after interrupted feature generation.

## Module-level reproduction (also works in the submission package)

Set PYTHONPATH to `code/business_entity_resolution/src` from the repository root,
or `src` when working inside the packaged `code/business_entity_resolution`.
For PowerShell, for example:

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
$env:POLARS_MAX_THREADS = "12"
```

Use `export PYTHONPATH=src PYTHONUTF8=1 POLARS_MAX_THREADS=12` on Linux/macOS.
Install `requirements.txt` in the package. Below, replace `DATASET` with the
organizer directory containing `train/` and `test/`, and use a fresh workspace.

```text
python -m er_v2.translit --dataset DATASET --out work/r5/models/translit.json
python -m er_v2.prepare --dataset DATASET --work work/r5 --splits train --workers 12 --translit work/r5/models/translit.json
python -m er_v2.run_block --work work/r5 --split train --top-k 64 --rescue-k 12 --chunk 10000
python -m er_v2.run_features --dataset DATASET --work work/r5 --split train --shard-pairs 1000000 --workers 12
python -m er_v2.train --dataset DATASET --work work/r5 --model-dir work/r5/models --device cuda --threads 12
python -m er_v2.prepare --dataset DATASET --work work/r5 --splits test --workers 12 --translit work/r5/models/translit.json
python -m er_v2.run_block --work work/r5 --split test --top-k 64 --rescue-k 12 --chunk 10000
python -m er_v2.run_features --dataset DATASET --work work/r5 --split test --shard-pairs 1000000 --workers 12 --stage1-model-dir work/r5/models --device cuda
python -m er_v2.predict --work work/r5 --model-dir work/r5/models --output output/r5 --device cuda --threads 12
```

Inspect `work/r5/models/metrics.json` for tuning fold 3, reporting fold 4,
per-country results and candidate-oracle ceilings. The learned transliteration
map excludes folds 3/4. Test processing includes every country, including France.
France has no training labels, so its accuracy cannot be measured locally.

Run the organizer validator after prediction, replacing `VALIDATOR` with the
provided `student_resource/utils/validate_submission.py` path:

```text
python VALIDATOR --matching output/r5/matching_results.tsv --candidate output/r5/candidate_pairs.tsv --test-dir DATASET/test --check-ids
```

Both files have one row per test S1 entity. Final matches are a subset of the
candidate file, which contains exactly the stage-2 inference pairs. Empty lists
remain empty, duplicates are removed, and each target has at most one S1 owner.
Only the matching TSV is uploaded for leaderboard scoring.

Before packaging, update the methodology document with actual full-run results.
Pass the freshly trained `work/r5/models` to `er_v2.package`; do not package
historical weights or the development engineering fixture as r5 results.
