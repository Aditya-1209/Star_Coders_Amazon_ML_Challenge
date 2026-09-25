# Reproducing the business entity resolution submission

All implementation code is in `src/er_baseline`. The saved model in `model/`
is the exact model used for the submission. Python 3.12 with SQLite FTS5 support
is required; the original run used Python 3.12.14 on macOS with an Apple M4 and
16 GB RAM. No GPU or external business data is required. Linux is also supported
by the multiprocessing and file-lock implementation; Windows is not supported.

## Environment

Run from this folder (`code/business_entity_resolution`):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
.venv/bin/python -c 'import sqlite3; c=sqlite3.connect(":memory:"); c.execute("CREATE VIRTUAL TABLE probe USING fts5(text)"); print("FTS5 available")'
.venv/bin/python -m unittest discover -s tests -v
```

Set the location of the organizer's extracted `dataset/` directory. Use a new
work directory; index building and training refuse existing destinations.

```bash
ER_DATASET=/absolute/path/to/student_resource/dataset
ER_WORK="$PWD/reproduction"
mkdir -p "$ER_WORK"
```

## Regenerate the submitted predictions using the exact saved model

```bash
.venv/bin/python -u -m er_baseline index \
  --sources "$ER_DATASET/test/test_source2.tsv" "$ER_DATASET/test/test_source3.tsv" \
  --output "$ER_WORK/test_search.sqlite"

.venv/bin/python -u -m er_baseline predict-parallel \
  --anchors "$ER_DATASET/test/test_source1.tsv" \
  --index "$ER_WORK/test_search.sqlite" --model model \
  --output "$ER_WORK/output" --workers 8 --chunk-size 128

.venv/bin/python -u -m er_baseline validate \
  --anchors "$ER_DATASET/test/test_source1.tsv" \
  --targets "$ER_DATASET/test/test_source2.tsv" "$ER_DATASET/test/test_source3.tsv" \
  --output "$ER_WORK/output" --receipt "$ER_WORK/output/validation.json"
```

The two required files are `matching_results.tsv` and `candidate_pairs.tsv` in
`$ER_WORK/output`. Every target in the candidate file is scored by the model;
matches are selected at threshold 0.605. IDs are sorted within lists and each
Source 1 row follows input order, including businesses with no matches.

For an interrupted prediction, repeat the same `predict-parallel` command with
`--resume`. Keep the chunk size at 128; worker count may be reduced on smaller
machines. Input, model, index, or relevant code changes invalidate resumption.
Do not run two prediction jobs against the same output directory. Check
`inference_progress.json` for progress and `inference_complete.json` for completed
inference. Only a successful `validation.json` establishes full validation.

On macOS, prefix a long command with `caffeinate -i` to prevent idle sleep.
This does not override lid-close sleep on battery. Saved chunks survive an
interruption. Runtime depends on hardware, cache, data distribution, and sleep;
the initial 768-row eight-worker sample processed about 93 anchors per second.

## Reproduce supervised preparation, training, and evaluation

The classifier uses a seeded 6,000-anchor sample, split into 4,198 training,
898 tuning, and 904 holdout anchors. Retrieval searches all training targets.
The small target sample produced by `prepare` is a development artifact;
the `train` command below deliberately uses the full training index.

```bash
.venv/bin/python -u -m er_baseline prepare \
  --dataset "$ER_DATASET" --output "$ER_WORK/dev" \
  --anchors 6000 --background 200000 --seed 42

.venv/bin/python -u -m er_baseline index \
  --sources "$ER_DATASET/train/train_source2.tsv" "$ER_DATASET/train/train_source3.tsv" \
  --output "$ER_WORK/train_search.sqlite"

.venv/bin/python -u -m er_baseline train \
  --dev "$ER_WORK/dev" --index "$ER_WORK/train_search.sqlite" \
  --output "$ER_WORK/model" --threads 6 --iterations 450 \
  --per-channel 20 --posting-budget 3000 --seed 42

.venv/bin/python -m er_baseline report \
  --run "$ER_WORK/model" --output "$ER_WORK/development_report.md"
```

Use `--model "$ER_WORK/model"` in prediction to evaluate a retrained model.
Floating-point/platform differences can affect retraining; use the included
weights when exact submission reproduction is required. Development metrics
are in `model/metrics.json`. Holdout macro F0.5 is 0.925549; test accuracy is
unknown, including accuracy on the previously unseen country France.

## Source map

- `data.py`, `prepare.py`: strict TSV parsing, sampling, and grouped partitions.
- `retrieval.py`, `text.py`: country-aware FTS5 candidates and 31 pair features.
- `model.py`, `metrics.py`: CatBoost training, threshold tuning, and macro F0.5.
- `parallel.py`: bounded multiprocessing, atomic checkpoints, ordered outputs.
- `validate.py`: streaming format, completeness, target existence, and subset checks.
- `submission.py`: hashes and packages completed, validated outputs.
- `__main__.py`: command-line entry point.

No hard-coded two-country filter is used. Unicode is preserved, but there is no
cross-script transliteration or neural semantic retrieval. Country blocking can
miss cross-country matches. A detailed methodology and error discussion is in
the archive-root `Documentation_template.md`.
