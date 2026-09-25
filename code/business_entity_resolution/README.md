# CPU business entity resolution baseline

For a fresh clone, start with the project-root `README.md` and its one-command
runner. This document describes lower-level experiments and retraining.

This pipeline matches noisy business records using a disk-backed SQLite search
index, 31 pairwise similarity features, and a CPU CatBoost classifier. It uses
only the supplied records and labels. It does not call business lookup,
geocoding, embedding, or external entity-resolution services.

The pipeline has two completed **development benchmarks** and a resumable full
test inference runner. Check `artifacts/full_inference_status.json` at the project
root for the actual full-run status. The first benchmark selects 6,000 Source 1 businesses, retains all of their
labeled matches, and adds 200,000 sampled distractors. The latest searches the
entire 10,320,219-record training target pool using the same supervised split.
Its holdout macro F0.5 is 0.9255 (904 held-out businesses), versus 0.9741 on the
easier reduced pool. The classifier itself is fitted on 4,198 sampled anchors;
this is not training on all 2.2 million Source 1 labels. Neither score measures
the official test set or unseen France.

## Setup and reproducibility

Use Python 3.12 with SQLite FTS5 support. From the project root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
export PYTHONPATH="$PWD/code/business_entity_resolution/src"
```

The existing development run used the project-root `.venv`, Python 3.12.14,
and an Apple M4 with 16 GB RAM. Dependencies are pinned in `requirements.txt`.
CatBoost is Apache-2.0 licensed; the model is a small boosted-tree classifier,
not a pretrained language model. See the upstream
[CatBoost license](https://github.com/catboost/catboost/blob/master/LICENSE).

Run these commands from the project root to reproduce in a fresh output location
(commands refuse to overwrite existing datasets, indexes, or model runs):

```bash
.venv/bin/python -m er_baseline prepare \
  --dataset student_resource/dataset \
  --output artifacts/dev \
  --anchors 6000 --background 200000 --seed 42

.venv/bin/python -m er_baseline index \
  --sources artifacts/dev/targets.tsv \
  --output artifacts/dev/search.sqlite

.venv/bin/python -m er_baseline train \
  --dev artifacts/dev \
  --index artifacts/dev/search.sqlite \
  --output artifacts/baseline_v1 \
  --threads 6 --iterations 450 --per-channel 20 --seed 42

.venv/bin/python -m er_baseline report \
  --run artifacts/baseline_v1 \
  --output reports/baseline_v1.md

.venv/bin/python -m unittest discover -s code/business_entity_resolution/tests -v
```

The initial package extraction and full-dataset inspection live outside this
pipeline; the training CLI operates directly on the organizer's extracted TSVs.
Supply a different `--dataset` path when running elsewhere.

## How it works

1. **Sampling and partitions.** Reservoir sampling selects Source 1 businesses
   and unrelated target records. Source 1 anchors are partitioned approximately
   70%/15%/15% into training, threshold tuning, and held-out evaluation, stratified
   by country and singleton status. Anchors sharing a labeled target are grouped
   together. No supervised positive group crosses partitions.
2. **Search index.** Each observed country gets its own SQLite FTS5 index. Labels
   are not stored in the search index. The shared development index contains
   unlabeled target text from every partition, as the candidate pool would at
   inference. Names, addresses, and name character trigrams form three retrieval
   channels. Each retrieves up to 20 records; their deduplicated union is scored.
   Rare observed terms limit the cost of common-name and address searches.
   The full-corpus run uses `--posting-budget 3000`: rarer terms restrict which
   documents can match while all selected query terms still contribute to BM25
   ranking. When every term is common, the two rarest are intersected. This
   setting was chosen using retrieval timings/recall on training anchors only,
   and is saved in the model configuration for inference.
3. **Pair features.** Unicode-preserving normalization, Latin accent folding,
   auxiliary legal-suffix removal, name/address fuzzy similarities, token
   overlaps, initials, number agreements/conflicts, missingness, and retrieval
   ranks provide 31 numeric features. IDs and country identities are not model
   features. Country labels are discovered dynamically, including France.
4. **Classification.** CatBoost trains on retrieved candidate pairs. Negative
   examples are actual retrieved nonmatches; missed positives are not inserted
   into validation candidate lists. Early stopping uses tuning-set Logloss.
5. **Threshold tuning.** A probability threshold maximizes tuning-set macro
   F0.5 over Source 1 businesses, including singletons and positives missed by
   candidate retrieval. The frozen threshold is then evaluated on holdout.

For a true set `T` and predicted set `P`, the per-business score is
`1.25 * |T ∩ P| / (0.25 * |T| + |P|)`. When both sets are empty, the score is 1.
Macro F0.5 is the mean of those scores, rather than F0.5 of pooled pairs.

## Saved artifacts

The run directory contains:

- `model.cbm` and `config.json`: fitted classifier, selected threshold, and
  feature/retrieval configuration.
- `metrics.json`: split-level and country-level results, candidate recall,
  candidate oracle score, timings, and memory measurement.
- `holdout_matching_results.tsv` and `holdout_candidate_pairs.tsv`: development
  outputs in the official two-column formats. **These are not test submissions.**
- `holdout_errors.json`: every holdout false positive and false negative, with
  an indication of whether retrieval found the pair.
- `pair_features.npz` and `probabilities.npy`: reproducible diagnostic data.

Model probability outputs are decision scores; no separate probability
calibration is fitted. The threshold must be reassessed when the candidate
corpus or retrieval settings change substantially.

## Full-data inference interface

The completed full-corpus benchmark reran sampled Source 1 training against **all**
training Source 2/3 records. The supervised anchor split stays unchanged; the
retrieval corpus and hard negative examples become much larger. This does not
train on all 2.2 million Source 1 labels.

```bash
.venv/bin/python -m er_baseline index \
  --sources student_resource/dataset/train/train_source2.tsv \
            student_resource/dataset/train/train_source3.tsv \
  --output artifacts/train_search.sqlite

.venv/bin/python -m er_baseline train \
  --dev artifacts/dev \
  --index artifacts/train_search.sqlite \
  --output artifacts/baseline_full_corpus \
  --threads 6 --iterations 450 --per-channel 20 --posting-budget 3000 --seed 42

.venv/bin/python -m er_baseline report \
  --run artifacts/baseline_full_corpus \
  --output reports/baseline_full_corpus.md
```

The full test index is built: 9,969,589 target records, including France.
Parallel inference exactly reproduced the 904-row serial holdout outputs.
On a seeded 768-row test sample, 4/6/8 workers produced identical files at
57.2/84.5/93.1 anchors per second respectively. These are short sequential
benchmarks with different cache warmth, not sustained full-run measurements.
See `reports/parallel_inference_benchmark.md`.

```bash
.venv/bin/python -m er_baseline index \
  --sources student_resource/dataset/test/test_source2.tsv \
            student_resource/dataset/test/test_source3.tsv \
  --output artifacts/test_search.sqlite

# Fresh full inference, followed automatically by strict full-file validation:
.venv/bin/python -u scripts/run_full_inference.py --workers 8 --chunk-size 128

# Only after an interruption (do not launch alongside the existing run):
.venv/bin/python -u scripts/run_full_inference.py \
  --workers 8 --chunk-size 128 --resume
```

The original Mac run used `caffeinate -i` and was deliberately stopped to move
processing to a desktop; its checkpoints remain local. Caffeinate prevents idle
sleep, but closing the lid or manually sleeping can still suspend processing.
The historical log is `artifacts/logs/full_test_inference.log`. Progress and ETA are
in `output/inference_progress.json`. The overall stage is recorded in
`artifacts/full_inference_status.json`: only `complete` means prediction,
validation, and packaging succeeded. `output/validation.json` contains the validation
receipt and SHA-256 hashes of the final files.

Workers save atomic per-chunk output files in `output/parts`. Resume verifies
their hashes and skips completed chunks. Input/model/code/chunk-size changes are
rejected; worker count may change between runs. A file lock prevents concurrent
writers. The final TSVs are merged only after every Source 1 record is processed.
Keep checkpoints until the full run and validation have completed successfully.

`candidate_pairs.tsv` is exactly the union of candidates passed to the model;
`matching_results.tsv` is its thresholded subset. Empty rows are retained, IDs
are deduplicated and sorted, and candidates are read only from the provided
index. The automatic validator checks every output row, list uniqueness, valid
target prefixes and existence against the raw test Source 2/3 files, and strict
match/candidate subset membership. It retains source ID sets, not millions of
candidate mappings. It also requires outputs to follow input order.

The official validator was passed on the full-corpus 904-row holdout benchmark.
Its all-at-once candidate mappings can consume excessive RAM on full test files;
the production pipeline uses the stricter streaming validator above. To rerun
that validation independently:

```bash
.venv/bin/python -m er_baseline validate \
  --anchors student_resource/dataset/test/test_source1.tsv \
  --targets student_resource/dataset/test/test_source2.tsv \
            student_resource/dataset/test/test_source3.tsv \
  --output output --receipt output/validation.json
```

## Known limitations and next experiments

- The initial reduced-pool run has much lower distractor density than the full
  corpus. The latest benchmark addresses this: 93.82% candidate recall, 97.17%
  pair precision, 84.59% pair recall, and a retuned threshold of 0.605. A short
  parallel test benchmark projects roughly 5.2 hours for all test anchors;
  sustained speed and final duration can differ.
- Country blocking assumes true matches keep the same country label. Unknown
  countries work when they exist in the indexed target pool, but country typos
  and cross-country true matches have no fallback channel.
- Cross-script names can be missed if addresses have no useful shared text.
  Unicode is retained, but this baseline does not perform transliteration or
  multilingual neural embedding retrieval.
- France is absent from the labeled data. Passing the French retrieval smoke
  test establishes functionality, not matching accuracy on French businesses.
- The methodology and automatic packaging workflow are implemented. Full test
  predictions must finish and pass validation before the ZIP is produced.

Algorithm references: [SQLite FTS5](https://www.sqlite.org/fts5.html) and
[CatBoost training](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit).
