# Star Coders - Amazon ML Challenge

> **On `r7-improvements`, start with [the r7 experiment guide](docs/README_r7_overnight.md).**
> It runs a CPU or CUDA experiment with phonetic retrieval, token-alignment
> features, a train/test shift diagnostic, and validation-selected feature
> ablations. Local holdout results are not Amazon leaderboard scores.
> The remainder of this README describes the historical CatBoost baseline.

Match noisy business records from Source 2 and Source 3 to every Source 1 business.
The pipeline produces both required TSVs and a verified submission ZIP, using only
the challenge dataset. **You can run the included trained model immediately;
retraining is optional.** No GPU is needed for this version.

Team: Adyanth Mallur, Aditya Patil, Akshay Gudur, and Advika Raj.

## What is ready

- A frozen CatBoost model and its configuration in `models/baseline_full_corpus/`.
- A complete CPU workflow: ZIP extraction, search index, parallel predictions,
  checkpoints, strict full-file validation, and submission packaging.
- A filled methodology write-up in `Documentation_template.md`; final test counts
  are inserted into the packaged copy automatically after validation succeeds.
- Tests for retrieval, scoring, data splits, interrupted-run recovery, and outputs.

The model scored **0.9255 macro F0.5 on 904 held-out training businesses** while
searching all 10.32 million training targets. This is not a leaderboard score.
Full test inference on the Mac was intentionally stopped to move to a desktop;
there is no completed full-test submission in this repository. Raw data,
indexes, local checkpoints, and generated outputs are not committed.

## 1. Prepare the computer

Recommended for the available desktop: **i9-13900K, 32 GB RAM, SSD, 12 workers**.
This is a starting configuration, not a measured speedup guarantee. Have at least
**20 GB of free SSD space** for data, indexes, checkpoints, outputs, and the ZIP.
The Mac ran with 16 GB RAM and 8 workers; reduce workers if the computer starts
swapping. GPU VRAM does not replace system RAM for the SQLite search index.

Supported environment: **Python 3.12 on Linux, WSL2 Ubuntu, or macOS**. Native
Windows Python is not supported because the runner uses POSIX file locks.

### Windows desktop: use WSL2

In an Administrator PowerShell terminal, install Ubuntu if needed:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if prompted, open **Ubuntu**, and create its Linux user. All remaining
commands run in the Ubuntu terminal, not PowerShell. Ubuntu 24.04 supplies
Python 3.12:

```bash
sudo apt update
sudo apt install -y git gh python3-venv python3-pip
```

Keep the clone, extracted dataset, and SQLite index inside the Linux filesystem
(for example `~/projects`), rather than working directly under `/mnt/c`.
If WSL reports substantially less than 16 GB usable memory, configure its memory
limit before starting; a 32 GB Windows host should leave several GB for Windows.
Keep the Windows host awake while the job runs. WSL cannot continue through host sleep.
Microsoft documents [WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install)
and [where to store files for performance](https://learn.microsoft.com/en-us/windows/wsl/filesystems).

### Linux or Mac

Use Python 3.12 with SQLite FTS5 support. On Ubuntu 24.04 use the apt commands
above. On macOS, install Python 3.12 and GitHub CLI first
(for example `brew install python@3.12 gh`).

## 2. Clone and set up

This is a **private repository**. Each teammate needs repository access and an
authenticated GitHub account. Do not put a GitHub token into a command or a file.
For first-time GitHub CLI setup, authenticate in the browser:

```bash
gh auth login --web --git-protocol https
gh auth setup-git
```

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git
cd Star_Coders_Amazon_ML_Challenge
bash scripts/setup.sh
```

Setup creates `.venv`, installs pinned dependencies, checks SQLite FTS5, and runs
the tests. The tests use tiny synthetic records; they do not need the challenge
ZIP and do not launch the full run. Commands below use `.venv/bin/python`, so
shell activation is optional.

## 3. Supply the dataset and run

Copy the organizer's `amazon_ml_dataset.zip` into the repository root. It is
ignored by Git. On WSL, you can copy it from Windows Downloads:

```bash
cp "/mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/amazon_ml_dataset.zip" .
```

Run the entire workflow:

```bash
.venv/bin/python -u scripts/run_full_inference.py \
  --archive amazon_ml_dataset.zip --workers 12
```

This command:

1. Extracts the seven known dataset TSVs into `student_resource/dataset/`.
2. Builds `artifacts/test_search.sqlite` from all test Source 2/3 records.
3. Uses the included model to predict every test Source 1 business, including France.
4. Saves small checkpoints and merges the two completed TSVs in input order.
5. Checks every output row and every candidate ID against the raw test data.
6. Builds and verifies `submissions/Star_Coders_submission.zip`.

If you already extracted the dataset, skip ZIP extraction:

```bash
.venv/bin/python -u scripts/run_full_inference.py \
  --dataset /absolute/path/to/student_resource/dataset --workers 12
```

For macOS, prefix the run command with `caffeinate -i` to prevent idle sleep.
**Caffeinate does not prevent lid-close sleep on battery.** On any computer,
avoid shutdown or system sleep while processing; turning off the screen is fine.

## Progress, stopping, and resuming

In another terminal at the repository root:

```bash
.venv/bin/python scripts/status.py
```

The command prints the pipeline stage, saved rows, throughput, and estimated
remaining active processing time. The ETA excludes future sleep or shutdown.
`artifacts/full_inference_status.json` records the overall stage;
`output/inference_progress.json` records prediction progress.

Press **Ctrl+C once** to stop. Allow the small set of queued chunks to finish.
Completed checkpoints remain on disk. On the **same machine and paths**, resume:

```bash
.venv/bin/python -u scripts/run_full_inference.py --workers 12 --resume
```

If you used a custom `--dataset`, `--work`, `--output`, or `--submission`, supply
the same options again. Keep the chunk size at its default of 128. Worker count
can change. Never run two jobs against the same work/output directory.

The manifest verifies inputs, model, code, and checkpoint hashes. Changing
code/model/data invalidates resume. Checkpoints contain absolute-path
fingerprints: **the desktop starts a fresh run; copying the Mac checkpoints is
not a supported resume method**. The saved model is portable and included in Git.

## Finished files and submission

Only overall stage **`complete`** means inference, validation, and packaging all
succeeded. A partly populated output directory is not a finished submission.

```text
output/
  matching_results.tsv       # upload this file to the leaderboard portal
  candidate_pairs.tsv        # exact candidate set passed to the model
  validation.json            # full checks and output hashes
submissions/
  Star_Coders_submission.zip
  Star_Coders_submission.manifest.json
```

The ZIP contains both TSVs, standalone source code, pinned dependencies, model
weights, tests, exact reproduction instructions, and the completed methodology.
It excludes raw data, the virtual environment, search indexes, and chunk files.
Nothing is uploaded to the challenge portal automatically.

## How the model works

Country-aware SQLite FTS5 indexes retrieve up to 20 candidates from each of three
channels: business name, address, and name character trigrams. Rare-term gates
keep common-text queries tractable. The deduplicated union, at most 60 records,
is scored with 31 fuzzy-text, address/number, missingness, and retrieval features.
A depth-6 CatBoost model accepts scores at or above **0.605**.

Training uses 4,198 sampled anchors, threshold tuning uses 898, and holdout uses
904. Related anchors sharing positive targets stay in one split. Retrieval for
this model searches all training targets, rather than the easier sampled target
pool used in the first experiment.

| Full-corpus holdout measure | Result |
| --- | ---: |
| Macro F0.5 | 0.9255 |
| Pair precision | 97.17% |
| Pair recall | 84.59% |
| Candidate recall | 93.82% |
| Singleton accuracy | 92.45% |

Training labels contain US and India only. France is processed but has no labeled
accuracy estimate. Alternate scripts, missing addresses, and ambiguous names
remain limitations. See the methodology and `models/baseline_full_corpus/metrics.json`.

## CPU versus GPU

This version spends most of its time in SQLite search and CPU feature generation.
The measured CatBoost fit took about 4.6 seconds, while full test search takes
hours. Adding a 3060 or 5090 does not move those searches onto the GPU.
Benchmark the 13900K with 8, 12, and 16 workers before assuming more workers help.
An NVIDIA GPU becomes useful for a future neural-embedding retrieval pipeline,
which is not implemented here.

## Retraining and lower-level commands

For optional retraining, metrics, and individual CLI stages, see
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).
Use `PYTHONPATH=code/business_entity_resolution/src` when invoking
`python -m er_baseline` from the root. The included frozen model is the simplest
way to reproduce the current baseline without retraining.

If a run fails, read the error in `artifacts/full_inference_status.json` and the
terminal output. Preserve `output/parts` and its manifest. Fix the environment
or restore unchanged input files, then resume. An interrupted index build is
rebuilt on the next run; an existing completed index is checked before reuse.
