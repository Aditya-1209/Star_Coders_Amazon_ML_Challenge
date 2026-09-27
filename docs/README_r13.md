# R13: improve difficult matches after retrieval

R13 starts from R12 commit `06656f9`. Its website objective is **0.985 macro
F0.5**. No full R13 GPU training run or website score has been measured yet.
The R11 logs report 0.98723 locally and the user reported 0.977 online, so a
local score over 0.985 does **not** establish that the website target is met.

## Changes from R12

- **Harder cross-encoder training.** For each sampled business, keep every
  positive and select at most five negatives: two highest-scoring stage-2
  mistakes, then lexical, neural-neighbour and random examples. Remove
  duplicates between channels and refill sparse channels from lexical
  candidates. Stage 2 fits on folds 2/5; mining only uses CE folds 0/1/8/9.
  Fold 3/4 labels never enter this training path. Unlike increasing sample
  caps, this directly targets the classifier's existing confusion.
- **Neural evidence from sibling records.** Select at most three stage-2
  matches per business with probability >=0.98 and an ownership margin >=0.10
  over another business claiming that target. Compare each candidate with
  these other targets using existing multilingual embeddings. Supply cosine
  maximum/mean/minimum, confidence-weighted cosine, support count, strongest
  confidence and improvement over source-to-target cosine to the final model.
  A candidate cannot support itself. No anchors means explicit zero evidence.
  These are learned features, not automatic match propagation.
- **Remove incomplete target context from the final model.** R12 graph/CE
  scores cover folds 3/4/6/7 during training, versus every business at test.
  R13 excludes `ce_rank_t`, `ce_gap_t`, `ncos_rank_t`, `ncos_gap_t` from the
  final model. Pair scores, within-business ranks and R12's full-pool lexical
  record competition remain. This fixes the final feature set; the unchanged
  R12 graph reference still has its original features.
- **Useful diagnostics.** `metrics.json` separates missing-address pair
  errors, retrieved-but-rejected matches and retrieval misses. Website target,
  local score and measured website score are separate fields. The last remains
  `null` until a real submission is scored.

The same gate chooses the final model: fit on 6/7, choose model/blend and
thresholds on 3A, apply one improvement gate on 3B, then report fold 4 without
reselection. A failed gate uses the rebuilt R12 graph reference. Earlier layers
already use fold 3; this is not a completely untouched validation experiment.
Fold 4 has also been inspected in previous experiments. France has no training
labels, so neither this gate nor local metrics certify its performance.

## L4 VM: one command

Recommended: **Google Compute Engine `g2-standard-32`, one L4 24 GB, 32 vCPUs,
128 GB RAM, Ubuntu 24.04 x86-64, 300 GB SSD Persistent Disk, on-demand**.
See [the Google Cloud setup and cost guide](GCP_r13.md) for exact VM settings,
driver installation and the creation command.

Use a separate clone if R12 is still running, so its code and work directory
stay stable. From the R13 checkout:

```bash
bash scripts/vm_r13.sh /absolute/path/to/student_resource
```

The resource directory must contain `dataset/{train,test}/*.tsv` and
`utils/validate_submission.py`. This launcher installs the pinned environment,
checks CUDA and the official validator, runs the tests, then launches detached.
It retains R12's 11-hour job deadline and a 12-hour fallback VM shutdown.
After success or failure it brings shutdown forward to ten minutes after exit,
unless the existing deadline is sooner. Logs and outputs remain on Persistent
Disk. Set `R13_SHUTDOWN_ON_EXIT=0` to retain only the fallback deadline.
It creates `.venv-r13`, `work/r13` and `output/r13`, with separate locks and
durable progress files. It does not submit anything to the website.

```bash
tail -f work/r13/runner.log
cat work/r13/run.json
# Same settings and checkout; completed stages are verified before reuse:
bash scripts/vm_r13.sh /absolute/path/to/student_resource --resume
```

`R13_MAX_HOURS`, `R13_THREADS`, `R13_ENCODER_PAIRS` and
`R13_CE_TRAIN_BUSINESSES` override the launch settings. Changing settings or
code requires a fresh work directory; changing only the deadline is allowed on
resume. Stopped VM disks and other provisioned resources may still incur costs.

## Desktop and direct execution

Use Python 3.12 and the same pinned requirements as R12. For Linux/WSL:

```bash
python3.12 -m venv .venv-r13
.venv-r13/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
.venv-r13/bin/python -m pip install -r code/business_entity_resolution/requirements_r10.txt
.venv-r13/bin/python -m pip check
.venv-r13/bin/python scripts/run_r13.py --profile desktop --preflight
.venv-r13/bin/python -u scripts/run_r13.py --profile desktop
```

On native Windows, use `py -3.12 -m venv .venv-r13` and replace the executable
with `.venv-r13\Scripts\python.exe`. Supply `--dataset` and `--validator` when
resources are elsewhere. Add `--resume` to the same command after interruption.
The direct Python runner stops its child processes at the deadline; only the VM
shell launcher schedules machine shutdown.

| Setting | `--profile vm` (default) | `--profile desktop` |
| --- | ---: | ---: |
| Intended hardware | L4 24 GB / 128 GB RAM | RTX 3060 12 GB / 32 GB RAM |
| CPU threads | 30 | 12 |
| Normalization window / queued chunks | 750,000 / 30 | 100,000 / 4 |
| Feature-shard pair cap | 6,000,000 | 2,000,000 |
| Encoding batch | 1,024 | 256 |
| CE train batch / accumulation | 64 / 1 | 16 / 4 |
| CE gradient checkpointing | off | on |
| CE mixed precision / optimizer | native BF16 / fused CUDA AdamW | FP16 / AdamW |
| CE inference batch ceiling | 1,024 | 128 |
| CE sampled training businesses | 250,000 | 180,000 |
| Job deadline | 11 hours | 24 hours |
| Free-disk reserve | 20 GiB | 12 GiB |

Both profiles use exact, tiled GPU neighbour search, three maximum CE epochs,
fold-9 early stopping, and inference batch reduction on CUDA OOM. The encoder
cap is one million: the R12 audit found only 833,203 eligible positive businesses,
so its two-million cap did not add training examples. Full-run time and peak
memory for R13 have not been benchmarked on either GPU. Reserve values are
minimum remaining space, not estimates of the entire dataset/work directory.

The VM normalization window now feeds all 30 workers; the earlier 100,000-row
window queued only four 25,000-row chunks. Records and normalization rules are
unchanged. BF16 is applied to CE forward/backward and inference, with FP32 loss
and stored logits. It requires native GPU support and disables unnecessary
FP16 gradient scaling. CPU tests retain FP32; retrieval precision is unchanged.
The requested precision and fused optimizer are recorded in the run settings
and `ce_model/training.json`.

Sibling feature construction reuses disk-backed embeddings; it does not train
another encoder or allocate a dense target-by-target matrix. The anchor index
uses about 76 MiB for 2.2 million source records and three slots. Embedding
gathers are bounded to 8,192 pairs. Mining retains the five-negative ceiling,
but actual row counts can increase because channel duplicates are refilled.

Inspect commands without starting work:

```bash
python scripts/run_r13.py --profile vm --plan
python scripts/run_r13.py --profile desktop --plan
```

The runner preserves older internal filenames such as `r10_models.json` for
compatibility. Its metadata version is `r13-mined-siblings-1`. New artifacts are
`r13_evidence_{train,test}.{parquet,json}`; `ce_model/training.json` records the
mining strategy. `selection.json` records whether R13 passed the gate.

Only a completed `work/r13/result.json` with official validation `PASS` signals
finished output. Upload `output/r13/matching_results.tsv` to measure the website
score; `candidate_pairs.tsv` is also validated. Do not interpret the local
99% reporting objective as a promised website result.

See [the review and validation record](../reports/r13_review.md).
