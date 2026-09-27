# R12 on the GCP L4 VM

For the **France-only, 3–4 hour budget follow-up**, use
[the cached France guide](README_r12_france.md). It requires the completed R12
cache and preserves India/US predictions. The full-training instructions below
took approximately **12.5 hours** in the recorded run and are not the short profile.

Target: Ubuntu 24.04, Python 3.12, `g2-standard-32` (32 vCPUs,
128 GB RAM, one NVIDIA L4 with 24 GB VRAM). Install the NVIDIA driver and
provide outbound internet access (Cloud NAT when VM external IPs are prohibited).
The pretrained E5 backbone is still pinned to the existing revision.

The second review uses the shared r10/r11 run logs. See
[the accuracy review](../reports/r12_accuracy_review.md) for evidence, changes,
and validation. R11's user-reported leaderboard score is 0.977; 0.99 remains
an unmeasured target, not an expected result from larger hardware alone.

## Run

From the reviewed **r12** checkout, with the VM running:

```bash
bash scripts/vm_r12.sh /absolute/path/to/student_resource
# Identical code/data/settings, after interruption:
bash scripts/vm_r12.sh /absolute/path/to/student_resource --resume
# Longer allowance agreed for this experiment (16-hour job / 17-hour VM):
R12_MAX_HOURS=16 bash scripts/vm_r12.sh /absolute/path/to/student_resource
```

The argument must contain the original seven TSVs under `dataset/{train,test}`
and `utils/validate_submission.py`. Both paths are passed explicitly; an old
repository-local `student_resource` directory cannot redirect the run.

The launcher verifies Python dependencies, CUDA and XGBoost, runs the full
`tests_v2` suite, saves the command plan, and launches with nohup. A launch lock
prevents duplicate jobs and logs are appended on resume. An interrupted package
installation is repaired on the next attempt rather than mistaken for readiness.

```bash
tail -f work/r12/runner.log
cat work/r12/run.json
tail -f work/r12/logs/train.log
```

## Profile and review changes

| Setting | R12 VM value | Reason |
|---|---|---|
| Encoder pair cap | 2,000,000, one epoch | Upper bound only; this dataset has at most 833,203 eligible positive businesses, so 1M was already sufficient |
| CE training business cap | 250,000 (was 180,000) | More labelled businesses and hard negatives; more training time |
| CE training batch / accumulation | 64 / 1, no checkpointing | Retains effective batch 64 with fewer forward/backward calls |
| Encode / CE score batch | 1,024 / up to 1,024 | CE length bucketing reduces padding; CUDA OOM halves its batch and retries the same pairs |
| Threads / feature shard | 30 / 6,000,000 pairs | Existing 128 GB profile; throughput still needs measurement |
| Neural neighbours / stage-1 rescue | 24 / 8 | Retained: reported retrieval ceiling leaves little reason to enlarge blindly |
| Disk reserve | 20 GiB | Checked before and during stages, so a stage can stop before filling the disk |
| Runtime / VM shutdown | 11 hours / 12 hours | Restores the session's original cloud spending window |

The encoder cap is **not** a promise of two million examples: selection remains
one positive per distinct eligible business/target, followed by the hard-negative
filter. The actual count is logged and saved in `neural_e5/finetune.json`.
The CE cap counts businesses, not pairs; actual counts appear in
`ce_model/training.json`. Increasing either cap can change accuracy in either
direction; this review does not measure an improvement.

CE training now retains all true matches for each sampled business, instead
of keeping only four in Source-2-first target order. On the actual 250,000
business training sample this restores 106,025 positive pairs. Address dropout
is independent per pair and removes only the target address, preserving the
field separator and the complete Source 1 address. This matches the raw data's
missing-field pattern. Neural optimizer and validation folds are unchanged.

GPU neighbour search remains exhaustive at the stored fp16 search precision,
not a CPU ANN approximation. Query batches and target tiles are sized from
available VRAM. Country-scoped Source 3 global IDs are preserved. Candidate
rows stream to a temporary parquet and only replace the final file on success,
instead of retaining every country's output in host RAM.

Record-side and look-alike competition now also feed stage 3 and CE fusion.
Graph record competition includes label-free direct-pair similarities from
businesses outside its fitting/evaluation folds, plus generated graph candidates.
Stage-2 record competition stays global across feature shards. CE ranks/gaps
are computed over the complete available CE score file before loading a subset
of folds, so selecting fold 4 cannot erase its competitors. Graph training still
uses only its original eligible folds; this does not claim identical train/test
population density or eliminate France distribution shift.

All original fold assignments are preserved: encoder folds 0/1/8/9; CE
optimization folds 0/1/8 with fold 9 early stopping; fold 4 remains report-only.
The split-dependent count/IDF exclusion and 3A/3B selection gate are unchanged.
The France estimate in the task context is not a measured France score.
The VM launcher now reports against a 0.99 local objective; that reporting
setting does not change thresholds, the selection gate, or leaderboard claims.

Override caps/threads/deadline **before starting a fresh run**, for example:

```bash
R12_ENCODER_PAIRS=1000000 R12_CE_TRAIN_BUSINESSES=180000 \
  bash scripts/vm_r12.sh /absolute/path/to/student_resource
```

`R12_THREADS` and `R12_MAX_HOURS` are also supported. The launcher schedules
VM shutdown one hour after the chosen Python limit; reboots clear that timer.
During setup the existing VM timer still applies. Disk and NAT costs remain
after shutdown. Resume requires the same settings; an interrupted stage starts
over, not from a mid-epoch checkpoint. Code changes invalidate old R12 caches.

## Before sharing or committing

```bash
export PYTHONPATH=code/business_entity_resolution/src
python -m unittest discover -s tests_v2
python scripts/run_r10.py --ann gpu-exact --r11-features \
  --encoder-pairs 2000000 --ce-train-businesses 250000 --plan
bash -n scripts/vm_r12.sh
```

The offline smoke test uses a tiny local BERT and real tree/graph/CE fitting;
it downloads no model. It now checks the new features reach graph training and
inference, and releases its memory-map handles before Windows temp cleanup.
CPU smoke results do not establish L4 throughput, peak memory or a leaderboard
score. Check the first real run's logs and `result.json` before submitting.
