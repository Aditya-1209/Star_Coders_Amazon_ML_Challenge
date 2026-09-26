# r10 + r11 desktop run logs (26 Sep 2026)

Hardware: i9-13900K, 32 GB RAM, RTX 3060 12 GB, Windows 11. Work dir was `D:\star_r10b`.

## Scores

| Run | Holdout F0.5 (fold 4) | Precision | Recall | US | India | Leaderboard |
|---|---:|---:|---:|---:|---:|---:|
| r6.1 stage 2 | 0.9624 | 99.37% | 91.6% | 0.967 | 0.956 | 0.949 |
| r8 fast path (stage 2 + neural features) | 0.9694 | 99.71% | 92.6% | 0.973 | 0.965 | 0.954 |
| **r10 stage 2** (neural retrieval) | **0.9870** | 99.70% | 96.76% | 0.9863 | 0.9880 | **0.975** |
| r10 graph stage 3 | 0.9877 | 99.73% | 96.99% | n/a | n/a | not uploaded |
| r11 stage 2 (r10 + `--lookalike --record-competition`) | 0.9872 | 99.74% | 96.78% | 0.9866 | 0.9882 | pending (France probe) |

Candidate ceiling (oracle F0.5): 0.979 with key + phonetic blocking; **0.9987** with neural top-24 neighbours merged.
Train-vs-test shift AUC (`metrics/shift_check.json`): India 0.648, US 0.692. There is no split-dependent shift.

France (15% of test, no labels) is inferred from the leaderboard at about 0.90 for r10 (0.87 before).

## Findings

- Retrieval is solved. The remaining US/India loss is the classifier: 89% of missed true
  pairs were retrieved and rejected (median score 0.21), and **72.5% of those have no address**.
- r11's name-competition and look-alike features change the holdout by only +0.0002. The
  no-address cases are ambiguous on names alone and need richer evidence (cross-encoder,
  sibling records, digits in names).
- Neighbour search: CPU FAISS took about 2 hours per split. Exact GPU search took 11 minutes on
  the 3060 and gives identical output (`logs/ann_test_gpu.log`).
- Cross-encoder training on the 3060 ran at 320 pairs/s (batch 16 × 4 with checkpointing),
  about 75 minutes per epoch. It was stopped at the start of epoch 1 (`logs/ce_train.log`) and moved to the VM run (r12).
- At 17:05 the PC shut down uncleanly (Kernel-Power 41, bugcheck 0, no dump; likely a power or
  hardware fault) and Windows Update then restarted it. `run.json` was left zero-filled, so resume
  was impossible. The remaining stages were run by hand (`logs/recover.log`). r12 fixes this with fsync before rename.

## Files

- `logs/`: one log per stage. `recover.log` is the manual-recovery timeline; `runner.log` is the original runner.
- `metrics/`: `*_metrics.json` hold holdout and tuning metrics, per-country scores, thresholds and feature lists.
  `shift_check.json` is the train-vs-test gate, `ann_train.json` the FAISS audit, `key_*.json` blocking stats.
