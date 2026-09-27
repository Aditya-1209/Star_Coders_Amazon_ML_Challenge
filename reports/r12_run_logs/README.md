# R12 GCP run — 27 September 2026

Completed run using reviewed branch `r12` at `06656f9` on a GCP
`g2-standard-32`: 32 vCPUs, 128 GB RAM, NVIDIA L4 24 GB, Ubuntu 24.04,
Python 3.12. Entry point:

```bash
R12_MAX_HOURS=16 bash scripts/vm_r12.sh "$PWD/student_resource"
```

The runner started at 2026-09-26 20:55 UTC. Final validation started at
2026-09-27 09:24 UTC and passed. Total runtime was approximately 12.5 hours.

## Results

| Metric | Final fusion | Graph reference |
| --- | ---: | ---: |
| Fold-4 macro F0.5 | 0.990387533 | 0.988420989 |
| Pair precision | 0.998661580 | 0.997662766 |
| Pair recall | 0.973913283 | 0.971249102 |

India macro F0.5: 0.991566657. US macro F0.5: 0.989603450.
Candidate oracle macro F0.5: 0.999081939. Selection gate passed on fold 3;
fold 4 was used for reporting. The inherited `selected: r10` and
`r10-ce-ann-1` names identify the fusion implementation within this r12 run.

The 0.99 local target was met. The user subsequently reported a **0.984
leaderboard score** for this submission. Raw run metrics remain unchanged:
their leaderboard field was unmeasured at the time the VM run finished.
France has no training labels, so there is no measured France holdout score.

Cross-encoder validation improved across all three epochs:

| Epoch | Training loss | Fold-9 validation loss |
| --- | ---: | ---: |
| 1 | 0.131447817 | 0.054947678 |
| 2 | 0.074790856 | 0.043568462 |
| 3 | 0.066053058 | 0.042298641 |

## Included artifacts

- `runner.log`: stage transitions, preflight versions, completion marker.
- `logs/`: all stage logs from the supplied VM archive, unchanged. Token
  preparation logs are empty because those stages emit no progress output.
- `metrics.json`, `selection.json`, `result.md`: final evaluation, selection
  gate and completion report.
- `submission/matching_results.tsv.gz`: compressed copy of the supplied
  submission; decompress before uploading to the challenge website.
- `submission/manifest.json`: original byte count, compressed byte count and
  original SHA-256. Decompression was checked against that SHA-256.

Official validation found all 1,732,544 required source-1 rows: 99,787 empty
matches and 1,632,757 non-empty matches. Candidate output also passed, but
`candidate_pairs.tsv` was not supplied in this archive.

The original submission SHA-256 is:

```text
418fce53733fc3c01dc4da00efe0409e2a3eb14faa17910cb6cb4b32169e5953
```

Raw logs retain the VM username, paths and timestamps for reproducibility.
No dataset or model weights are included. A credential-pattern scan found
no matches in the supplied logs and metadata.
