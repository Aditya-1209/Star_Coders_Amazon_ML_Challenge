# R16 implementation and validation record

Objective: improve the available R11–R13 pipeline toward **0.99 website macro
F0.5**, on the existing Mumbai L4 / 128 GB / 200 GB balanced-disk VM.
Full R16 training and website performance: **not measured**.

## Reviewed evidence

| Source | Evidence |
| --- | --- |
| R11, code through `11792f6`, published `r11-logs` | Local stage-2 macro F0.5 0.987234331; user-reported website 0.977. Earlier error audit points to retrieved-but-rejected and missing-address pairs. |
| R12 code `06656f9`, results published in `7ba5312` | Completed L4 run: local fusion 0.990387533 versus graph 0.988420989; precision 0.998661580, recall 0.973913283; no recorded website result. |
| R13 `bfb2ef3` | Mined CE negatives, neural sibling evidence, final context exclusions, L4 BF16/fused optimizer and RAM token cache. No completed full-data R13 result was available. |

R14/R15 were unavailable in the remote branch list and local project. The user
then authorized proceeding with another model using the available work.
R16 starts from `bfb2ef3`, rather than inventing results for missing versions.

The measured [R12 results](https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge/blob/7ba5312fa6df86e12e3d42b97ce1168d8a5d23ce/reports/r12_run_logs/README.md)
show that dropping back to graph alone loses about 0.20 local percentage points.
This motivates retaining a CE-fused reference in the new validation gate.
The [R12 stage log](https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge/blob/7ba5312fa6df86e12e3d42b97ce1168d8a5d23ce/reports/r12_run_logs/runner.log)
shows approximately 12.5 hours end to end: 2.9 hours CE training, 2.3 hours CE
scoring, 2.1 hours feature generation, 1.5 hours encoding, with the rest in
preparation, encoder fine-tuning, retrieval, graph models and final decisions.
The new default allows 16 hours of job time rather than relying on an 11-hour
cap to complete a longer measured workload.

## Changes and limitations

R16 uses one R12-recipe CE, retaining all positive matches, lexical/neural/random
negatives and target-address dropout. It deliberately disables R13's mined
negative sampling for this controlled experiment. BF16, fused optimization and
the RAM cache are retained, so the rebuilt reference is not asserted to be
bit-identical to the historical R12 model or to reproduce its score.

Two final ensembles compare the R12 feature family against sibling evidence
with incomplete CE/neural target competition features removed. The new business
classifier learns whether any true match exists. It aggregates only explicitly
listed frozen CE/stage-2 and pair-local inputs, plus sibling evidence. Final
pair-model predictions are not stacked into its training features. Full ground
truth labels preserve positive businesses even if retrieval missed all matches.

Pair models use folds 6/7; the business classifier uses the same folds. Final
early stopping, thresholds, blends and business rejection cutoffs use 3A. Only
the winning proposal is compared with the rebuilt CE reference on 3B. Acceptance
requires gain >=0.0002, a positive approximate paired 95% lower bound and no
country drop beyond 0.0005. Ties/failures keep the reference. These cutoffs are
fixed before running this experiment. Fold 4 is report-only; its labels cannot
change the stored selection. No unknown-country accuracy is inferred.

This protocol still inherits earlier layers' use of fold 3 and past inspection
of fold 4. A passed local gate does not guarantee a website improvement. The
final workflow writes one pair of official-format output files using the selected
models and records both local and still-unmeasured website objectives.

New training/inference entry points use isolated R16 paths. The generic runner
accepts a command builder instead of mutating its global functions, and includes
the R16 wrapper in code fingerprints. Existing R10/R13 command defaults remain
unchanged. No cloud VM is created or training launched by preparing this branch.

## Validation

On 2026-09-27, Python 3.12 on macOS ARM, with the pinned NumPy/Polars/XGBoost
environment and temporary neural dependencies:

```text
python -m unittest discover -s tests_v2 -v
Ran 81 tests in 94.062s
OK (skipped=6)
```

**75 passed; six Linux launcher tests skipped** because this Mac lacks `flock`
and `sha256sum`. R12/R13/R16 launch contracts remain runnable on Linux. Real
shell tests with fake Python/sudo verify R16 supervisor success, failure,
opt-out and preservation of earlier shutdown deadlines without stopping a host.

The R16 offline integration fixture trains a real tiny BERT cross-encoder,
saves/reloads it, performs FAISS retrieval, XGBoost/graph prediction, sibling
construction, all four final pair models and the business classifier. It fits
thresholds/gates, reports fold 4, and validates both reference and enhanced
inference output paths using the organizer's validator. Both output files
contain every synthetic test business, including France and empty matches.
Re-evaluation leaves the saved selection unchanged.

Additional checks verify label-free/subset-stable business aggregates, boolean
feature handling, stable integer business IDs, no-candidate businesses,
reference/enhanced feature isolation, full-truth presence labels, exclusion of
gate/holdout labels from fitting, target reassignment after business rejection,
fallback on ties, and rejection of a proposal that improves 3A but degrades 3B.
The generated R16 command plan and both shell scripts' syntax pass.

No full R16 GPU training, CUDA memory benchmark, measured cloud speedup or
website submission was performed. The tiny fixture establishes execution and
output correctness, not challenge accuracy. The source branch contains no
full trained model weights; the documented VM run generates them.
