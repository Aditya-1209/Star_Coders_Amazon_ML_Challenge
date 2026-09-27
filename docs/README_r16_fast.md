# R16-fast: business presence using completed R12 evidence

This is the deadline-sized alternative to `codex/r16-l4`, on branch
`codex/r16-fast`. It reuses immutable `work/r12` normalization, candidates,
neural scores and the **actual saved R12 selection**. It does not fine-tune
an encoder/CE, retrieve neighbors, tokenize or score with the CE again.

It builds the tested R15 cross-source/alternative features, fits two enhanced
pair models and one business presence model. Presence uses only aggregates
of earlier out-of-fold evidence, not fitted final pair predictions. Full
organizer truth labels businesses, including positives missed by retrieval.
Businesses with no candidates are included. Missing evidence has an explicit
indicator. No ID, country, candidate counts or IDF become presence features.
Presence filtering precedes exclusive target ownership.

Models train on 6/7; early stopping and proposal selection use 3A. One proposal
is gated once on 3B against the frozen R12 predictions and threshold using the
**unchanged .0005 / positive one-sided lower bound / .002 country guard**.
Failure writes the exact R12 decisions. Fold 4 is report-only after selection.
It is not pristine: earlier experiments inspected fold 4 and inherited models
used fold 3. Local gains do not guarantee the .99 public leaderboard goal;
France has no labels. This cannot raise the R12 retrieval ceiling.

## Same VM, after R15 finishes

Use a separate clone; do not switch branches under a running job. Keep the
original R12 environment and original dataset path (preflight verifies both).
The launcher refuses to start while the original R15 manifest is `running`.
Do not run the full `vm_r16.sh` launcher for this experiment.

```bash
cd ~
git clone --branch codex/r16-fast --single-branch https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git Star_Coders_r16_fast
cd ~/Star_Coders_r16_fast
R16_BASE_WORK="$HOME/Star_Coders_Amazon_ML_Challenge/work/r12" \
R16_PYTHON="$HOME/Star_Coders_Amazon_ML_Challenge/.venv-r12/bin/python" \
R16_MAX_HOURS=3 bash scripts/vm_r16_fast.sh "$HOME/Star_Coders_Amazon_ML_Challenge/student_resource"
tail -n 20 work/r16_fast/runner.log
```

The three-hour limit is a cap, **not a completion estimate**. Full-scale R16-fast
runtime/peak RAM/score have not yet been measured. R15's feature stages took
about 4m23s on this VM; the extra presence classifier/search add work. Allow time
for downloading and submitting; retain R12/R15 as fallbacks. The launcher prints
the existing shutdown timer and **does not cancel, extend or shorten it**.
The known current VM timer is 27 September 2026 17:44 UTC / 23:14 IST; download
before then. Cloud runtime limits also remain independent.

Required checks run before detached launch: dependency consistency, immutable
parent/dataset/fold provenance, GPU probe, disk reserve, full tests and plan.
No packages are installed and no new cloud resources are created. Resume uses
the same invocation with `--resume`; code/data/settings must remain identical.

After `Complete`, read:

```bash
cat work/r16_fast/result.md
cat work/r16_fast/logs/validate.log
cat work/r16_fast/selection.json
```

Download `~/Star_Coders_r16_fast/output/r16_fast/matching_results.tsv` only after
completion and official validation PASS. Selection records `r16_fast` or `r12`.
The complete branch `codex/r16-l4` remains a separate full-training experiment.

For an offline command plan:

```bash
python3 scripts/run_r16_fast.py --plan
```
