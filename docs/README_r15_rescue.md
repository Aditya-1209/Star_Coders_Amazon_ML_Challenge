# R15 address rescue

Branch `codex/r15-address-rescue` is a separate experiment based on the measured
R15 failure, not an overwrite of its completed artifacts or the running R16.

The supplied R15 3A audit found 9,928 false-negative pairs, of which 8,743 were
retrieved. 7,206 of those rejected true pairs (82.42%) had no target address.
R15's broad classifier/blend gained only .00004943 on 3B, below the original
.0005 gate, with a negative confidence lower bound. Country regression was not
the blocking condition. R15 correctly retained R12.

This experiment fits two missing-target-address specialists (pair-weighted and
business-balanced) on 6/7, early stopping on 3A. It uses existing R12 neural
scores plus R15 alternative/other-source evidence. A sibling's similarity
features are exposed as trusted only when its CE logit is at least 2.2 and it
beats its own source's second-best candidate by at least 1; a single candidate
still requires that logit. No labels select the donor; there is no self-support.
Missing addresses remain null evidence, not agreement. Raw untrusted sibling
similarity features are excluded from the specialist. Count/IDF exclusions and
incomplete target-context exclusions remain intact.

Only missing-target-address scores are blended with R12; all other candidate
scores remain identical to R12. Global/country decision thresholds are tuned on
3A, so addressed decisions and exclusive ownership can still change. A single
proposal faces the unchanged .0005/positive lower-bound/.002 country guard on
3B. Fold 4 is report-only. Failure still selects actual R12: retaining a weak
proposal would not make it a better model. Previous experiments have examined
3B/fold 4; this is not a fresh nested validation experiment. Website score and
France behavior remain unmeasured. Existing missing candidates are not rescued.

## Run after R16-fast completes

Use the same VM, existing data and environment, in a separate clone. Do not
switch code under R16 or run both heavy jobs on the single L4. The launcher
checks the standard R15/R16-fast manifest paths and refuses to start while
either is `running`. Override `R15_RESCUE_R16_WORK` if R16 runs elsewhere.

```bash
cd ~
git clone --branch codex/r15-address-rescue --single-branch https://github.com/Aditya-1209/Star_Coders_Amazon_ML_Challenge.git Star_Coders_r15_rescue
cd ~/Star_Coders_r15_rescue
R15_RESCUE_BASE_WORK="$HOME/Star_Coders_Amazon_ML_Challenge/work/r12" \
R15_RESCUE_PYTHON="$HOME/Star_Coders_Amazon_ML_Challenge/.venv-r12/bin/python" \
R15_RESCUE_MAX_HOURS=3 bash scripts/vm_r15_rescue.sh "$HOME/Star_Coders_Amazon_ML_Challenge/student_resource"
tail -n 20 work/r15_rescue/runner.log
```

The three-hour cap is not an expected runtime or guarantee. This reuses neural
models and scores but reconstructs evidence in a fresh work directory. The
whole prepared feature table is read once to add confidence features; full-data
memory/time have not been measured on the 128 GB VM. The normal disk reserve,
immutable parent/data/environment checks, lock/resume fingerprints and official
submission validation remain. The launcher preserves the shutdown timer: the
known current deadline is 23:14 IST on 27 September 2026. Finish and download
before that timer and before the website deadline; keep R12/R16 outputs.

After `Complete`:

```bash
cat work/r15_rescue/result.md
cat work/r15_rescue/logs/validate.log
cat work/r15_rescue/selection.json
```

Download `~/Star_Coders_r15_rescue/output/r15_rescue/matching_results.tsv` only
after official validation PASS. The report names `r15_rescue` or `r12` as the
selected model. No .99 public score is claimed in advance.

For a command plan: `python3 scripts/run_r15_rescue.py --plan`.
