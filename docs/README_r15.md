# R15: cross-source corroboration after the completed R12 run

Goal: **0.99 on the challenge leaderboard**. R12 reached 0.990387533 locally
and **0.984 online (user-reported)**. R15 has no measured full-data or website
result yet. The improvement must be demonstrated by running and submitting it.

## Why this experiment

R12's CE fusion improved the graph reference from 0.988420989 to 0.990387533
on fold 4. Precision is 0.998661580; recall is 0.973913283. Candidate oracle
F0.5 is 0.999081939: accepting/rejecting retrieved candidates remains the main
opportunity. The older 72.5% missing-address error figure predates R12's
completed CE and must not be treated as a measured R12 error breakdown.

R15 starts by measuring R12's remaining errors on **3A**, including missing
addresses, retrieved-but-rejected pairs, retrieval misses and false positives.
It then fits new final classifiers using two additions:

1. **Other-source corroboration selected by CE.** For a Source 2 candidate,
   compare its normalized name/address with this business's highest-CE Source
   3 candidate, and vice versa. Features include CE strength, name/address
   string agreement, numerical-address overlap and CE margin. A candidate
   cannot corroborate itself. Missing addresses or missing sibling records
   produce missing evidence, not perfect agreement. Weak or conflicting
   corroboration is learned by the classifier rather than forced into matches.
2. **Signed alternative margins within each business and target source.**
   Compare each candidate with the best *other* candidate's CE and lexical
   evidence. A winner now has a positive margin when it wins decisively;
   the existing max-minus-current gap is always zero for a winner.

No newly computed feature uses corpus frequencies, IDF, absolute candidate
counts, country identity, entity IDs as numbers, or labels. Feature construction
precedes fold filtering. Existing count/IDF exclusions remain unchanged.
Final classifiers additionally exclude incomplete cross-business CE/neural
target ranks and gaps (`ce_rank_t`, `ce_gap_t`, `ncos_rank_t`, `ncos_gap_t`),
because training CE tables expose only graph folds whereas test exposes every
business. This correction is also present in the independently running R13.

## Relationship to other runs

R13 was inspected at `bfb2ef3`: it retrains CE with stage-2 hard negatives and
adds embedding-based support from up to three high-confidence stage-2 matches.
R15 instead uses the already trained **R12 CE**, chooses a cross-source sibling
with CE, and measures lexical/address agreement. It does not duplicate the
expensive R13 CE retraining experiment. No `r14` branch was available remotely
at review time; its implementation could not be compared.

## Preserved validation rules

- Reuse the completed encoder (folds 0/1/8/9) and CE (fit 0/1/8, validation 9).
- Final model fitting uses 6/7; early stopping and model/threshold selection
  use 3A. The original gain >=0.0005, positive confidence lower bound and
  country-regression gate run once on 3B.
- **The comparison and fallback are the actual saved R12 fusion model**,
  which produced the 0.984 submission, not the weaker graph reference.
- Fold 4 is report-only after selection is frozen. No threshold is changed
  based on its result. Earlier experiments have inspected fold 4, and inherited
  models have already used fold 3; this is not an entirely fresh validation set.
- France has no labels. Local gate success does not establish France or
  leaderboard improvement. If the gate rejects R15, output uses R12's frozen
  model, thresholds and exclusivity decision.

## Launch on the existing GCP VM

Use the existing `g2-standard-32` VM: L4 24 GB, 32 vCPU, 128 GB RAM,
Ubuntu 24.04 and Python 3.12. The completed `work/r12`, original dataset,
official validator and `.venv-r12` must remain available. R15 writes only
`work/r15` and `output/r15`; it does not modify or copy the large R12 caches.

From the VM repository, with no uncommitted source edits:

```bash
git remote set-branches --add origin r15
git fetch origin
git switch --track origin/r15
bash scripts/vm_r15.sh "$PWD/student_resource"
```

If local branch `r15` already exists, use `git switch r15` followed by
`git pull --ff-only origin r15` instead of creating it again.

The launcher checks provenance, matching data/validator fingerprints, CUDA,
dependencies and free disk; runs the full tests; and detaches the job. It
re-arms VM shutdown for **seven hours after setup**: a six-hour job budget
plus one-hour download buffer. This is a limit, not a measured runtime or ETA.
`R15_MAX_HOURS` changes the budget, `R15_THREADS` the threads, and
`R15_BASE_WORK` the completed parent path. Parent dataset paths and files must
match the original R12 manifest; do not relocate or edit them during this run.

```bash
tail -f work/r15/runner.log
cat work/r15/logs/audit.log
tail -n 20 work/r15/logs/features_test.log
cat work/r15/result.md
# Only after an interruption, same code/settings/data:
bash scripts/vm_r15.sh "$PWD/student_resource" --resume
```

Stages: baseline audit; train/test evidence; final fitting; selection; holdout
evaluation; test inference; official validation. Evidence logs include exact
processed/total pair counts per country. No encoder training, embedding,
retrieval, CE training or CE scoring is repeated. The feature stages are
primarily CPU work; final XGBoost training uses the L4.

Submission: `output/r15/matching_results.tsv`. Wait for `Complete` and
validation PASS, download the output, then stop the VM to avoid idle compute.
An existing `work/r15/run.json` requires checked resume; parent/code/settings
changes reject resume. Interrupted stages restart; completed stages are
fingerprint-verified. Low disk space or deadline expiry stops the child process
group; it does not publish a partial submission as complete.

Inspect the plan without requiring VM artifacts:

```bash
python scripts/run_r15.py --plan
```
