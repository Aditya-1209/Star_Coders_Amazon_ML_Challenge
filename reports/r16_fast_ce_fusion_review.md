# R16 fast CE fusion: implementation review, 2026-09-27

Branch: `codex/r16-fast-ce-fusion`, based on the completed R12 branch at
`7ba5312`. This ships a new final model and bounded runner, not the audit-only
`fcb5a53` change, the 16-hour R16 pipeline, or a renamed failed presence/rescue run.

The new model fits business-balanced, depth-4 XGBoost fusion of R12 and the
separate R14 graph/CE-A/swapped-CE-A/CE-B predictions. All new optimizer,
early-stopping and threshold-selection labels are restricted to R12 3A.
One proposal faces the original `r10.gate` on 3B: 0.0005 gain, positive
approximate one-sided lower bound, maximum 0.002 country regression. The old
`r10.py` and `decision.py` are unchanged. Fold 4 is read after selection.
Rejected proposals must reproduce both original R12 TSV SHA-256 hashes.

The teammate's final R14 fusion scores are not usable in this new training
protocol: cross-fitting makes the 3A predictions depend on a model fitted
with 3B labels. The teammate also reports that the simple final-score blend
has already been tried (+0.00007 on its fold-4 comparison). This implementation
requires existing pre-fusion components instead; no fresh R12 export or
neural training is requested. The score manifest checks declared provenance,
and original donor train ID row order is checked against R12 fold membership.
Producer declarations cannot be independently established from score values.

## Validation

- `python -m unittest discover -s tests_v2 -v`: **66 passed in 47.421 s**.
- Twelve new tests cover mapped IDs (including differing donor indices),
  duplicate/unknown/non-finite rejection, union/missing-score behavior,
  training-fold compatibility, optimizer exclusion of 3B/4, one unchanged gate,
  delayed holdout access, exact fallback hashes, accepted-model output,
  official validator `--check-ids`, parent immutability, checked resume,
  expired deadlines and termination of an actively running child process.
- End-to-end fixture fits real CPU XGBoost trees. Both accepted synthetic
  proposals and frozen fallback outputs pass the organizer validator.
- `scripts/run_r10.py --plan` passes. The new plan has exactly four stages:
  select (including small fusion fit), evaluate, inference, official validate.
- Shell syntax and fake-interpreter launcher forwarding pass. The new launcher
  keeps the original venv executable path and issues no install/shutdown calls.
- `git diff --check` passes.

Tests ran locally on macOS / Python 3.12, Polars 1.44.2, NumPy 2.5.3,
XGBoost 3.4.1, PyArrow 25.0.1, two threads, with existing temporary dependency
packages. The two inherited Linux R12 launcher tests required a local
`flock -n FD` adapter using Python `fcntl.flock`, and canonical
`TMPDIR=/private/tmp` paths. No production scripts were changed for this
platform adjustment. This is not a native Linux/L4 execution claim.

## Production status

No full-data fusion training or new website submission has run. The actual
pre-fusion R14 files have not been supplied/inspected, so production launch
is blocked until they and the original donor ID mapping are available.
The provided final `fusion_ab` score files must not be substituted.

Estimated VM compute time: 10–30 minutes after inputs arrive, excluding
transfer; unbenchmarked on full-data L4. Default wall cap: 45 minutes.
The absolute shutdown argument stops work ten minutes before the existing
VM shutdown (17:34:11 UTC for the stated 17:44:11 UTC cutoff). Original
`work/r12`, `.venv-r12`, export and timers are not modified.
See `docs/README_r16_fast_ce_fusion.md` for exact artifacts and launch commands.
