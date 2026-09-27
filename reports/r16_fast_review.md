# R16-fast review

The full `codex/r16-l4` branch at `f2e2b60` retrains the neural pipeline and
plans 12–16 hours on the existing L4. It cannot be treated as a dependable
same-day run before 23:30 IST. This separate `codex/r16-fast` branch retains
the useful final-layer business-presence idea and reuses completed R12 evidence.

Changes and reasons:

- Reuse R12 normalization, graph candidates, CE/neural scores, model weights
  and frozen selection. Skip all encoder/CE/retrieval passes and installations.
  Full parent/data/dependency/fold checks from the tested R15 runner remain.
- Use R15's label-free opposite-source corroboration and within-source
  alternative margins. No self-support; missing addresses never mean agreement.
- Fit two enhanced pair models and one shallow presence classifier. The
  presence classifier uses earlier out-of-fold evidence aggregates, never
  predictions from final models fitted on its own labels. Labels use complete
  truth so unretrieved positives are not mislabeled singletons. Include all
  businesses, including those without candidates. Exclude candidate counts,
  country IDs, record IDs and IDF from its feature set.
- Search presence cutoffs 0/.25/.5/.75/.9/.95 on 3A only, including actual R12
  scores plus the existing enhanced pair variants/blends. Suppress businesses
  before exclusive target ownership so rejected businesses cannot steal matches.
- Keep folds 6/7 for final training and 3A for early stopping/proposal selection.
  Gate a single proposal on 3B with the original .0005 gain, positive approximate
  one-sided 95% lower bound and .002 labeled-country guard. Full R16's weaker
  .0002 gate and rebuilt reference are not carried over. Failed proposals use
  exact completed R12 decisions. Report fold 4 only after selection is frozen.
- Separate clone and launch lock; refuse launch while R15 is running. Reuse
  original Python/data paths; do not change the existing shutdown timer. Default
  three-hour job cap leaves time to download but does not guarantee completion.
- Produce both required TSVs, preserving every Source 1 row including France
  and empty matches, then run the official ID-aware validator before completion.

This is a reduced experiment, not an equivalent shortcut to full R16 training.
It retains R12's retrieval ceiling and cannot learn France-specific labels.
The current r12 public score is .984 (user-reported), while local fold-4 macro
F0.5 is .990387533. R16-fast has no measured full-data GPU/public score yet.
Inherited models used fold 3 and earlier experiments inspected fold 4, so
validation is not fully nested or pristine. A local gain cannot certify .99.

Validation: Python 3.12 `python -m unittest discover -s tests_v2` passed
72 tests in 201.099 seconds (six Linux-only skips on Windows). Two new Linux
launcher tests passed separately on Ubuntu in 2.256 seconds. The real offline
tiny-BERT pipeline exercised feature generation, all three final classifiers,
selection, frozen fold-4 reporting and both output paths. Both outputs passed
the official ID-aware validator, including France and empty-match rows. R12
fallback TSV was byte-identical and parent artifacts remained unchanged.
Focused presence tests passed (six); Windows/Ubuntu fast plans, the original
`run_r10.py --plan`, bash syntax and `git diff --check` passed. No full-size
R16-fast L4 run has been benchmarked. See `docs/README_r16_fast.md` for launch.

While preparing this branch, the user confirmed R15 completed in about ten
minutes and its gate selected R12. R16-fast's business-presence classifier is
therefore the distinct additional hypothesis; success is not assumed.
