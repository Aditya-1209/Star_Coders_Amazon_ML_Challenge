# r7 overnight experiment — 26 September 2026

The completed CPU run scored **0.967381 macro F0.5 (96.74%)** on 220,507
held-out training businesses. **It did not reach 97–98, and this is not an
Amazon leaderboard score.** Both test output files passed the official validator
with ID-existence checks for all 1,732,544 required Source 1 businesses.

| Holdout result | Macro F0.5 | Pair precision | Pair recall | Businesses |
|---|---:|---:|---:|---:|
| Stage 2 | 0.963950 | 0.994018 | 0.919314 | 220,507 |
| Final stage 3 | **0.967381** | 0.995076 | 0.930945 | 220,507 |
| Final stage 3 — US | 0.971064 | 0.995587 | 0.937171 | 132,439 |
| Final stage 3 — India | 0.961843 | 0.994298 | 0.921625 | 88,068 |

The branch previously documented 0.9658 for r6.1 stage 3. The new result is
approximately 0.16 percentage points higher than that historical local score;
the old submission was not rerun in this experiment. The user's approximately
95 Amazon score is a different evaluation and cannot be compared directly.

## What the controlled comparisons showed

All new retrieval probing, feature selection, cutoff selection and blend selection
used tuning fold 3. Fold 4 was used only for reporting. Supervised transliteration
excluded both folds 3 and 4.

* The multi-word phonetic channel recovered 1,678 additional true pairs on the
  220,699-business tuning fold. Direct candidate recall rose from 0.943897 to
  0.946093; mean candidates rose from 67.34 to 69.25 per business.
* With the same expanded candidates and stage-1 ensemble, the new stage-2 name
  features improved tuning F0.5 from 0.963077 to 0.963913: +0.084 percentage points.
* Extra stage-3 name features had a much smaller tuning gain, from 0.967350 to
  0.967370. This small difference should not be treated as strong evidence of a
  leaderboard improvement.
* The selected blend used stage 3 alone, with a global cutoff of
  0.6926170587539673. No country-specific cutoff passed the minimum tuning-gain
  rule, so every country retained the global cutoff.
* Train-versus-test AUC did not increase with the new features: India
  0.671535 → 0.670474, US 0.719538 → 0.719243. This diagnostic passed its gate;
  it does not establish France accuracy or eliminate distribution shift.

France has no labeled holdout. No Amazon submission was made, and no new
Amazon score is available. The 0.980917 holdout candidate oracle is a theoretical
retrieval ceiling using true labels, **not** the model's score.

## Completed artifacts and verification

* Code evaluated: `fa2e529` on `r7-improvements`.
* CPU execution: six threads; the detached runner took 3 hours 24 minutes
  51 seconds, excluding earlier interactive preparation, blocking and probes.
* All 32 correctness and synthetic pipeline tests passed before the full run.
* Official validator: PASS, with `--check-ids`, no warnings reported.
* Final matches: 5,649,320 pairs; both output TSVs contain 1,732,544 rows.
* Output sizes and SHA-256 hashes were rechecked after completion.

Local files (ignored by Git):

```text
output/r7_overnight/matching_results.tsv    upload candidate for Amazon
output/r7_overnight/candidate_pairs.tsv     exact final candidate set
work/r7_overnight/models/                  selected models and training metrics
work/r7_overnight/logs/validate.log         official validator output
work/r7_overnight/overnight.json            completed stage records and commands
```

The [aggregate JSON report](r7_overnight_result.json) contains precise metrics,
ablation trials, diagnostics, timing, and hashes of the output files and selected
models. Raw data, generated submissions and large models remain local.
