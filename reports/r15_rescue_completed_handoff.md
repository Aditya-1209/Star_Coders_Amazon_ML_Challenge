# Completed R15 address-rescue handoff

Recorded from the user's VM update on 27 September 2026. The raw rescue
artifacts have not been downloaded into this workspace, so the values below
are reported results, not an independent rerun or artifact verification.

## Outcome

Commit `78ee70708c36ca33527966c6dc7038da5220a50d`, branch
`codex/r15-address-rescue`, completed with official validation PASS and
**selected the existing R12 model**. All 72 tests passed on Linux in 144.035
seconds. Audit started at 12:53:31 UTC; validation started at 13:02:14 UTC
(8 minutes 43 seconds later). Completion was confirmed after validation;
the exact completion timestamp was not supplied.

The selected 3A proposal was the business-balanced missing-address specialist,
blended at weight 0.25, with threshold 0.7190439701080322 and no country
thresholds. Its 3A macro F0.5 was 0.9904601765766562 versus R12's
0.9904225628407861. Increasing specialist weight generally reduced the 3A
gain; pure specialist scores were worse than R12.

The 3B gain was **0.000012225370436292927**, below the required **0.0005**.
The approximate one-sided 95% lower bound was **-0.00001267394856654612**.
The country guard passed; the overall improvement gate failed.

| 3B metric | US proposal | US R12 | India proposal | India R12 |
| --- | ---: | ---: | ---: | ---: |
| Macro F0.5 | 0.9898612316162939 | 0.9898284028992123 | 0.9916617081444551 | 0.991680468268499 |
| Pair precision | 0.99868439894 | 0.99864865470 | Slight decrease reported | — |
| Pair recall | 0.97230064129 | 0.97230499791 | Slight decrease reported | — |

The experiment did not establish the intended recall recovery. This does not
establish that all donor-based approaches are ineffective.

The reported fold-4 score **0.990388 evaluates retained R12**, not the rejected
specialist proposal. The output has 1,732,544 Source 1 rows, including 99,787
empty rows, matching the reported R12 counts. Counts alone are not a byte-level
equality check. Do not resubmit this fallback as a new model improvement.

## Three related proposals retained R12

These are related experiments against the same baseline and previously
examined validation partitions, not three independent confirmations.
All retained the unchanged 0.0005 improvement gate.

| Proposal | 3B gain in F0.5 units | Approximate one-sided 95% lower bound | Selected |
| --- | ---: | ---: | --- |
| R15 cross-source final layer | 0.0000494283 | -0.0000383699 | R12 |
| R16-fast presence/blend | 0.00003255008649915252 | -0.000042150524791609645 | R12 |
| R15 missing-address specialist | 0.000012225370436292927 | -0.00001267394856654612 | R12 |

The first row uses the precision provided in the earlier handoff. No new
public leaderboard score follows from these runs. Keep the strongest measured
public submission while collecting complementary evidence.

## Evidence to preserve and inspect next

Preserve the actual rescue `selection.json`, `metrics.json`, `baseline_audit.json`,
`run.json`, `result.json`, `result.md` and `logs/`, plus the checkout commit and
diff. Keep the original R12 frozen selection, model metadata and submission.
The expected rescue work directory from its launch guide is
`~/Star_Coders_r15_rescue/work/r15_rescue`; confirm the actual location before
copying. This repository report preserves the supplied summary, not those files.

Before another donor experiment, inspect on 3A:

- Coverage of missing-address candidates and true matches by a confident
  opposite-source donor, including ambiguous or absent donors.
- Corrected and newly introduced false positives/false negatives compared
  with frozen R12, including addressed records changed by exclusive ownership.
- Errors by donor confidence, numeric/name agreement and country. Count
  businesses as well as pairs; do not infer France performance from US/India.

Prioritize completed R13/R14 predictions with complementary errors. A remote
branch check after this update found R13 still at `bfb2ef3`, with no completed
R13 scores available locally, and no R14 branch. Do not spend another full
training run merely to produce a differently named output.

The user subsequently confirmed that an R12 export archive was downloaded
and sent to the R14 teammate. Verify that existing archive before requesting
anything more; it has not yet been made accessible in this workspace.
Use [the R12 export checks](../docs/README_r12_evidence_export.md) to verify
final fused scores and entity/source mappings. Align comparison scores on
entity IDs and the original R12 business partitions; audit their training
provenance before combining them. Preserve the union of candidates and mark
missing scores explicitly. Freeze one proposal on 3A, apply the unchanged
gate on 3B, and report fold 4 only after selection. These partitions have
already been examined in previous work; do not describe them as untouched.
