# R12 France follow-up review — 27 September 2026

Built from remote `r12` at `7ba5312`. No full-data training, new submission
inference, cloud launch or accuracy experiment was run in this review.

The published run took about 12.5 hours on g2-standard-32 (32 vCPU, 128 GB,
L4 24 GB). Final local macro F0.5 was 0.990387533: India 0.991566657 and US
0.989603450. The user's France 92% remains user-reported; the repository has
no France truth. None of the new code establishes a new France/Amazon score.

Read-only audit of the supplied R12 submission, joined by entity ID to the
organizer records cached locally:

| Country | Test businesses | Empty submitted match lists | Blank Source 2 / Source 3 addresses |
| --- | ---: | ---: | ---: |
| France | 259,452 | 14,632 | 21,537 / 21,541 |
| India | 809,986 | 46,757 | 52,764 / 59,240 |
| US | 663,106 | 38,398 | 55,107 / 55,317 |

All Source 1 addresses are present. Similar empty-output/missing-field rates
do not identify the cause of France errors. Among France Source 1 names,
40,789 contain non-ASCII characters. The earlier R7 learned transliteration
map would rewrite unrelated tokens in 124 of these names. The new script
guard fixes this bug; the count is from that local dictionary, not a measurement
of R12's dictionary or the number of wrong matches it caused.

This experiment adds a distinct text/domain-transfer signal rather than another
sibling/no-match variant. Later repository handoffs report that the R15 and
R16-fast final-layer variants retained R12. Their failure is a reason to keep
a frozen reference and an improvement gate, not proof that this variant works.

The implementation and one-command launch are documented in
[the France guide](../docs/README_r12_france.md). It reuses immutable R12 graph
features, neural scores and models; trains two small final trees plus a bounded
domain classifier; preserves India/US test decisions; and defaults to a
3.5-hour process deadline. Cache/model/schema/data/version checks reject
incompatible artifacts. Runtime on the L4 remains unmeasured. A missing cache
requires a different time budget; the historical full run did not fit four hours.

Validation completed here:

- 24 offline regression tests passed: 11 new France-path checks plus existing
  decision/feature/metric tests. No real model optimization in this test set.
- Tests exercise the Indic-script guard, raw-text features, empty-field
  behavior, Source 3 global IDs, France-only preparation, density-weight bounds,
  optimizer fold isolation, selection-only fold 3 access, tie fallback, cache
  tamper rejection and actual TSV generation for both decisions.
- Synthetic fallback output is byte-identical to its reference; a synthetic
  France update changes no India/US row and preserves empty business rows.
- Python AST parsing, shell syntax, dry command plan and whitespace checks pass.

Not performed: full-suite neural smoke test, L4 execution, memory/throughput
benchmark, France-labelled evaluation or challenge submission. Fold 4 is
report-only after a saved selection; the previously used partitions are not
pristine. The selected deployed local score will remain R12 because there are
no France training rows; the new proposal's local result is labelled a proxy.
