# Full-corpus retrieval timing probe

The index contains all 10,320,219 training Source 2/3 records. These measurements use the same 60 randomly selected training-partition anchors (seed 2026). Holdout labels were not used to select a setting.

| Search setting | Seconds / query | Candidate recall | Mean candidates |
| --- | ---: | ---: | ---: |
| Original unrestricted search | 0.6839 | 91.53% | 47.1 |
| Rare-term gate: 1,000 postings | 0.0525 | 90.40% | 42.3 |
| Rare-term gate: 3,000 postings | 0.0554 | 92.09% | 46.4 |
| Rare-term gate: 10,000 postings | 0.0662 | 92.09% | 46.8 |

Selected a 3,000-posting rare-term gate: it matched the 10,000 setting on this small recall sample while being faster. Terms omitted from the gate still contribute to BM25 ranking. The complete 6,000-anchor benchmark measures the resulting pipeline on larger train/tune/holdout partitions.

These timings are a small-sample estimate. They do not include classifier scoring, output writing, multiprocessing, or the unseen French test distribution. The full test run has not been executed.
