# Parallel CPU inference benchmark

Hardware: Apple M4, 10 CPU cores, 16 GB RAM. GPU not used.

The test index contains **9,969,589** Source 2/3 records: India 4,717,565;
US 3,817,031; France 1,434,993. Building it took 506.4 seconds.

## Correctness

- All 904 full-corpus holdout rows produced exactly the same matching and
  candidate lists as the serial reference, including empty rows.
- A seeded reservoir sample of 768 test businesses contained 341 India,
  288 US, and 139 France records. Outputs were byte-for-byte identical across
  runs with 4, 6, and 8 workers.
- All 13 tests passed. They cover retrieval, scoring, grouped splits,
  interrupted-run recovery, rejection of altered inputs and corrupt checkpoints,
  serial equivalence, empty rows, and submission validation failures.
- Strict streaming validation passed on all 768 test-sample rows and 35,352
  candidate pairs, checking existence against all 9,969,589 raw test target IDs.
  Output hashes and pair counts agreed with the inference completion receipt.

## Throughput

Each worker uses one CatBoost/BLAS thread and its own read-only SQLite connection.
Read-only memory mappings allow OS index pages to be shared. Each benchmark used
32 anchors per chunk, with at most twice the worker count of chunks in flight.

| Workers | Test anchors | Seconds | Anchors/second | Linear full-test projection |
| --- | ---: | ---: | ---: | ---: |
| 4 | 768 | 13.436 | 57.158 | 8.42 hours |
| 6 | 768 | 9.090 | 84.488 | 5.70 hours |
| 8 | 768 | 8.253 | 93.057 | 5.17 hours |

These short runs were sequential; later runs benefited from warmed OS caches.
The projections are estimates, not measured full-run durations. Thermal state,
other applications, record distribution, and term caching can change throughput.
Per-worker RSS includes shared mapped pages and must not be added as if all
pages were private allocations. Full measurements are in
`parallel_inference_benchmark.json`.

## Full run

The production run uses 8 workers and 128 anchors per checkpoint. It retains the
frozen full-corpus model and threshold 0.605. Atomic chunk files allow an
interrupted run to resume; completed files merge in original Source 1 order.

`scripts/run_full_inference.py` automatically runs strict streaming validation
after prediction. Validation checks every required row, every target ID against
the raw test files, duplicate IDs, and the match/candidate subset relationship.
It holds source ID sets in memory, never all candidate mappings. Successful
completion is recorded in `artifacts/full_inference_status.json` and
`output/validation.json`; file existence alone is not evidence of completion.

The official validator was previously passed for the 904-row holdout benchmark.
The full test job uses our stricter streaming validator to avoid retaining the
full candidate mapping in RAM. Neither formatting validation nor the sampled
holdout score establishes test-set accuracy, particularly for unseen France.
