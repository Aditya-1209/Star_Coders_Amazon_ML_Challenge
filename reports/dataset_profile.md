# Dataset inspection

> Historical data-inspection report, written before model training. Completed model results and the current desktop workflow are described in the project-root README.md.

All six source files and the ground-truth file were scanned completely, using a streaming TSV reader.

| Split | Source | Rows | Country counts |
| --- | --- | ---: | --- |
| train | S1 | 2,206,821 | India: 883,188; US: 1,323,633 |
| train | S2 | 5,034,616 | India: 2,017,799; US: 3,016,817 |
| train | S3 | 5,285,603 | India: 2,115,547; US: 3,170,056 |
| test | S1 | 1,732,544 | France: 259,452; India: 809,986; US: 663,106 |
| test | S2 | 4,887,273 | France: 703,378; India: 2,312,565; US: 1,871,330 |
| test | S3 | 5,082,316 | France: 731,615; India: 2,405,000; US: 1,945,701 |

## Training labels

- Ground-truth rows: 2,206,821.
- Labeled matching pairs: 7,638,365.
- Source 1 businesses with no matches: 123,247 (5.58%).
- Mean matching records per Source 1 business: 3.461.
- Ground-truth and training Source 1 row counts agree: True.
- Duplicate IDs within ground-truth lists: 0 lists.
- Invalid ground-truth source prefixes: 0.

| Matches per Source 1 | Businesses |
| ---: | ---: |
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

## Data quality

- `train/train_source1.tsv`: blank fields: entity_id=0, business_name=0, business_address=0, country=0; wrong source prefixes: 0.
- `train/train_source2.tsv`: blank fields: entity_id=0, business_name=0, business_address=168,967, country=0; wrong source prefixes: 0.
- `train/train_source3.tsv`: blank fields: entity_id=0, business_name=0, business_address=175,916, country=0; wrong source prefixes: 0.
- `test/test_source1.tsv`: blank fields: entity_id=0, business_name=0, business_address=0, country=0; wrong source prefixes: 0.
- `test/test_source2.tsv`: blank fields: entity_id=0, business_name=0, business_address=129,408, country=0; wrong source prefixes: 0.
- `test/test_source3.tsv`: blank fields: entity_id=0, business_name=0, business_address=136,098, country=0; wrong source prefixes: 0.

## Sampled true-match difficulty

A reservoir sample of 2,000 training Source 1 rows (seed 42) contains 6,971 true matching pairs. The percentages below describe this sample, not the full population.

- same_country: 6,971 / 6,971 (100.00%).
- exact_business_name: 339 / 6,971 (4.86%).
- normalized_exact_business_name: 1,516 / 6,971 (21.75%).
- exact_business_address: 163 / 6,971 (2.34%).
- normalized_exact_business_address: 589 / 6,971 (8.45%).

## Compute implications

- Unrestricted training comparisons: 22,774,876,013,799.
- Unrestricted test comparisons: 17,272,751,604,416.
- Build a retrieval/blocking index and score only candidate pairs; do not materialize the Cartesian product.
- Keep raw TSVs on disk and process them in chunks; start experiments on a representative development subset.
- RAM and GPU needs for training remain to be benchmarked on the chosen pipeline.

## Next experiment

1. Create a seeded split by Source 1 business, keeping its labeled matches together. Check for shared target IDs before treating groups as independent.
2. Build candidate retrieval from names and addresses, and measure candidate recall before training the matcher. Evaluate with realistic unrelated records in the candidate pool.
   Preserve Unicode text and assess alternate-script names, initials, and missing addresses explicitly; exact-name retrieval alone cannot cover the observed noise.
3. Train a pair classifier with string-similarity features and hard negatives from retrieval.
4. Tune the match threshold against macro F0.5 per Source 1, including singletons and true matches missed during retrieval.
5. Evaluate generalization across US and India; preserve unknown country labels so France is included at inference.
6. Save the exact candidates scored by the model. Generate both required TSV outputs and validate them before submission.
   The supplied validator skips ID-existence checks by default (`--check-ids` enables them), and reports matches outside the candidate set only as warnings. Audit both conditions before submitting.

## Limitations

- Full ID uniqueness and full ground-truth referential integrity were not checked.
- True-pair text similarity statistics use a seeded reservoir sample of Source 1 rows.
- No model has been trained; no validation or test performance is claimed.
