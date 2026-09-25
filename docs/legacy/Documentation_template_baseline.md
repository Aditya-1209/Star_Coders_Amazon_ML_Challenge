# ML Challenge 2026: Business Entity Resolution

**Team name:** Star Coders

**Team members:** Adyanth Mallur, Aditya Patil, Akshay Gudur, Advika Raj

**Prepared:** 25 September 2026

**Approach:** Country-aware lexical blocking followed by a CatBoost pair classifier

## 1. Executive summary

We index the supplied target records with SQLite FTS5 and retrieve candidates
through business-name, address, and name-trigram channels. A small CatBoost model
scores 31 string-similarity and retrieval features; its threshold is selected
using per-business macro F0.5, including businesses with no matches. All business
data comes from the challenge files, and the pipeline runs on CPU.

The full-corpus development holdout score is **0.925549 macro F0.5**, with
**97.17% pair precision**, **84.59% pair recall**, and **93.82% candidate recall**.
This is a 904-business training-data holdout result, not a test or leaderboard score.

## 2. Problem analysis and data preparation

Training contains 2,206,821 Source 1 businesses and 10,320,219 Source 2/3 records.
Test contains 1,732,544 Source 1 businesses and 9,969,589 Source 2/3 records.
Training covers India and the US; test additionally includes 259,452 French
Source 1 businesses. Countries are treated as open string labels rather than a
fixed set of learned categories.

A complete streaming data scan found 123,247 training singletons (5.58%) and
7,638,365 labeled pairs. Training target files contain 344,883 blank addresses;
test target files contain 265,506. In a seeded sample of 2,000 training businesses
with 6,971 positive pairs, only 4.86% of pairs had exactly equal names and 2.34%
had exactly equal addresses. Exact-string joining alone would miss most matches.

For supervised development, reservoir sampling with seed 42 selects 6,000
Source 1 businesses. Splits contain 4,198 training, 898 threshold-tuning, and
904 held-out businesses. Splitting groups anchors that share labeled target
records, and approximately stratifies by country and singleton status. Neither
anchors nor labeled positive target groups cross partitions.

The initial experiment used 220,838 targets, including all positives of the
sampled anchors and 200,000 background targets. The submitted model is trained
and evaluated with retrieval over **all 10,320,219 training targets**. Labels
are never stored in the search index. The target corpus is shared as unlabeled
retrieval data across partitions. The classifier is still fitted on the 4,198
training anchors, not on all 2.2 million labeled Source 1 businesses.

## 3. Candidate generation

Text normalization preserves Unicode letters, folds Latin accents, standardizes
case and punctuation, and expands a fixed set of address abbreviations. An
auxiliary name view removes common legal suffixes; the original normalized name
is retained as a separate comparison feature. No geocoding, registration lookup,
external entity database, or external business-data augmentation is used.

Country blocking selects an FTS5 index for each observed country. Three channels
search normalized names, addresses, and character trigrams of names. Queries
select up to 6 name terms, 7 address terms, and 10 trigrams using corpus document
frequencies. A posting budget of 3,000 restricts matching to rare query terms;
all selected query terms still contribute to BM25 ranking. If every available
term exceeds that budget, the two rarest terms are intersected. The budget was
chosen using timing and recall probes on training anchors.

Each channel returns at most 20 records. Their deduplicated union, at most 60
records per Source 1 business, is the exact set scored by the classifier and
written to `candidate_pairs.tsv`. No true labels are injected into evaluation
candidates. The full-corpus holdout averages 46.24 candidates per anchor.

The 93.82% holdout candidate recall measures the recall ceiling of this retrieval
stage; it does not guarantee retention of every match. A perfect classifier
limited to these candidates would reach 0.974992 macro F0.5 on this holdout.

## 4. Matching model and threshold

The model is CatBoostClassifier 1.2.10, trained from scratch on retrieved pairs.
The CatBoost library is Apache-2.0 licensed; no third-party pretrained model is
used. This is a 450-tree classifier of depth 6, far below the 8-billion-parameter
limit. The saved model file is 526,000 bytes.

The 31 numeric features cover normalized-name and suffix-stripped-name fuzzy
similarity, token sort/set/partial similarity, exact nonempty matches, token
Jaccard and containment, name length and initials, Latin-script fraction
difference, address similarity and missingness, number and postal-code
agreement/conflict, inverse per-channel retrieval ranks, and channel count.
IDs and country identities are not learned features.

Training uses binary Logloss, learning rate 0.075, L2 leaf regularization 5,
64 numeric borders, random seed 42, and CPU execution. Early stopping allows
50 rounds without tuning-Logloss improvement; this run retains all 450 trees.
Retrieved nonmatches provide the negative examples. No synthetic positives are
added to the evaluation candidate sets.

Threshold **0.605** maximizes macro F0.5 on the 898-business tuning partition.
For each business with true set T and prediction set P, the score is
`1.25 * |T intersection P| / (0.25 * |T| + |P|)`; if both sets are empty, the
score is 1. True matches missed during retrieval remain in the denominator.
The reported holdout was used neither for fitting nor threshold selection.
Classifier scores are not separately calibrated probabilities.

## 5. Results and error analysis

| Split | Businesses | Macro F0.5 | Pair precision | Pair recall | Candidate recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 4,198 | 0.924398 | 98.42% | 84.25% | 92.69% |
| Tune | 898 | 0.896169 | 97.53% | 80.79% | 91.43% |
| Holdout | 904 | 0.925549 | 97.17% | 84.59% | 93.82% |

Holdout contains 2,641 correctly predicted pairs, 77 false positives, and 481
false negatives. Of the missed positive pairs, 193 were absent from retrieval
and 288 were retrieved but rejected by the classifier. Singleton accuracy is
92.45% across 53 singleton businesses. Predicting empty lists for everyone
would score 0.058628 macro F0.5.

| Holdout country | Businesses | Macro F0.5 | Pair precision | Pair recall |
| --- | ---: | ---: | ---: | ---: |
| India | 363 | 0.884311 | 95.12% | 77.71% |
| US | 541 | 0.953219 | 98.37% | 89.06% |

Manual inspection of saved holdout errors found false merges between near-equal
names with conflicting street numbers, and ambiguous common names when the
target address was missing. Misses included shortened or completely changed
business names, missing addresses, and alternate-script names with too little
shared address text. These are observed examples, not an exhaustive frequency
breakdown of all error categories.

France has no supplied labels, so French accuracy is unknown. Preserving French
records and passing retrieval/output tests establishes functionality, not
predictive quality. The lower India recall also indicates that alternate-script
and severe-noise matching remain important limitations.

## 6. Test inference and output validation

Inference uses 8 independent CPU workers on an Apple M4 with 16 GB RAM. Each
worker scores batches with one CatBoost thread. SQLite connections are read-only;
file-backed memory mappings allow index pages to be shared through the OS.
Chunks of 128 anchors are saved atomically with input and output hashes. A
resume verifies completed chunks and rejects changed inputs, model, code, or
chunk size. Final TSVs merge in the original Source 1 order.

<!-- FULL_TEST_RESULTS -->
The initial Mac run was intentionally stopped before completion to move the
workload to a desktop. No completed full-test submission is claimed. Final
counts and validation evidence are inserted automatically into the packaged
write-up after a full run succeeds.
<!-- END_FULL_TEST_RESULTS -->

The streaming validator checks exact headers, all required Source 1 rows,
uniqueness, target prefixes and existence against raw test Source 2/3 files,
empty lists, and strict match/candidate subset membership. It retains source ID
sets instead of all candidate mappings. The organizer's validator passed on the
904-row full-corpus development holdout; full test validation uses the streaming
checker to keep memory bounded. No test-set accuracy or leaderboard result is
claimed.

Parallel inference exactly reproduced both serial exports for all 904 holdout
businesses. A 768-business test sample, including France, gave identical files
with 4, 6, and 8 workers. Its 35,352 candidate pairs passed streaming validation
against all 9,969,589 raw test target IDs. Tests cover metric handling, grouped
splits, retrieval ranking, Unicode, missing fields, recovery after interruption,
damaged-checkpoint rejection, and malformed submissions.

## 7. Conclusions and limitations

This baseline makes full-corpus entity matching feasible on a 16 GB CPU laptop
through selective lexical retrieval and inexpensive pair classification.
Precision is strong on the sampled full-corpus holdout, but missed candidates,
ambiguous names, missing addresses, and cross-script variation limit recall.
Larger supervised samples and transliteration or multilingual retrieval are
future experiments; they are not part of these submitted predictions.

## Appendix: code and reproducibility

`code/business_entity_resolution/src/er_baseline/` contains data preparation,
retrieval, features, training, inference, scoring, validation, and packaging.
The included README gives commands using only the supplied challenge data.
Pinned dependencies, unit tests, frozen model weights, feature configuration,
and development metrics accompany the source. The exact saved model can be
used to regenerate both output files; separate instructions reproduce training.

The archive excludes raw datasets, the original dataset ZIP, virtual environments,
search indexes, and intermediate chunk files. Only the two completed prediction
files and their validation receipt are included under `output/`.
