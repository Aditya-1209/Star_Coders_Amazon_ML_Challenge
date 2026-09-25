# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Star Coders
**Team Members:** Adyanth Mallur, Aditya Patil, Akshay Gudur, Advika Raj
**Submission Date:** 25 September 2026

---

## 1. Executive Summary

We normalize every record (Unicode transliteration, OCR-typo repair, canonical
abbreviations), then generate candidates with **country-scoped, IDF-weighted
inverted-key blocking** over six key families. Each Source 1 business keeps its
top 64 candidates. A **two-stage gradient-boosted classifier** (XGBoost, trained
from scratch on the provided labels) scores each pair: stage 1 uses 43 pairwise
string and blocking features. Stage 2 adds 10 *context* features describing how the
pair ranks among its business's candidates and against **every other business
competing for the same Source 2/3 record**. This uses the fact that each target
record belongs to at most one business. On a held-out 10% of training
businesses (220k, singletons included), the pipeline scores **0.949 macro
F0.5** (99.1% pair precision, 89.5% pair recall).

---

## 2. Methodology

### 2.1 Problem Analysis

* **Scale:** train has 2.21M Source 1 and 10.32M Source 2/3 records; test has 1.73M
  and 9.97M. There are 7.64M labelled pairs and 5.6% singletons. The Cartesian
  product (about 10^13 pairs) is infeasible, so blocking is mandatory.
* **Exclusivity:** all 7,638,365 labelled target IDs are unique. Every Source 2/3
  record is matched to **at most one** Source 1 business, and we exploit this
  explicitly (Section 4).
* **Country:** in a labelled sample 100% of true pairs share the country label, so
  blocking keys are scoped by the country string (an open set, so France is
  handled with no special code).
* **Name noise:** legal-suffix changes (Pvt Ltd / Private Limited / LLP / SARL /
  SAS), duplicated or reordered words, character typos, digit-for-letter OCR
  swaps (`F0rt`, `J0nes`, `lnvestment`), website forms (`maurewilliamscolombier.com`),
  DBA names, and, for India, names written in **9 Indic scripts** (Devanagari,
  Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi).
* **Address noise:** component reordering, abbreviations (St/Street/Saint, R/Rue,
  AV/Avenue), missing components or entire addresses (about 3% blank), a literal
  `null`, state names vs codes, leading zeros (`AF-0684` vs `AF-684`), and
  house-number ranges.
* Exact-name agreement is under 5% of true pairs, so similarity must be fuzzy.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted pair classifier + global assignment
**Core Innovation:** stage-2 "competition" features that turn the at-most-one-owner
property of Source 2/3 records into learned evidence, with the entire
1.7M × 64-candidate test graph scored at once.

---

## 3. Candidate Generation (Blocking)

Every record is exploded into keys, each prefixed by type and country:

| key | content |
| --- | --- |
| `n` | each core-name token (legal words removed) |
| `p` | adjacent pairs of core-name tokens (sorted) |
| `c` | first 8 chars of the space-free core name (catches `payneenterprises.com`) |
| `a` | address tokens containing a digit (house/plot/PIN numbers) |
| `w` | alphabetic address tokens of 4 or more characters (street, locality, city) |
| `q` | a numeric address token joined with the following token (`3315_fremont`) |

* Keys whose document frequency among Source 2/3 exceeds a per-type cap
  (2,000 for `n`, `a` and `w`; 500 for `p`, `c` and `q`) are dropped as uninformative.
* Each shared key adds `log(N/df)` to a pair score. Keys are joined in Polars in
  chunks of 100k businesses, and the **top 64 targets per business** are kept.
* **Candidate pairs generated:** 110,104,366 for test (mean 63.6 per business,
  741 businesses with none); 139,984,627 for train.
* **Recall ceiling:** 92.93% of all 7.64M labelled training pairs are among the
  candidates (87.4% within the top 10). Full test blocking takes about 8 min.
* **How we avoid losing true matches:** there are six complementary key families,
  so a typo in one token, a transliterated name or a missing address still leaves
  others. OCR repair and transliteration run before keying. Rare-key weighting
  keeps specific tokens ahead of generic ones.

`candidate_pairs.tsv` contains exactly these 64 (or fewer) candidates, which are
all scored by the model; every match is a subset of them.

---

## 4. Matching Model

**Features used (53):**
- **Name features:** RapidFuzz ratio, token-sort, token-set, partial ratio and
  Jaro-Winkler on the core name; ratio and token-set on the full name; ratio and
  partial ratio on the space-free name; token Jaccard, containment both ways and
  intersection size; lengths and token count; a non-Latin-script flag for each
  side.
- **Address features:** ratio, token-sort, token-set and partial ratio; token
  Jaccard, containment and intersection; the same four on numeric tokens only;
  first-number equality; a number-conflict flag; address lengths (0 means missing).
- **Blocking features:** summed IDF score, number of shared keys, rank within the
  business, score relative to the business's best and the target's best, the
  target's rank among competing businesses, and candidate counts per business
  and per target.
- **Stage-2 context features (from stage-1 probability p1):** rank, max, sum and
  count above 0.5 across the business's candidates; rank of this business among
  all businesses claiming the target; max and sum over those claimants; best
  *competing* claimant's p1; and p1 relative to the business's best.

**Model type:** XGBoost gradient-boosted trees (Apache-2.0; hist on CUDA,
depth 10, eta 0.08, early stopping on log-loss). There is no pretrained model;
everything is trained from the challenge labels.

**Training protocol (folds are hashes of the Source 1 id):** stage 1 is trained
on fold 0 (validated on fold 1) and on fold 1 (validated on fold 0). Each model
predicts the other fold, so stage-2 inputs are out-of-fold. Stage 2 is trained
on fold 2, the threshold is tuned on fold 3, and fold 4 is an untouched holdout.

**Threshold selection method:** grid search of the stage-2 probability (0.2 to
0.95) maximizing per-business macro F0.5 on fold 3. The best value is **0.675**.
After thresholding, each target is kept only for its highest-scoring business.

---

## 5. Results & Error Analysis

Macro F0.5 over *all* Source 1 businesses in the fold, including singletons and
businesses whose matches were missed by blocking:

| fold (about 220k businesses) | macro F0.5 | pair precision | pair recall |
| --- | ---: | ---: | ---: |
| fold 3, stage 1 only | 0.9442 | 98.79% | 88.81% |
| fold 3, stage 2 (tuning) | 0.9491 | 99.10% | 89.51% |
| **fold 4, stage 2 (holdout)** | **0.9494** | **99.13%** | **89.51%** |
| fold 4, candidate oracle | 0.9703 | 100% | 92.91% |

* **F_0.5 Score (macro):** 0.9494 on the untouched holdout.
* **Test sanity:** 92.7% of test businesses receive at least one match
  (US 93.8%, India 91.8%, France 92.5%), with about 3.1–3.3 matches each. Train has
  3.46 true matches per business and 94.4% non-singletons, which is consistent
  with about 90% recall at 99% precision. France behaves like the labelled countries.
* **Common false positives (preliminary, from spot checks):** businesses with generic names ("Prime Industries",
  "Vision Clinic") at the same street, and chains or branches with the same name
  in the same city.
* **Common false negatives (preliminary):** the remaining 7% blocking misses (Indic-script names
  whose transliteration differs strongly from the Latin name *and* whose address
  is missing or very short; heavily abbreviated website names), plus correct
  candidates rejected for low name similarity when the address is missing.

---

## 6. Conclusion

Cheap, high-recall key blocking combined with a strong pairwise model gets most of
the way. The largest single gain came from treating the problem globally: the
stage-2 features, which see every business competing for a record, lifted macro
F0.5 by 0.5 points at higher precision. The pipeline runs end to end in about an
hour on one desktop (CPU for text, GPU for the model). The next gains are in
blocking recall (learned transliteration dictionary, phonetic keys) and in
clustering Source 2 / Source 3 records with each other.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains `src/er_v2/` (all source),
`models/v2/` (trained models, threshold, validation metrics), `README.md` (exact
commands) and `requirements.txt` (pinned). The entry points, run in order, are
`er_v2.prepare`, `er_v2.run_block`, `er_v2.run_features`, `er_v2.train` and
`er_v2.predict`. They produce `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. Both files pass the official validator with
`--check-ids`.

### B. Compute

i9-13900K, 32 GB RAM, RTX 3060 12 GB. The timings are: normalization 45 s
(all files, 28 processes); blocking 8 min for test and 11 min for train;
features 7 min for test and 11 min for train; two-stage training with scoring
of all train pairs 11 min; test inference 4 min.
