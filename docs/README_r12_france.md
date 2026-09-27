# R12 France update on the existing L4 VM

Use this **cached follow-up**, not the original 12.5-hour full-training recipe.
It targets the existing **g2-standard-32: 32 vCPUs, 128 GB RAM, one L4 24 GB**.
The earlier run confirms that hardware. Keep the completed `work/r12`, its
`.venv-r12`, and the original organizer data at their existing paths.

The job defaults to a **3.5-hour cap**, with a maximum allowed cap of 4 hours.
This is a runtime limit, **not a benchmark or guarantee of completion**. It
skips all encoder training, embedding, neighbour search, original feature
generation, graph expansion, cross-encoder training and CE scoring. Only the
new text features and three small tree fits run. A fresh full run does not
fit this budget based on the measured R12 timings.

## Run

Wait for other jobs on the VM to finish before updating this checkout. From
the original repository on the VM:

```bash
git switch r12
git pull --ff-only origin r12
bash scripts/vm_r12_france.sh "$PWD/student_resource"
```

If the data lives elsewhere, replace the last argument with its original
absolute `student_resource` path. The folder must contain `dataset` and
`utils/validate_submission.py`. No package installation or model download is
needed. The launcher uses the existing environment, checks CUDA/dependencies,
runs the small regression tests and starts a detached job.

For a separate checkout, point to the original caches and environment:

```bash
R12_BASE_WORK=/absolute/original/repo/work/r12 \
R12_PYTHON=/absolute/original/repo/.venv-r12/bin/python \
  bash scripts/vm_r12_france.sh /absolute/original/student_resource
```

If the old cache is missing, this shortcut cannot run. The submission gzip
on GitHub and `r12_export.tar` alone do **not** contain the required graph
features, CE scores, model weights and producer manifest. Do not start the
old full-training launcher expecting a four-hour result.

Monitor and resume:

```bash
tail -f work/r12_france/runner.log
cat work/r12_france/run.json
# Same code, data and settings; restarts only the interrupted stage:
bash scripts/vm_r12_france.sh "$PWD/student_resource" --resume
```

Outputs are separate: `output/r12_france/matching_results.tsv` and
`candidate_pairs.tsv`. Use them only after `work/r12_france/result.json` says
`status: complete` and `official_validation: PASS`. Check `selected` and
`changes`: **`r12` means the proposal failed its gate and the old answers were
retained**, not a new accuracy improvement. The launcher refuses a busy GPU;
the runner locks its workspace and the parent run, checks disk reserve, kills
the whole stage process group on timeout, and validates completed artifacts
before resume.

The job limit stops the process, **not VM billing**. Stop the VM after downloading
results. Keep or set a separate Compute Engine maximum runtime/STOP policy for
the four-hour spending window. This launcher does not change or extend existing
guest/cloud shutdown timers; a shorter existing timer can interrupt the job.

## What changed

1. **Fix Indic-map leakage into French names.** The original normalization
   applied its learned Indic token map to any non-ASCII name. It now checks
   for an Indic script. A read-only audit found 124 affected French Source 1
   names in the local R7 cache, including `mam → maa` and `je → jay`. This is
   a real bug, but its observed coverage cannot explain an eight-point gap.
   The new follow-up computes its own features from raw text, bypassing that
   map. It leaves the frozen R12 feature values unchanged for old-model scoring.
2. **Add transferable name/address evidence.** Accent folding, apostrophe word
   boundaries, legal-form/article removal, symmetric name coverage and trailing
   name comparison distinguish shared generic prefixes from actual agreement.
   Street abbreviations are normalized separately from names. Locality words,
   numbers, `bis`/`ter`, units and leading-zero postcodes remain. There is no
   US state-name substitution in these new features, and blank fields never
   count as matches. These are learned features, not hard accept/reject rules.
3. **Add bounded France-domain weighting.** One 64-round, depth-3 classifier
   distinguishes 250k sampled France pairs from equally many labelled-country
   fitting pairs using text similarities, field presence and source-side
   indicators. It never uses match labels, IDs, country names, fold 3 or fold 4.
   Its density odds are clipped to `[1/3, 3]` and combined with business-balanced
   weights. Two depth-7 final models compare plain and adapted weights, with
   at most 900 rounds and early stopping. The four CE/neural target rank/gap
   features are excluded because their graph populations differ across splits.
4. **Protect current India/US performance.** Test changes are confined to
   France. Exact pair-set equality is checked for every other country before
   output. Candidates and target ownership remain country-scoped. The official
   validator checks every output row and entity ID.

## Validation protocol and limits

Train final models on **6/7**, early-stop/select on **3A**, then test one frozen
proposal on **3B**. A small grid compares balanced/transfer models at weights
0.25, 0.5 and 1 against the actual saved R12 scores. A proposal must improve 3A
and pass the existing 3B gate: gain at least 0.0005 macro F0.5, positive paired
approximate one-sided lower bound, and no labelled-country regression over
0.002. Otherwise keep R12. There is no manual France cutoff or automatic
lowering of the match threshold based on the reported score.

Fold **4** is prepared and reported only after selection is saved. Earlier
models used fold 3 and previous experiments inspected these partitions, so
this is not a pristine or fully nested validation experiment.

France has no training labels. The reported 92% is user-supplied; it is not
measured by this update. `proposal_proxy_fold4` measures what the proposal
would do on India/US. `local_fold4` measures the deployed local decisions,
which stay R12 because this patch only changes France. Neither is a new France
score. Density weighting can help covariate shift but cannot establish that
the label relationship transfers to French. R12's candidate ceiling remains.

The new profile has **not** been trained or benchmarked on the VM by this
change. Small offline regression tests, static Python/shell checks and command
plan checks are the validation performed here. No new accuracy is claimed.
