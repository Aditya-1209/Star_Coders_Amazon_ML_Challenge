# R10: neural retrieval, cross-encoder and graph matching

R10 is based on `r8-neural` at `ad75b66`, in its own `codex/r10` branch.
R9 is unchanged. Target machine: **i9-13900K, 32 GB RAM, RTX 3060 12 GB**.

**Status: implementation and CPU smoke testing only. No full R10 training or
RTX 3060 benchmark has been completed.** The user-reported R8 reference is
95.6 local / 95.4 online. The older R8 guide describes a different experiment
at 96.94 local; those numbers are not interchangeable. **97.5 is the target,
not a measured result or guarantee.**

## What changes

1. **Scalable neural retrieval.** Country-scoped FAISS IVF-SQ8 retrieves a
   shortlist of 96 neighbors; exact cosine reranking keeps 24. Small datasets
   use exact search. A sample of 32 queries per country is compared with exact
   search, and the stage stops below 95% ANN recall@24. This measures approximate
   search quality, not entity-match recall. Lexical and phonetic candidates are
   retained in the union.
2. **Neural rescue.** The top eight neural candidates survive stage-1 lexical
   pruning even when their string features are poor. Training and inference use
   the same rule. The final matcher still has to accept the pair.
3. **A trained cross-encoder.** A multilingual E5-small backbone jointly reads
   both records. Separate name/address token budgets prevent long addresses
   from consuming the names' entire context. It learns from positive matches,
   lexical hard negatives, ANN hard negatives and random negatives. Empty-match
   businesses participate. Occasional address removal and record-order swaps
   augment only supplied training text.
4. **CE + graph fusion.** Two XGBoost variants combine cross-encoder logits,
   relative score features, bi-encoder cosine, string features and graph support.
   One uses equal total training weight per business to better reflect the
   macro objective. Validation chooses the variant and blend with the reference.
5. **Memory and reproducibility.** Embeddings and token IDs use memory-mapped
   arrays. Encoding, ANN construction, pair scoring and feature generation are
   chunked. One heavy process runs at a time. Fresh encoder training chooses one
   positive per business and excludes positive targets from its hard-negative
   pool to avoid contradictory in-batch contrastive labels.

The shared R8 files change only where R10 needs candidate rescue, memory-mapped
cosines, distinct lexical output names and safer contrastive sampling. Existing
commands default to the old lexical pruning rule (`--neural-rescue-k 0`).

## Set up on the desktop

Use Python 3.12 and a separate checkout/environment. WSL2 Ubuntu or Linux is the
recommended full-run environment. The Python runner has no Bash dependency and
uses portable file locks; native Windows still depends on availability of the
pinned FAISS/PyTorch wheels and has not been tested in this task.

Keep data and work directories on an SSD. **Budget 120 GB free for a fresh run**,
including normalized records, expanded features, embeddings, token caches,
models and outputs. This is a planning allowance, not measured peak disk usage.
The runner stops before a stage if less than 12 GiB remains; a stage itself can
use more than the reserve. On WSL, leave RAM for Windows and keep the files in
the Linux filesystem. Avoid other large GPU jobs during this run.

From the R10 checkout:

```bash
python3.12 -m venv .venv-r10
source .venv-r10/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/business_entity_resolution/requirements_r10.txt
```

Install a **CUDA-enabled wheel matching the pinned PyTorch version and your
NVIDIA driver**, using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if the default package does not expose CUDA. Do not blindly reuse the older
R8 `cu126` wheel command with a different PyTorch version. The preflight below
actually runs a tiny XGBoost CUDA fit as well as checking PyTorch CUDA.

Supply the organizer's seven TSVs under `student_resource/dataset/{train,test}`
and the organizer validator at `student_resource/utils/validate_submission.py`.
No business lookup service or external entity data is used. The pretrained
backbone is [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT, approximately 118M parameters), pinned at revision
`fd1525a9fd15316a2d503bf26ab031a61d056e98`. Its first use downloads model weights.

```bash
export PYTHONPATH=code/business_entity_resolution/src
export POLARS_MAX_THREADS=12 OMP_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false
python -m unittest discover -s tests_v2 -v
python scripts/run_r10.py --preflight
python scripts/run_r10.py --plan
python -u scripts/run_r10.py --max-hours 24
```

`--plan` prints commands without loading models, accessing the GPU, or starting
training. A 24-hour limit is a stop limit, **not an estimated duration**. There
is no measured full-run ETA yet. The runner writes `work/r10/run.json` and one
log per stage under `work/r10/logs/`. The full pipeline includes fresh
normalization, lexical blocking, encoder training, embeddings, ANN retrieval,
features, stage 2, graph reference, CE training/scoring, fusion and validation.

To reuse a compatible R8 encoder without modifying it:

```bash
python -u scripts/run_r10.py --encoder /absolute/path/to/r8/work/neural_e5 --max-hours 24
```

The saved `finetune.json` must say encoder folds **0/1/8/9 only**. Use weights
from the same organizer dataset and unchanged Source 1 ordering. The runner
checks the declared folds and fingerprints the files, but cannot reconstruct
the historical training data from an R8 model. R10 always regenerates its own
embeddings, candidates, token caches and features; it does not overwrite R8/R9
work or import arbitrary old prediction caches.

## Desktop defaults

| Setting | Default | Purpose |
|---|---:|---|
| CPU workers / ANN threads | 12 | Bound CPU contention and worker memory |
| Encoder training batch | 64 | In-batch positives plus hard negatives |
| Embedding batch / length | 256 / 64 | Bounded CUDA inference batches |
| CE physical batch / accumulation | 16 / 4 | Effective 64 pairs per optimizer step |
| CE scoring batch / pair length | 128 / 128 | Bounded pair inference |
| CE training businesses | 180,000 | Maximum sampled businesses; up to 4 positives plus negatives each |
| CE validation businesses | 15,000 | Fold-9 early stopping |
| CE epochs | Up to 3 | Stops after one non-improving epoch |
| ANN nlist / nprobe | 4,096 / 96 | SQ8 country index; smaller for small countries |
| ANN query batch / shortlist / retained | 128 / 96 / 24 | Bounded exact reranking |
| Feature shard / dense matrix batch | 150,000 / 50,000 | Avoid full dense feature copies |
| Graph support batch | 1,000 businesses | Limit intermediate support joins |

CUDA CE training uses FP16 autocast, gradient scaling, gradient clipping and
activation checkpointing. Large arrays remain on disk. These settings are
chosen for 12 GB VRAM and 32 GB RAM; **peak use and throughput are not yet
measured on that hardware**. FAISS uses the CPU while the encoders and XGBoost
use CUDA. See [FAISS index storage](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
and [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).

After an interruption, rerun the **identical command** with `--resume`:

```bash
python -u scripts/run_r10.py --max-hours 24 --resume
```

If using `--encoder` or custom paths/settings, include those again. Completed
stages are verified and skipped. The interrupted stage starts again: there is
no mid-epoch optimizer resume. The deadline does not shut down your computer.
After code, data, dependency or batch-size changes, use a **new** `--work` and
`--output`; resume intentionally rejects changed settings.

If CUDA runs out of memory, start a fresh run with `--ce-batch 8
--ce-score-batch 64 --encode-batch 128`. If the ANN audit fails, start a fresh run
with `--nprobe 192 --search-k 192`. Avoid lowering the ANN audit threshold merely
to make the run finish.

## Selection and evidence

| Layer | Fitting labels | Stopping / selection |
|---|---|---|
| Bi-encoder | 0/1/8/9 | Fixed epoch budget |
| Stage-1 ensemble | 0/8 and 1/9 | Fold 3, complementary scores on training groups |
| Stage 2 | 2/5 | Fold 3 |
| Graph reference | 6/7 | Fold 3 |
| Cross-encoder | 0/1/8 | Fold 9 |
| CE + graph fusion | 6/7 | Fold 3A |
| Blend / cutoff | None | Fold 3A; one acceptance comparison on 3B |
| Final report | None | Fold 4, after choice is frozen |

The gate requires at least 0.0005 absolute macro-F0.5 gain, a positive approximate
one-sided 95% lower bound over paired business differences, and no seen-country
regression greater than 0.002. If it fails, the runner uses the **rebuilt
R8-style graph reference**. That reference uses R10's candidates, rescue and
encoder: it is not the exact old 95.4 leaderboard submission. Keep the old
submission for comparison.

Fold 3B is independent only of the new final layer; inherited earlier layers
use all of fold 3. Fold 4 was already inspected during historical experiments.
The normal-approximation gate is a screening measure, not proof of leaderboard
improvement. France has no labels; its score cannot be measured locally, and
it uses the global cutoff. R10's field truncation, ANN approximation and new
neural signal all need real-data evaluation before claiming an improvement.

When the run completes:

- `work/r10/result.md` and `result.json`: selected model, measured local result,
  explicit 97.5 target status, stage timings and output hashes.
- `work/r10/metrics.json`: macro F0.5, pair precision/recall, per-country scores,
  and the candidate oracle ceiling. This reports actual measurements; the
  Amazon score remains null until an actual submission is scored.
- `work/r10/selection.json`: every 3A trial and the single 3B gate result.
- `work/r10/ann_{train,test}.json`: ANN/exact overlap audit per country.
- `work/r10/ce_model/training.json`: folds, training settings and loss history.
- `output/r10/matching_results.tsv`: final selected matches for the portal.
- `output/r10/candidate_pairs.tsv`: final graph candidates actually scored.
- `output/r10/reference/`: rebuilt reference output for comparison.

Only `result.json` with `status: complete` and `official_validation: PASS`
means the full run finished. The official validator checks format and IDs,
not model accuracy. Nothing is automatically submitted to Amazon.
