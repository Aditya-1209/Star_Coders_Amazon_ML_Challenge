# R10 implementation verification — 26 September 2026

Based on r8-neural commit `ad75b66e255b9f16acf6302be5e71ffe869229ee`.
The original r9 checkout was left clean and unchanged.

**42 tests passed in 34.225 seconds** with Python 3.12 on macOS arm64,
PyTorch 2.14.0, Transformers 5.17.0, sentence-transformers 6.1.0,
FAISS CPU 1.15.1, XGBoost 3.4.1 and Polars 1.44.2.

The complete `tests_v2` suite included:

- Existing accuracy, r4, r5, r7 and graph pipeline regressions.
- Real, offline neural training and model serialization with a tiny local BERT.
- ANN candidate union, lexical rescue, embedding memory maps, stage 2,
  two-hop graph, cross-encoder scores, tree fusion, selection and inference.
- A real compressed IVF-SQ8 index checked against exact nearest neighbors.
- Keyed score joins, duplicate/missing-score failures, singleton-aware metrics,
  training/gate/holdout separation and frozen selection during evaluation.
- Successful resume without recomputing a completed stage, plus refusal to
  reuse modified stage artifacts.
- Both reference and R10 output branches checked by the organizer's validator,
  including synthetic France businesses and empty matches.

The CPU path avoids a macOS OpenMP interaction between XGBoost and PyTorch by
using NumPy for embedding cosine. Neural and ANN pipeline processes do not
import the XGBoost runtime just to calculate fold IDs. The Transformers 5
pair-token API change is covered by the real neural smoke test.

`git diff --check` and Python compilation passed. The runner's `--plan` was
also checked. Tests were run in a temporary source copy; every delivered source
file was byte-compared with that tested copy before packaging.

**Not measured:** full-dataset R10 macro F0.5, Amazon leaderboard result,
France accuracy, full pretrained E5 fine-tuning, CUDA execution, desktop peak
RAM/VRAM, full-run disk use or runtime. The tests establish implementation
behavior on synthetic data, not a competitive accuracy gain. The 97.5 target
remains unverified. Use `docs/README_r10.md` for the desktop experiment.
