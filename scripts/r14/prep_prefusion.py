"""Package R14 PRE-fusion component predictions for the r16 fast CE fusion (codex/r16-fast-ce-fusion).

Components (all existing files, nothing retrained):
  graph         r10 desktop stage-3 graph score (probability); fit folds 6/7, tuned on 3
                D:/star_r10b/work/{eval_preds_stage3 (folds 3/4), test_preds_stage3}.parquet
  ce_a          CE-A logits (e5-small cross-encoder, 250k businesses of folds 0/1/8, early stop fold 9)
  ce_a_swapped  CE-A logits with (target, source) pair order at scoring time
  ce_b          CE-B logits (e5-small, 500k businesses of folds 0/1/8, seed 2027, early stop fold 9)
CE scoring covered only pairs with graph score in (0.001, 0.999); other pairs have no CE row.
Files are keyed by entity IDs (sid, tid), so no index alignment is needed downstream.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import polars as pl

from er_v2.folds import fold_expr

W = Path("D:/star_r10b/work")
R14 = Path("D:/star_r14")
OUT = R14 / "r14_prefusion"
OUT.mkdir(exist_ok=True)
N = W / "norm"


def ids(split):
    s1 = pl.read_parquet(N / f"{split}_source1.parquet", columns=["entity_id"])["entity_id"]
    tg = pl.concat([pl.read_parquet(N / f"{split}_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
    return s1, tg


def keyed(frame, s1, tg, col):
    return frame.select(sid=s1.gather(frame["sidx"]), tid=tg.gather(frame["tidx"]), **{col: pl.col(col)})


def write(frame, name):
    path = OUT / name
    frame.write_parquet(path, compression="zstd")
    return name


files = {c: {} for c in ("graph", "ce_a", "ce_a_swapped", "ce_b")}
rows = {}
ce_src = {"ce_a": "ce_{}_uncertain.parquet", "ce_a_swapped": "ce_{}_uncertain_swap.parquet", "ce_b": "ce_{}_uncertain_b.parquet"}
s1, tg = ids("train")
graph = pl.read_parquet(W / "eval_preds_stage3.parquet", columns=["sidx", "tidx", "fold", "score"])
ce_train = {c: pl.read_parquet(R14 / p.format("train")).with_columns(fold_expr()) for c, p in ce_src.items()}
for fold in (3, 4):
    g = graph.filter(pl.col("fold") == fold)
    files["graph"][f"fold{fold}"] = write(keyed(g, s1, tg, "score"), f"graph_fold{fold}.parquet")
    rows[f"graph_fold{fold}"] = len(g)
    for c, frame in ce_train.items():
        part = frame.filter(pl.col("fold") == fold)
        files[c][f"fold{fold}"] = write(keyed(part, s1, tg, "ce_logit"), f"{c}_fold{fold}.parquet")
        rows[f"{c}_fold{fold}"] = len(part)
pl.DataFrame({"entity_id": s1}).write_parquet(OUT / "train_source1_ids.parquet", compression="zstd")
s1t, tgt = ids("test")
gt = pl.read_parquet(W / "test_preds_stage3.parquet", columns=["sidx", "tidx", "score"])
files["graph"]["test"] = write(keyed(gt, s1t, tgt, "score"), "graph_test.parquet")
rows["graph_test"] = len(gt)
for c, p in ce_src.items():
    part = pl.read_parquet(R14 / p.format("test"))
    files[c]["test"] = write(keyed(part, s1t, tgt, "ce_logit"), f"{c}_test.parquet")
    rows[f"{c}_test"] = len(part)

commit = subprocess.run(["git", "-C", "C:/Users/ashma/SC_r14", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ce_a_meta = json.loads((W / "ce_model/training.json").read_text())
meta = {
    "model_id": f"R14 desktop run (branch r14 @ {commit}); graph = r10 desktop stage 3 in D:/star_r10b/work; "
                "CE-A = D:/star_r10b/work/ce_model; CE-B = cloud L4 run r14b (scripts/r14/cloud_ce.py)",
    "score_stage": "pre_fusion",
    "producer_note": ("Existing pre-fusion predictions copied from the producing runs, keyed by entity IDs "
                      "(sid = Source 1 entity_id, tid = Source 2/3 entity_id). graph.score = stage-3 graph probability "
                      "(eval_preds_stage3 'score' for folds 3/4, test_preds_stage3 'score'); graph model fit on folds 6/7, "
                      "threshold/blend tuned on fold 3. CE files hold raw logits (ce_logit) from e5-small cross-encoders "
                      "trained on businesses of folds 0/1/8 with fold 9 as validation (1 epoch each): CE-A 250k businesses "
                      f"(fold9 loss {ce_a_meta['history'][-1]['fold9_loss']:.4f}), CE-B 500k businesses seed 2027 (fold9 loss 0.0445). "
                      "ce_a_swapped = CE-A weights scored with (target, source) order. CE scored only graph pairs with "
                      "0.001 < graph score < 0.999; pairs outside that band have no CE row (missing, not negative). "
                      "The final cross-fitted fusion_ab outputs are NOT included."),
    "train_source1_ids": "train_source1_ids.parquet",
    "components": {
        "graph": {"fit_folds": [6, 7], "selection_folds": [3], "score_kind": "probability", "score_column": "score",
                  "files": files["graph"]},
        **{c: {"fit_folds": [0, 1, 8], "selection_folds": [9], "score_kind": "logit", "score_column": "ce_logit",
               "files": files[c]} for c in ("ce_a", "ce_a_swapped", "ce_b")},
    },
    "row_counts": rows,
}
(OUT / "inputs.json").write_text(json.dumps(meta, indent=2))
sums = []
for p in sorted(OUT.glob("*")):
    if p.name == "SHA256SUMS":
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    sums.append(f"{h}  {p.name}")
(OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
print(json.dumps(rows, indent=1))
print("\n".join(sums))
