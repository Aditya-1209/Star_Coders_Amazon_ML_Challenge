"""Run ON THE r12 VM (repo on branch r12, same venv). Exports r12's final pair scores so r14 can
ensemble them. Read-only on work/r12 (safe next to R15/R16 runs that verify parent artifacts): writes
~/r12_export/{r12_fold3,r12_fold4,r12_test}.parquet with sidx, tidx, score, sid, tid
(the exact scores r12's selected stacker thresholded), plus copies of selection/metrics.

    cd ~/Star_Coders_Amazon_ML_Challenge
    PYTHONPATH=code/business_entity_resolution/src .venv-r12/bin/python ~/export_r12_scores.py
    tar -cf ~/r12_export.tar -C ~ r12_export
"""
import argparse
from pathlib import Path

import polars as pl

from er_v2.r10 import chosen_scores

args = argparse.Namespace(work=Path("work/r12"), device="cuda", threads=16, batch_rows=50000)
out = Path.home() / "r12_export"
out.mkdir(exist_ok=True)
import shutil
for f in ("selection.json", "metrics.json"):
    shutil.copy2(args.work / f, out / f)
for name, split, fold in (("r12_fold3", "train", 3), ("r12_fold4", "train", 4), ("r12_test", "test", None)):
    selection, scores = chosen_scores(args, split, fold)
    norm = args.work / "norm"
    s1 = pl.read_parquet(norm / f"{split}_source1.parquet", columns=["entity_id"])["entity_id"]
    tg = pl.concat([pl.read_parquet(norm / f"{split}_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
    scores.select("sidx", "tidx", "score").with_columns(
        sid=s1.gather(scores["sidx"]), tid=tg.gather(scores["tidx"])).write_parquet(out / f"{name}.parquet")
    print(name, selection["selected"], len(scores), flush=True)
