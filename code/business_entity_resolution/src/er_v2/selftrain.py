"""Self-training for countries without labels (France), plus its validation proxy.

The stage-3 model never saw French records. Its most confident French decisions
become pseudo-labels (score >= HI and the record's owner under exclusivity ->
positive; score <= LO -> negative) and the stage-3 model is retrained on the
labelled training rows plus those pseudo-labelled rows. Only test *inputs* and
our own predictions are used: no labels, no external data.

proxy mode : validates the idea on India with the leave-one-country-out models
             (train.py / stage3.py --exclude-country India). India rows of folds
             6/7 are treated as unlabelled and pseudo-labelled; India fold 4 is
             scored before and after, with the threshold frozen (as for France).
test mode  : applies it to the unseen test countries and writes new TSVs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from .decision import best_per_target, blend_scores
from .metrics import macro_f05
from .runtime import BATCH_ROWS, DEFAULT_THREADS, positive_int
from .stage3 import HOLD_FOLD, TRAIN_FOLDS, TUNE_FOLD
from .train import PARAMS, decide, fit, fold_expr, predict_frame

HI, LO = 0.97, 0.02
PSEUDO_WEIGHT = 1.0


def pseudo_labels(rows: pl.DataFrame, score: str = "score", hi: float = HI, lo: float = LO) -> pl.DataFrame:
    """Confident rows only; positives must also own their target (exclusivity)."""
    owners = best_per_target(rows.select("sidx", "tidx", score), score).select("sidx", "tidx")
    pos = rows.filter(pl.col(score) >= hi).join(owners, on=["sidx", "tidx"], how="semi")
    neg = rows.filter(pl.col(score) <= lo)
    return pl.concat([pos.with_columns(label=pl.lit(1, pl.Int8)),
                      neg.with_columns(label=pl.lit(0, pl.Int8))], how="diagonal_relaxed")


def retrain(train_rows: pl.DataFrame, pseudo: pl.DataFrame, valid: pl.DataFrame, feats: list[str],
            params: dict, rounds: int) -> xgb.Booster:
    rows = pl.concat([
        train_rows.select(*feats, "label").with_columns(w=pl.lit(1.0, pl.Float32)),
        pseudo.select(*feats, "label").with_columns(w=pl.lit(PSEUDO_WEIGHT, pl.Float32)),
    ], how="vertical_relaxed")
    return fit(rows, feats, valid, rounds, params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["proxy", "test"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--model-dir", required=True, help="stage-3 model dir (LOCO dir for proxy)")
    ap.add_argument("--output", default="output_selftrain")
    ap.add_argument("--country", default="India", help="proxy mode: the left-out country")
    ap.add_argument("--countries", nargs="*", default=None,
                    help="test mode: countries to adapt (default: test countries absent from train)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=positive_int, default=DEFAULT_THREADS)
    ap.add_argument("--rounds", type=positive_int, default=2000)
    args = ap.parse_args()
    work, mdir = Path(args.work), Path(args.model_dir)
    meta = json.loads((mdir / "stage3_metrics.json").read_text(encoding="utf-8"))
    feats, thr, weight = meta["features"], meta["threshold"], meta["stage3_weight"]
    params = {**PARAMS, "device": args.device, "nthread": args.threads}
    m3 = xgb.Booster(model_file=str(mdir / "stage3.json"))
    m3.set_param({"device": args.device})
    t = time.time()
    log = lambda m: print(f"[{time.time() - t:6.0f}s] {m}", flush=True)

    if args.mode == "proxy":
        c = args.country
        rows = pl.read_parquet(work / f"stage3_train_no{c}.parquet")
        s1 = pl.read_parquet(work / "norm" / "train_source1.parquet", columns=["idx", "country"])
        country = s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country")
        rows = rows.join(country, on="sidx", how="left").with_columns(fold_expr())
        rows = blend_scores(rows.with_columns(p3=pl.Series(predict_frame(m3, rows, feats, BATCH_ROWS))), weight)
        from .run_block import load_split
        from .run_features import ground_truth_pairs
        s1f, tg = load_split(work, "train")
        truth = ground_truth_pairs(Path("student_resource/dataset"), s1f, tg)
        del s1f, tg
        hold_a = country.with_columns(fold_expr()).filter((pl.col("fold") == HOLD_FOLD) & (pl.col("country") == c))
        hold = rows.filter((pl.col("fold") == HOLD_FOLD) & (pl.col("country") == c))
        before = macro_f05(decide(hold, thr, "score"), truth, hold_a["sidx"])
        unl = rows.filter(pl.col("fold").is_in(TRAIN_FOLDS) & (pl.col("country") == c)).drop("label")
        pseudo = pseudo_labels(unl)
        check = pseudo.join(truth.with_columns(true=pl.lit(1)), on=["sidx", "tidx"], how="left")
        pos_prec = check.filter(pl.col("label") == 1)["true"].fill_null(0).mean()
        neg_err = check.filter(pl.col("label") == 0)["true"].fill_null(0).mean()
        log(f"pseudo-labels on {c}: {pseudo['label'].sum():,} pos (precision {pos_prec:.4f}), "
            f"{(pseudo['label'] == 0).sum():,} neg (error {neg_err:.4f}) from {len(unl):,} rows")
        seen = rows.filter(pl.col("country") != c)
        m = retrain(seen.filter(pl.col("fold").is_in(TRAIN_FOLDS)), pseudo,
                    seen.filter(pl.col("fold") == TUNE_FOLD), feats, params, args.rounds)
        hold = blend_scores(hold.with_columns(p3=pl.Series(predict_frame(m, hold, feats, BATCH_ROWS))), weight)
        after = macro_f05(decide(hold, thr, "score"), truth, hold_a["sidx"])
        res = {"country": c, "before": before, "after": after, "pseudo_pos_precision": pos_prec,
               "pseudo_neg_error": neg_err, "hi": HI, "lo": LO}
        (mdir / f"selftrain_proxy_{c}.json").write_text(json.dumps(res, indent=2))
        print(f"unseen {c} fold4 F0.5: before {before['macro_f05']:.4f} -> after {after['macro_f05']:.4f} "
              f"(P {before['pair_precision']:.4f}->{after['pair_precision']:.4f}, "
              f"R {before['pair_recall']:.4f}->{after['pair_recall']:.4f})")
        return

    # test mode
    from .predict import write_lists
    train_rows = pl.read_parquet(work / "stage3_train.parquet")
    test = pl.read_parquet(work / "stage3_test.parquet")
    preds = pl.read_parquet(work / "test_preds_stage3.parquet").select("sidx", "tidx", "p2", "p3", "score")
    test = test.drop([c for c in ("p2", "p3", "score") if c in test.columns]).join(preds, on=["sidx", "tidx"])
    s1 = pl.read_parquet(work / "norm" / "test_source1.parquet", columns=["idx", "entity_id", "country"])
    test = test.join(s1.select(sidx=pl.col("idx").cast(pl.UInt32), country="country"), on="sidx")
    train_countries = set(pl.read_parquet(work / "norm" / "train_source1.parquet", columns=["country"])
                          ["country"].unique())
    targets = args.countries or sorted(set(test["country"].unique()) - train_countries)
    log(f"adapting countries {targets}")
    unl = test.filter(pl.col("country").is_in(targets))
    pseudo = pseudo_labels(unl)
    log(f"pseudo-labels: {pseudo['label'].sum():,} pos, {(pseudo['label'] == 0).sum():,} neg from {len(unl):,} rows")
    m = retrain(train_rows.filter(pl.col("fold").is_in(TRAIN_FOLDS)), pseudo,
                train_rows.filter(pl.col("fold") == TUNE_FOLD), feats, params, args.rounds)
    m.save_model(str(mdir / "stage3_selftrain.json"))
    keep_cols = list(dict.fromkeys(["sidx", "tidx", "p2", *feats]))
    adapted = blend_scores(unl.select(keep_cols).with_columns(
        p3=pl.Series(predict_frame(m, unl, feats, BATCH_ROWS))), weight).select("sidx", "tidx", "score")
    final = pl.concat([preds.select("sidx", "tidx", "score").join(adapted, on=["sidx", "tidx"], how="anti"),
                       adapted])
    matches = decide(final, thr, "score")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tg_ids = pl.concat([pl.read_parquet(work / "norm" / "test_source2.parquet", columns=["entity_id"]),
                        pl.read_parquet(work / "norm" / "test_source3.parquet", columns=["entity_id"])])["entity_id"]
    s1o = s1.select("idx", "entity_id")
    write_lists(out / "candidate_pairs.tsv", s1o, final.sort("sidx", "score", descending=[False, True]),
                tg_ids, "candidate_entity_ids")
    write_lists(out / "matching_results.tsv", s1o, matches.sort("sidx", "score", descending=[False, True]),
                tg_ids, "matched_entity_ids")
    changed = (decide(preds.select("sidx", "tidx", "score"), thr, "score")
               .join(matches, on=["sidx", "tidx"], how="anti"))
    log(f"{len(matches):,} matches ({len(changed):,} original matches dropped by adaptation)")


if __name__ == "__main__":
    main()
