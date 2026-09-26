"""R9: cross-fitted graph expansion and a guarded macro-weighted ensemble.

No stage runs on import. The runner invokes one stage per process to release RAM
and GPU allocations between stages. Fold 4 is read for metrics only in evaluate.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import polars as pl
import xgboost as xgb

from .decision import decide_country, tune_threshold
from .graph import hop_keep
from .metrics import by_country, macro_f05
from .run_block import load_split
from .run_features import ground_truth_pairs
from .stage3 import build, feature_cols
from .train import PARAMS, decide, fit, fold_expr, predict_frame

VERSION = "r9-crossfit-graph-1"
VARIANTS = {
    "pair": {"max_depth": 8, "eta": 0.05, "seed": 109, "reg_lambda": 3.0},
    "business": {"max_depth": 10, "eta": 0.05, "seed": 209, "reg_lambda": 5.0},
}


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_frame(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_suffix(".partial.parquet")
    frame.write_parquet(temporary)
    temporary.replace(path)


def tune_half() -> pl.Expr:
    """Stable split inside fold 3: A=0 fits stopping/cutoffs, B=1 gates changes."""
    return (pl.col("sidx").hash(seed=909) % 2).cast(pl.Int8)


def business_weights(frame: pl.DataFrame) -> pl.DataFrame:
    """Equal total loss weight per retrieved business, with mean row weight 1.

    This is a surrogate for the macro objective, not a differentiable F0.5 loss.
    Thresholds are subsequently tuned to the actual exclusive macro metric.
    """
    if frame.is_empty():
        raise ValueError("Cannot weight an empty training frame")
    scale = len(frame) / frame["sidx"].n_unique()
    return frame.with_columns(w=(scale / pl.len().over("sidx")).cast(pl.Float32))


def per_business(pred: pl.DataFrame, truth: pl.DataFrame, anchors: pl.Series) -> pl.DataFrame:
    a = pl.DataFrame({"sidx": anchors.cast(pl.UInt32)}).unique()
    p = pred.select("sidx", "tidx").unique().join(a, on="sidx", how="semi")
    t = truth.select("sidx", "tidx").unique().join(a, on="sidx", how="semi")
    counts = a.join(p.group_by("sidx").agg(np=pl.len()), on="sidx", how="left")
    counts = counts.join(t.group_by("sidx").agg(nt=pl.len()), on="sidx", how="left")
    counts = counts.join(p.join(t, on=["sidx", "tidx"]).group_by("sidx").agg(tp=pl.len()),
                         on="sidx", how="left").fill_null(0)
    return counts.select("sidx", f=pl.when((pl.col("np") + pl.col("nt")) == 0).then(1.0)
                         .otherwise(1.25 * pl.col("tp") / (pl.col("np") + 0.25 * pl.col("nt"))))


def paired_gate(candidate: pl.DataFrame, baseline: pl.DataFrame, truth: pl.DataFrame,
                anchors: pl.Series, minimum_gain: float) -> dict:
    if len(anchors) < 2:
        raise ValueError("The gate needs at least two businesses")
    a = per_business(candidate, truth, anchors)
    b = per_business(baseline, truth, anchors).rename({"f": "baseline"})
    delta = a.join(b, on="sidx").select(d=pl.col("f") - pl.col("baseline"))["d"]
    gain = float(delta.mean())
    se = float(delta.std(ddof=1)) / np.sqrt(len(delta))
    lower = gain - 1.645 * se
    return {"gain": gain, "standard_error": float(se), "approx_one_sided_95_lower": float(lower),
            "minimum_gain": minimum_gain, "anchors": len(delta),
            "passed": bool(gain >= minimum_gain and lower > 0),
            "caveat": "Normal approximation over businesses; shared targets may correlate errors."}


def crossfit_inputs(frame: pl.DataFrame, models: dict[int, object], features: list[str],
                    batch_rows: int, training: bool) -> pl.DataFrame:
    """Model trained on 6 scores 7, model trained on 7 scores 6; mean elsewhere."""
    frame = frame.filter(hop_keep())
    scores = np.empty(len(frame), dtype=np.float32)
    folds = frame.select(fold_expr())["fold"].to_numpy() if training else np.full(len(frame), -1)
    for scored_fold, training_fold in ((6, 7), (7, 6)):
        mask = folds == scored_fold
        if mask.any():
            scores[mask] = predict_frame(models[training_fold], frame.filter(pl.Series(mask)), features, batch_rows)
    unseen = ~np.isin(folds, [6, 7])
    if unseen.any():
        subset = frame.filter(pl.Series(unseen))
        scores[unseen] = (predict_frame(models[6], subset, features, batch_rows)
                          + predict_frame(models[7], subset, features, batch_rows)) / 2
    # Only pair-local p1 survives; p2 now denotes the cross-fitted graph score.
    return frame.select("sidx", "tidx", "p1").with_columns(p2=pl.Series(scores))


def mixed_predictions(frame: pl.DataFrame, baseline: pl.DataFrame, weight: float) -> pl.DataFrame:
    """Keyed union; missing evidence is zero. Never average by row position."""
    if not 0 <= weight <= 1:
        raise ValueError("Blend weight must be in [0, 1]")
    return (frame.select("sidx", "tidx", "score")
            .join(baseline.select("sidx", "tidx", base_score="score"),
                  on=["sidx", "tidx"], how="full", coalesce=True)
            .select("sidx", "tidx", score=(weight * pl.col("score").fill_null(0)
                                           + (1 - weight) * pl.col("base_score").fill_null(0))))


def country_table(args, split="train"):
    return pl.read_parquet(args.base / "norm" / f"{split}_source1.parquet",
                           columns=["idx", "country"]).select(sidx=pl.col("idx").cast(pl.UInt32), country="country")


def truth_pairs(args, fold=None):
    left, right = load_split(args.base, "train", columns=["idx", "entity_id", "country"])
    if fold is not None:
        left = left.filter((pl.col("idx").hash(seed=11) % 10) == fold)
    return ground_truth_pairs(args.dataset, left, right)


def seen_filter(args):
    excluded = country_table(args).filter(pl.col("country").is_in(args.exclude_country))["sidx"]
    return ~pl.col("sidx").is_in(excluded.implode())


def model_params(args, **extra):
    return {**PARAMS, "device": args.device, "nthread": args.threads,
            "max_cached_hist_node": 1024, **extra}


def read_model(args, filename):
    model = xgb.Booster(model_file=str(args.work / "models" / filename))
    model.set_param({"device": args.device, "nthread": args.threads})
    return model


def base_meta(args):
    return json.loads((args.base / "models/stage3_metrics.json").read_text())


def baseline_decision(args, pred, anchors):
    meta = base_meta(args)
    return decide_country(pred, meta["threshold"], anchors, meta.get("country_thresholds", {}), "score")


def tag(args):
    return "".join(f"_no{c}" for c in args.exclude_country)


def crossfit(args):
    # No fold-4 rows or labels are materialized in either fitting stage.
    frame = (pl.scan_parquet(args.base / f"stage3_train{tag(args)}.parquet")
             .filter(pl.col("fold").is_in([6, 7, 3])).collect(engine="streaming"))
    seen = seen_filter(args)
    valid = frame.filter((pl.col("fold") == 3) & (tune_half() == 0) & seen)
    features = base_meta(args)["features"]
    info = {"version": VERSION, "features": features, "models": {}, "stopping_fold": "3A"}
    for training_fold in (6, 7):
        model = fit(frame.filter((pl.col("fold") == training_fold) & seen), features, valid,
                    args.crossfit_rounds, model_params(args, max_depth=8, eta=0.06, seed=900 + training_fold))
        name = f"crossfit_{training_fold}.json"
        model.save_model(str(args.work / "models" / name))
        info["models"][str(training_fold)] = {"file": name, "best_iteration": model.best_iteration}
        del model
        gc.collect()
    save_json(args.work / "crossfit.json", info)


def expand_graph(args, split):
    if split == "test" and json.loads((args.work / "selection.json").read_text())["selected"] == "baseline":
        save_json(args.work / "expand_test.json", {"skipped": "baseline retained"})
        return
    info = json.loads((args.work / "crossfit.json").read_text())
    models = {k: read_model(args, info["models"][str(k)]["file"]) for k in (6, 7)}
    source = args.base / (f"stage3_train{tag(args)}.parquet" if split == "train" else "stage3_test.parquet")
    old = pl.read_parquet(source)
    inputs = crossfit_inputs(old, models, info["features"], args.batch_rows, split == "train")
    del old, models
    gc.collect()
    frame = build(split, args.base, inputs, args.base / f"feats_{split}",
                  lambda message: print(message, flush=True), batch_rows=args.batch_rows,
                  support_anchors=1000, workers=args.threads, prune_hops=True,
                  block_chunk=2000, anchor_threshold=args.anchor_threshold, hop_k=args.hop_k)
    old_count = len(inputs)
    missing = inputs.select("sidx", "tidx").join(frame.select("sidx", "tidx"), on=["sidx", "tidx"], how="anti")
    if len(missing):
        raise RuntimeError("Second expansion lost existing candidates")
    if split == "train":
        truth = truth_pairs(args)
        frame = frame.join(truth.with_columns(label=pl.lit(1, pl.Int8)), on=["sidx", "tidx"], how="left")
        frame = frame.with_columns(pl.col("label").fill_null(0), fold_expr())
    save_frame(args.work / f"graph_{split}.parquet", frame)
    save_json(args.work / f"expand_{split}.json", {"before": old_count, "after": len(frame),
              "added": len(frame) - old_count, "anchor_threshold": args.anchor_threshold, "hop_k": args.hop_k})


def fit_final(args):
    frame = (pl.scan_parquet(args.work / "graph_train.parquet")
             .filter(pl.col("fold").is_in([6, 7, 3])).collect(engine="streaming"))
    seen = seen_filter(args)
    train = frame.filter(pl.col("fold").is_in([6, 7]) & seen)
    valid = frame.filter((pl.col("fold") == 3) & (tune_half() == 0) & seen)
    del frame
    features = feature_cols(train, base_meta(args)["feature_profile"])
    info = {"version": VERSION, "features": features, "variants": {}, "stopping_fold": "3A"}
    for name, params in VARIANTS.items():
        tr = business_weights(train) if name == "business" else train
        va = business_weights(valid) if name == "business" else valid
        model = fit(tr, features, va, args.rounds, model_params(args, **params))
        model.save_model(str(args.work / "models" / f"{name}.json"))
        info["variants"][name] = {"parameters": params, "best_iteration": model.best_iteration,
                                  "weighting": "equal business" if name == "business" else "equal pair"}
        del model, tr, va
        gc.collect()
    save_json(args.work / "final_models.json", info)


def final_scores(args, frame, choice):
    info = json.loads((args.work / "final_models.json").read_text())
    names = list(VARIANTS) if choice == "mean" else [choice]
    score = np.zeros(len(frame), dtype=np.float32)
    for name in names:
        model = read_model(args, f"{name}.json")
        score += predict_frame(model, frame, info["features"], args.batch_rows) / len(names)
        del model
        gc.collect()
    return frame.select("sidx", "tidx").with_columns(score=pl.Series(score))


def select(args):
    anchors = country_table(args).filter((fold_expr() == 3) & ~pl.col("country").is_in(args.exclude_country))
    a = anchors.filter(tune_half() == 0)["sidx"]
    b = anchors.filter(tune_half() == 1)["sidx"]
    if not len(a) or not len(b):
        raise ValueError("Selection requires both 3A and 3B businesses in seen countries")
    truth = truth_pairs(args, 3)
    frame = pl.scan_parquet(args.work / "graph_train.parquet").filter(pl.col("fold") == 3).collect()
    base = pl.scan_parquet(args.base / "eval_preds_stage3.parquet").filter(pl.col("fold") == 3).collect()
    base = base.join(anchors.select("sidx"), on="sidx", how="semi")
    frame = frame.join(anchors.select("sidx"), on="sidx", how="semi")
    trials, best, best_pred = [], None, None
    # Choose one proposal using 3A only; 3B gets exactly one pass/fail comparison.
    scores = {name: final_scores(args, frame, name) for name in VARIANTS}
    mean = scores["pair"].join(scores["business"].rename({"score": "other"}), on=["sidx", "tidx"])
    scores["mean"] = mean.select("sidx", "tidx", score=(pl.col("score") + pl.col("other")) / 2)
    for name, predictions in scores.items():
        for weight in (0.5, 0.75, 1.0):
            mixed = mixed_predictions(predictions, base, weight)
            value, cutoff = tune_threshold(mixed, truth, a, "score")
            trial = {"model": name, "r9_weight": weight, "threshold": cutoff, "fold3A_macro_f05": value}
            trials.append(trial)
            if best is None or value > best["fold3A_macro_f05"] + 1e-12:
                best, best_pred = trial, mixed
    # Ownership is resolved within the evaluated population, as in cutoff tuning.
    proposal_b = decide(best_pred.join(pl.DataFrame({"sidx": b}), on="sidx", how="semi"), best["threshold"], "score")
    base_b = baseline_decision(args, base.join(pl.DataFrame({"sidx": b}), on="sidx", how="semi"), anchors)
    gate = paired_gate(proposal_b, base_b, truth, b, args.minimum_gain)
    per_candidate = by_country(proposal_b, truth, anchors.filter(tune_half() == 1))
    per_base = by_country(base_b, truth, anchors.filter(tune_half() == 1))
    country_ok = all(per_candidate[c]["macro_f05"] >= per_base[c]["macro_f05"] - 0.002 for c in per_base)
    gate["no_seen_country_drop_over_0.002"] = country_ok
    gate["passed"] = gate["passed"] and country_ok
    result = {"version": VERSION, "selected": "r9" if gate["passed"] else "baseline", "proposal": best,
              "gate": gate, "trials_on_3A": trials, "excluded_countries": args.exclude_country,
              "fold3B_candidate": macro_f05(proposal_b, truth, b), "fold3B_baseline": macro_f05(base_b, truth, b),
              "fold3B_candidate_by_country": per_candidate, "fold3B_baseline_by_country": per_base,
              "fold3_candidate_oracle": macro_f05(frame.join(truth, on=["sidx", "tidx"]), truth, anchors["sidx"]),
              "protocol_note": "3B gates only the new final layer; baseline/earlier layers used all of fold 3. Fold 4 is report-only and was seen in past r7 experiments."}
    save_json(args.work / "selection.json", result)
    print(json.dumps({"selected": result["selected"], "proposal": best, "gate": gate}, indent=2), flush=True)


def evaluate(args):
    # Load the frozen decision before any holdout truth. Never reselect here.
    selection = json.loads((args.work / "selection.json").read_text())
    anchors = country_table(args).filter(fold_expr() == 4)
    base = pl.scan_parquet(args.base / "eval_preds_stage3.parquet").filter(pl.col("fold") == 4).collect()
    baseline = baseline_decision(args, base, anchors)
    frame = pl.scan_parquet(args.work / "graph_train.parquet").filter(pl.col("fold") == 4).collect()
    proposal = selection["proposal"]
    scores = mixed_predictions(final_scores(args, frame, proposal["model"]), base, proposal["r9_weight"])
    proposed = decide(scores, proposal["threshold"], "score")
    chosen = baseline if selection["selected"] == "baseline" else proposed
    truth = truth_pairs(args, 4)
    report = {"version": VERSION, "selected": selection["selected"], "selection": selection,
              "local_fold4": macro_f05(chosen, truth, anchors["sidx"]),
              "baseline_fold4": macro_f05(baseline, truth, anchors["sidx"]),
              "proposal_fold4_diagnostic": macro_f05(proposed, truth, anchors["sidx"]),
              "by_country": by_country(chosen, truth, anchors),
              "baseline_by_country": by_country(baseline, truth, anchors),
              "proposal_by_country_diagnostic": by_country(proposed, truth, anchors),
              "baseline_candidate_oracle_fold4": macro_f05(base.join(truth, on=["sidx", "tidx"]), truth, anchors["sidx"]),
              "candidate_oracle_fold4": macro_f05(frame.join(truth, on=["sidx", "tidx"]), truth, anchors["sidx"]),
              "amazon_score": None, "france_score": None,
              "limitations": ["Local macro F0.5 is not Amazon accuracy or a leaderboard result.",
                              "France has no labels. A held-out-country proxy cannot measure France.",
                              selection["protocol_note"]]}
    if args.exclude_country:
        report["unseen_country_proxy"] = {c: report["by_country"][c] for c in args.exclude_country if c in report["by_country"]}
    save_json(args.work / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


def inference(args):
    from .predict import write_lists
    selection = json.loads((args.work / "selection.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if selection["selected"] == "baseline":
        for name in ("candidate_pairs.tsv", "matching_results.tsv"):
            shutil.copyfile(args.baseline_output / name, args.output / name)
        return
    proposal = selection["proposal"]
    frame = pl.read_parquet(args.work / "graph_test.parquet")
    base = pl.read_parquet(args.base / "test_preds_stage3.parquet")
    scores = mixed_predictions(final_scores(args, frame, proposal["model"]), base, proposal["r9_weight"])
    save_frame(args.work / "test_predictions.parquet", scores)
    chosen = decide(scores, proposal["threshold"], "score")
    norm = args.base / "norm"
    s1 = pl.read_parquet(norm / "test_source1.parquet", columns=["idx", "entity_id"])
    targets = pl.concat([pl.read_parquet(norm / f"test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
    write_lists(args.output / "candidate_pairs.tsv", s1, scores, targets, "candidate_entity_ids")
    write_lists(args.output / "matching_results.tsv", s1, chosen, targets, "matched_entity_ids")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["crossfit", "expand_train", "fit", "select", "evaluate", "expand_test", "inference"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--crossfit-rounds", type=int, default=700)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--hop-k", type=int, default=15)
    parser.add_argument("--anchor-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-gain", type=float, default=0.0005)
    parser.add_argument("--exclude-country", nargs="*", default=[])
    args = parser.parse_args()
    if min(args.threads, args.rounds, args.crossfit_rounds, args.batch_rows) < 1:
        parser.error("Runtime counts must be positive")
    if not 1 <= args.hop_k < 65535 or not 0 <= args.anchor_threshold <= 1 or args.minimum_gain < 0:
        parser.error("Invalid retrieval or gain setting")
    (args.work / "models").mkdir(parents=True, exist_ok=True)
    if args.stage.startswith("expand_"):
        expand_graph(args, args.stage.removeprefix("expand_"))
    else:
        {"crossfit": crossfit, "fit": fit_final, "select": select, "evaluate": evaluate,
         "inference": inference}[args.stage](args)


if __name__ == "__main__":
    main()
