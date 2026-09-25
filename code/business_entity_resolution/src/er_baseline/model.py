import csv
import json
from pathlib import Path
import resource
import sys
import time

from catboost import CatBoostClassifier
import numpy as np

from .data import Record, read_records, read_truth, read_tsv, write_tsv
from .metrics import aggregate, evaluate, tune_threshold
from .retrieval import Retriever
from .text import FEATURE_NAMES, features


def peak_memory_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def train(dev, index, output, per_channel=20, iterations=450, threads=6, seed=42, posting_budget=0):
    started = time.monotonic()
    dev, output = Path(dev), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    output.mkdir(parents=True)
    anchors = list(read_records(dev / "anchors.tsv"))
    truth = read_truth(dev / "ground_truth.tsv")
    split = dict(read_tsv(dev / "splits.tsv", ["source1_entity_id", "split"]))
    if set(truth) != {a.entity_id for a in anchors} or set(split) != set(truth):
        raise ValueError("Anchor, label and split IDs differ")
    if not all(sum(v == part for v in split.values()) for part in ("train", "tune", "holdout")):
        raise ValueError("Each partition must contain at least one anchor")
    retriever = Retriever(index, per_channel, posting_budget)
    max_pairs = len(anchors) * 3 * per_channel
    x = np.empty((max_pairs, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(max_pairs, dtype=np.uint8)
    owners = np.empty(max_pairs, dtype=np.int32)
    target_ids = []
    offsets = [0]
    candidate_counts, retrieved_true = [], []
    for i, anchor in enumerate(anchors):
        candidates = retriever.query(anchor)
        start = len(target_ids)
        for j, (target, ranks) in enumerate(candidates, start):
            x[j] = features(anchor, target, ranks)
            y[j] = target.entity_id in truth[anchor.entity_id]
            owners[j] = i
            target_ids.append(target.entity_id)
        end = len(target_ids)
        offsets.append(end)
        candidate_counts.append(end - start)
        retrieved_true.append(int(y[start:end].sum()))
        if (i + 1) % 250 == 0:
            print(f"Retrieved/scored features for {i + 1:,}/{len(anchors):,} anchors; {end:,} pairs; {time.monotonic() - started:.1f}s", flush=True)
    index_metadata = retriever.metadata
    retriever.close()
    n = len(target_ids)
    x, y, owners = x[:n], y[:n], owners[:n]
    offsets = np.array(offsets, dtype=np.int64)
    true_counts = np.array([len(truth[a.entity_id]) for a in anchors], dtype=np.int32)
    anchor_split = np.array([split[a.entity_id] for a in anchors])
    train_mask = anchor_split[owners] == "train"
    tune_mask = anchor_split[owners] == "tune"
    if len(np.unique(y[train_mask])) != 2 or len(np.unique(y[tune_mask])) != 2:
        raise ValueError("Training and tuning candidates must contain both labels")
    retrieval_seconds = time.monotonic() - started
    np.savez_compressed(output / "pair_features.npz", x=x, y=y, owners=owners, offsets=offsets, target_ids=np.array(target_ids), true_counts=true_counts)
    model = CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=0.075,
        loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=5,
        random_seed=seed, thread_count=threads, task_type="CPU",
        allow_writing_files=False, border_count=64,
    )
    training_start = time.monotonic()
    model.fit(x[train_mask], y[train_mask], eval_set=(x[tune_mask], y[tune_mask]),
              early_stopping_rounds=50, verbose=100)
    model.save_model(str(output / "model.cbm"))
    training_seconds = time.monotonic() - training_start
    scores = model.predict_proba(x)[:, 1]
    tuning = anchor_split == "tune"
    threshold, tuning_score, threshold_curve = tune_threshold(scores, y, owners, true_counts, tuning)
    config = {"threshold": threshold, "per_channel": per_channel, "feature_names": FEATURE_NAMES,
              "posting_budget": posting_budget,
              "model": "CatBoostClassifier", "model_library_license": "Apache-2.0", "seed": seed}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.save(output / "probabilities.npy", scores)
    partitions = {}
    candidate_counts = np.asarray(candidate_counts)
    retrieved_true = np.asarray(retrieved_true)
    countries = np.array([a.country for a in anchors])
    for partition in ("train", "tune", "holdout"):
        selected = anchor_split == partition
        values = evaluate(scores, y, owners, true_counts, selected, threshold)
        values["candidate_pair_recall"] = float(retrieved_true[selected].sum() / true_counts[selected].sum())
        values["candidate_oracle_macro_f05"] = aggregate(true_counts[selected], retrieved_true[selected], retrieved_true[selected])["macro_f05"]
        values["mean_candidates"] = float(candidate_counts[selected].mean())
        values["empty_prediction_baseline_f05"] = float((true_counts[selected] == 0).mean())
        values["countries"] = {country: evaluate(scores, y, owners, true_counts, selected & (countries == country), threshold) for country in sorted(set(countries)) if (selected & (countries == country)).any()}
        partitions[partition] = values
    feature_importance = sorted(zip(FEATURE_NAMES, model.get_feature_importance().tolist()), key=lambda item: -item[1])
    manifest = json.loads((dev / "manifest.json").read_text())
    full_corpus = index_metadata["records"] == manifest["full_training_target_pool"]
    scope = {
        "sampled_anchors": len(anchors),
        "indexed_targets": index_metadata["records"],
        "full_training_target_pool": full_corpus,
    }
    limitations = [
        "The classifier is trained and evaluated on sampled Source 1 businesses, not every labeled business.",
        "The partitions share an unlabeled search index, but no anchor or labeled positive target crosses partitions.",
        "France has no labels, so this run does not establish French matching accuracy.",
        "Country-restricted lexical retrieval can miss alternate-script names when addresses provide no shared text.",
    ]
    if not full_corpus:
        limitations.insert(0, "The reduced target pool includes sampled labels and distractors; this is easier than full-corpus retrieval and is not a leaderboard score estimate.")
    report = {
        "development_manifest": manifest, "index": index_metadata,
        "evaluation_scope": scope,
        "config": config, "partitions": partitions,
        "candidate_pairs": n, "best_iteration": model.get_best_iteration(),
        "retrieval_and_features_seconds": round(retrieval_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "total_seconds": round(time.monotonic() - started, 3),
        "peak_process_memory_mb": round(peak_memory_mb(), 1),
        "feature_importance": feature_importance,
        "threshold_curve": threshold_curve,
        "limitations": limitations,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    holdout_indices = [i for i, a in enumerate(anchors) if split[a.entity_id] == "holdout"]
    for filename, column, only_matches in (("holdout_matching_results.tsv", "matched_entity_ids", True), ("holdout_candidate_pairs.tsv", "candidate_entity_ids", False)):
        rows = []
        for i in holdout_indices:
            start, end = offsets[i:i + 2]
            ids = [target_ids[j] for j in range(start, end) if not only_matches or scores[j] >= threshold]
            if len(ids) != len(set(ids)):
                raise AssertionError("Duplicate retrieved target")
            rows.append((anchors[i].entity_id, ",".join(sorted(ids))))
        write_tsv(output / filename, ["source1_entity_id", column], rows)
    errors = []
    for i in holdout_indices:
        start, end = offsets[i:i + 2]
        probabilities = {target_ids[j]: float(scores[j]) for j in range(start, end)}
        predicted = {target for target, score in probabilities.items() if score >= threshold}
        actual = truth[anchors[i].entity_id]
        for target in sorted(predicted ^ actual):
            errors.append({"source1_entity_id": anchors[i].entity_id, "target_entity_id": target,
                           "country": anchors[i].country,
                           "error": "false_positive" if target in predicted else "false_negative",
                           "retrieved": target in probabilities, "probability": probabilities.get(target)})
    (output / "holdout_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(json.dumps({"threshold": threshold, "partitions": partitions, "timing": {"retrieval_features_s": retrieval_seconds, "training_s": training_seconds}, "peak_memory_mb": report["peak_process_memory_mb"]}, indent=2), flush=True)
    return report


def predict(anchors_path, index, model_dir, output):
    """Stream arbitrary S1 inputs into both official-format TSV outputs."""
    model_dir, output = Path(model_dir), Path(output)
    config = json.loads((model_dir / "config.json").read_text())
    if config["feature_names"] != FEATURE_NAMES:
        raise ValueError("Model feature schema does not match this code version")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("matching_results.tsv", "candidate_pairs.tsv")):
        raise FileExistsError("Refusing to overwrite existing prediction files")
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "model.cbm"))
    retriever = Retriever(index, config["per_channel"], config.get("posting_budget", 0))
    seen = set()
    try:
        with (output / "matching_results.tsv").open("w", encoding="utf-8", newline="") as mf, (output / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as cf:
            matches = csv.writer(mf, delimiter="\t", lineterminator="\n")
            candidates_out = csv.writer(cf, delimiter="\t", lineterminator="\n")
            matches.writerow(["source1_entity_id", "matched_entity_ids"])
            candidates_out.writerow(["source1_entity_id", "candidate_entity_ids"])
            for n, anchor in enumerate(read_records(anchors_path), 1):
                if not anchor.entity_id.startswith("S1-") or anchor.entity_id in seen:
                    raise ValueError(f"Invalid/duplicate Source 1 ID: {anchor.entity_id}")
                seen.add(anchor.entity_id)
                candidates = retriever.query(anchor)
                ids = [target.entity_id for target, _ in candidates]
                if candidates:
                    x = np.array([features(anchor, target, ranks) for target, ranks in candidates], dtype=np.float32)
                    scores = model.predict_proba(x, thread_count=1)[:, 1]
                    accepted = [target for target, score in zip(ids, scores) if score >= config["threshold"]]
                else:
                    accepted = []
                candidates_out.writerow([anchor.entity_id, ",".join(sorted(ids))])
                matches.writerow([anchor.entity_id, ",".join(sorted(accepted))])
                if n % 1000 == 0:
                    mf.flush(); cf.flush()
                    print(f"Predicted {n:,} anchors", flush=True)
    finally:
        retriever.close()
