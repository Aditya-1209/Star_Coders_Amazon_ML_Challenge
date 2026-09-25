"""Competition scoring: per-anchor F0.5, including empty truth/predictions."""

import numpy as np


def f05(true_ids, predicted_ids):
    truth, predicted = set(true_ids), set(predicted_ids)
    if not truth and not predicted:
        return 1.0
    return 1.25 * len(truth & predicted) / (0.25 * len(truth) + len(predicted))


def aggregate(true_counts, predicted_counts, true_positives):
    true_counts = np.asarray(true_counts)
    predicted_counts = np.asarray(predicted_counts)
    true_positives = np.asarray(true_positives)
    denominator = 0.25 * true_counts + predicted_counts
    scores = np.ones(len(true_counts), dtype=float)
    np.divide(1.25 * true_positives, denominator, out=scores, where=denominator > 0)
    tp, predicted, actual = int(true_positives.sum()), int(predicted_counts.sum()), int(true_counts.sum())
    singletons = true_counts == 0
    return {
        "anchors": len(true_counts), "macro_f05": float(scores.mean()) if len(scores) else None,
        "pair_precision": tp / predicted if predicted else None,
        "pair_recall": tp / actual if actual else None,
        "true_positives": tp, "false_positives": predicted - tp, "false_negatives": actual - tp,
        "singletons": int(singletons.sum()),
        "singleton_accuracy": float(np.mean(predicted_counts[singletons] == 0)) if singletons.any() else None,
    }


def evaluate(scores, labels, owners, true_counts, selected, threshold):
    accepted = scores >= threshold
    predicted = np.bincount(owners[accepted], minlength=len(true_counts))
    tp = np.bincount(owners[accepted & (labels == 1)], minlength=len(true_counts))
    return aggregate(true_counts[selected], predicted[selected], tp[selected])


def tune_threshold(scores, labels, owners, true_counts, selected):
    # Endpoints include an empty prediction for every entity; tie-break toward
    # higher thresholds, reflecting the precision-heavy objective.
    grid = np.unique(np.r_[np.linspace(0, 1, 201), np.nextafter(1.0, 2.0)])
    values = [(float(t), evaluate(scores, labels, owners, true_counts, selected, t)["macro_f05"]) for t in grid]
    threshold, score = max(values, key=lambda item: (item[1], item[0]))
    return threshold, score, [{"threshold": threshold, "macro_f05": score} for threshold, score in values]

