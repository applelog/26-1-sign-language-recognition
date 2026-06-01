import math
from collections import Counter

import numpy as np


def stratified_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for label in sorted(np.unique(y)):
        label_indices = np.where(y == label)[0]
        shuffled = rng.permutation(label_indices)
        test_count = max(1, int(round(len(shuffled) * test_ratio))) if len(shuffled) > 1 else 0

        if test_count >= len(shuffled):
            test_count = len(shuffled) - 1

        test_indices.extend(shuffled[:test_count].tolist())
        train_indices.extend(shuffled[test_count:].tolist())

    if not train_indices:
        raise ValueError("학습 데이터가 부족합니다. 각 라벨마다 최소 2개 이상의 샘플이 필요합니다.")

    return (
        X[np.array(train_indices)],
        X[np.array(test_indices)] if test_indices else np.empty((0, X.shape[1])),
        y[np.array(train_indices)],
        y[np.array(test_indices)] if test_indices else np.empty((0,), dtype=y.dtype),
    )


def fit_knn_model(X, y, k=5):
    unique_labels = sorted(np.unique(y).tolist())
    effective_k = min(k, len(X))
    return {
        "type": "simple_knn",
        "k": effective_k,
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int32),
        "labels": unique_labels,
        "feature_dim": int(X.shape[1]),
    }


def _distance_weights(distances):
    return 1.0 / (distances + 1e-6)


def predict_single(model, features):
    X = model["X"]
    y = model["y"]
    k = model["k"]
    labels = model["labels"]

    sample = np.asarray(features, dtype=np.float32)
    distances = np.linalg.norm(X - sample, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y[nearest_indices]
    nearest_distances = distances[nearest_indices]
    weights = _distance_weights(nearest_distances)

    weighted_scores = {label: 0.0 for label in labels}
    for label, weight in zip(nearest_labels, weights):
        weighted_scores[int(label)] += float(weight)

    prediction = max(weighted_scores, key=weighted_scores.get)
    total_weight = sum(weighted_scores.values()) or 1.0
    confidence = weighted_scores[prediction] / total_weight
    probability_vector = np.array(
        [weighted_scores[label] / total_weight for label in labels],
        dtype=np.float32,
    )

    return int(prediction), float(confidence), probability_vector


def predict_batch(model, X):
    predictions = []
    confidences = []
    for row in X:
        label, confidence, _ = predict_single(model, row)
        predictions.append(label)
        confidences.append(confidence)
    return np.asarray(predictions), np.asarray(confidences)


def classification_report_text(y_true, y_pred):
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    lines = []
    lines.append(f"{'label':>8} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")

    correct = 0
    total = len(y_true)
    macro_precision = []
    macro_recall = []
    macro_f1 = []

    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)
        correct += tp

        lines.append(
            f"{label:>8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}"
        )

    accuracy = correct / total if total else 0.0
    lines.append("")
    lines.append(f"{'accuracy':>8} {accuracy:>32.4f} {total:>10d}")
    lines.append(
        f"{'macro avg':>8} {np.mean(macro_precision):>10.4f} {np.mean(macro_recall):>10.4f} "
        f"{np.mean(macro_f1):>10.4f} {total:>10d}"
    )

    return "\n".join(lines)
