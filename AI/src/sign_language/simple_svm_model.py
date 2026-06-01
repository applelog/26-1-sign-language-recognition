import math

import numpy as np

try:
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ModuleNotFoundError as error:
    missing = getattr(error, "name", "dependency")
    print(f"필수 패키지 '{missing}'가 현재 Python 환경에 없습니다.")
    print("SVM 학습/인식을 쓰려면 먼저 설치하세요: pip install scikit-learn")
    raise SystemExit(1)


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


def fit_svm_model(X, y, kernel="rbf", c_value=10.0, gamma="scale"):
    unique_labels = sorted(np.unique(y).astype(int).tolist())
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=kernel,
                    C=c_value,
                    gamma=gamma,
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    classifier.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32))

    return {
        "type": "sklearn_svm",
        "classifier": classifier,
        "labels": unique_labels,
        "feature_dim": int(X.shape[1]),
        "params": {
            "kernel": kernel,
            "C": c_value,
            "gamma": gamma,
            "class_weight": "balanced",
            "probability": True,
        },
    }


def predict_single(model, features):
    classifier = model["classifier"]
    labels = model["labels"]
    sample = np.asarray(features, dtype=np.float32).reshape(1, -1)

    prediction = int(classifier.predict(sample)[0])
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(sample)[0]
        class_labels = classifier.classes_.astype(int).tolist()
        probability_by_label = {label: 0.0 for label in labels}
        for label, probability in zip(class_labels, probabilities):
            probability_by_label[int(label)] = float(probability)
        confidence = probability_by_label.get(prediction, 0.0)
        probability_vector = np.array([probability_by_label[label] for label in labels], dtype=np.float32)
    else:
        confidence = math.nan
        probability_vector = np.zeros(len(labels), dtype=np.float32)

    return prediction, float(confidence), probability_vector


def predict_batch(model, X):
    classifier = model["classifier"]
    X = np.asarray(X, dtype=np.float32)
    predictions = classifier.predict(X).astype(np.int32)

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
    else:
        confidences = np.full(len(predictions), math.nan, dtype=np.float32)

    return predictions, confidences


def classification_report_text(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4, zero_division=0)
