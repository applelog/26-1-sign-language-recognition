import argparse
import csv
import glob
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.sign_language.simple_svm_model import (
    classification_report_text,
    fit_svm_model,
    predict_batch,
    stratified_split,
)


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "gesture_config.json"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports" / "metrics"
SAVE_MODEL_PATH = MODEL_DIR / "sign_language_model.pkl"
PILOT_MODEL_PATH = MODEL_DIR / "pilot_sign_language_model.pkl"
DATA_PATTERNS = [
    str(ROOT_DIR / "gesture_data_label_*.csv"),
    str(ROOT_DIR / "dataset" / "features" / "**" / "*.csv"),
]
PILOT_DATA_PATTERNS = [
    str(ROOT_DIR / "team_share" / "data_collection_bundle" / "dataset" / "features" / "**" / "*.csv"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="정적 수화 SVM 모델 학습기")
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="설정의 active_label_ids 대신 CSV에 존재하는 전체 라벨로 학습합니다.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="현재 존재하는 일부 라벨만 학습해 models/pilot_sign_language_model.pkl로 저장합니다.",
    )
    parser.add_argument(
        "--collector",
        help="지정한 수집자의 CSV만 학습합니다. 라벨 폴더 구조에서는 파일명 prefix를 사용합니다. 예: --collector gangwoo",
    )
    return parser.parse_args()


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_feature_files(include_bundle=False, collector_id=None):
    matched = []
    patterns = DATA_PATTERNS + (PILOT_DATA_PATTERNS if include_bundle else [])
    for pattern in patterns:
        matched.extend(glob.glob(pattern, recursive=True))
    unique_paths = sorted(set(matched))
    if collector_id:
        unique_paths = [
            file_path
            for file_path in unique_paths
            if Path(file_path).parent.name == collector_id
            or Path(file_path).name.startswith(f"{collector_id}__")
        ]
    return unique_paths


def load_dataset(csv_files, expected_dim):
    all_data = []
    for file_path in csv_files:
        print(f"데이터 불러오는 중: {Path(file_path).relative_to(ROOT_DIR)}")
        data = np.genfromtxt(file_path, delimiter=",")
        if data.size == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != expected_dim + 1:
            raise ValueError(
                f"{file_path}의 열 수가 {data.shape[1]}입니다. "
                f"라벨 1개와 특징 {expected_dim}개가 필요합니다."
            )
        if not np.isfinite(data).all():
            raise ValueError(f"{file_path}에 숫자가 아닌 값 또는 무한값이 있습니다.")
        all_data.append(data)
    if not all_data:
        raise ValueError("유효한 데이터가 없습니다.")
    dataset = np.vstack(all_data)
    return dataset[:, 1:].astype(np.float32), dataset[:, 0].astype(np.int32)


def label_display(labels_by_id, label_id):
    label = labels_by_id.get(int(label_id))
    if not label:
        return str(label_id)
    return f"{label_id}_{label['code']}"


def write_confusion_matrix_csv(path, matrix, label_ids, labels_by_id):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted"] + [label_display(labels_by_id, label_id) for label_id in label_ids])
        for label_id, row in zip(label_ids, matrix):
            writer.writerow([label_display(labels_by_id, label_id)] + row.astype(int).tolist())


def write_confusion_matrix_text(path, matrix, label_ids, labels_by_id):
    labels = [label_display(labels_by_id, label_id) for label_id in label_ids]
    width = max(8, max(len(label) for label in labels) + 2)
    lines = ["Confusion Matrix", "rows=actual, columns=predicted", ""]
    lines.append("".ljust(width) + "".join(label.rjust(width) for label in labels))
    for label, row in zip(labels, matrix):
        lines.append(label.ljust(width) + "".join(str(int(value)).rjust(width) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metrics_report(
    *,
    config,
    model,
    y_true,
    y_pred,
    confidences,
    avg_inference_time_ms,
    save_path,
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    label_ids = sorted(np.unique(y_true).astype(int).tolist())
    labels_by_id = {int(label["id"]): label for label in config.get("labels", [])}
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0))
    mean_confidence = float(np.mean(confidences)) if len(confidences) else float("nan")

    label_rows = []
    for label_id in label_ids:
        metrics = report[str(label_id)]
        label = labels_by_id.get(label_id, {})
        label_rows.append(
            {
                "label_id": label_id,
                "code": label.get("code", str(label_id)),
                "name": label.get("name", ""),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )

    summary = {
        "model_path": str(save_path.relative_to(ROOT_DIR)),
        "model_type": model.get("type"),
        "model_params": model.get("params", {}),
        "feature_dim": int(model.get("feature_dim", config.get("feature_dim", 83))),
        "test_frames": int(len(y_true)),
        "label_count": int(len(label_ids)),
        "accuracy": accuracy,
        "macro_f1_score": macro_f1,
        "mean_confidence": mean_confidence,
        "avg_inference_time_ms": float(avg_inference_time_ms),
        "labels": label_rows,
        "confusion_matrix_labels": [
            {
                "label_id": label_id,
                "display": label_display(labels_by_id, label_id),
            }
            for label_id in label_ids
        ],
        "confusion_matrix": matrix.astype(int).tolist(),
    }

    (REPORT_DIR / "latest_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (REPORT_DIR / "classification_report.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["label_id", "code", "name", "precision", "recall", "f1_score", "support"],
        )
        writer.writeheader()
        writer.writerows(label_rows)
    write_confusion_matrix_csv(REPORT_DIR / "confusion_matrix.csv", matrix, label_ids, labels_by_id)
    write_confusion_matrix_text(REPORT_DIR / "confusion_matrix.txt", matrix, label_ids, labels_by_id)
    (REPORT_DIR / "latest_summary.txt").write_text(
        "\n".join(
            [
                "정적 수화 SVM 성능 요약",
                f"Accuracy: {accuracy:.4f}",
                f"Macro F1-score: {macro_f1:.4f}",
                f"Mean confidence: {mean_confidence:.4f}",
                f"Average inference time: {avg_inference_time_ms:.4f} ms/frame",
                f"Test frames: {len(y_true)}",
                f"Labels: {len(label_ids)}",
                f"Model: {save_path.relative_to(ROOT_DIR)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def main():
    args = parse_args()
    config = load_config()
    expected_dim = int(config.get("feature_dim", 83))
    active_label_ids = {int(label_id) for label_id in config.get("active_label_ids", [])}
    csv_files = load_feature_files(include_bundle=args.pilot, collector_id=args.collector)
    if not csv_files:
        print("학습 데이터가 없습니다. team_share/data_collection_bundle로 수집한 데이터를 먼저 모아주세요.")
        return

    try:
        X, y = load_dataset(csv_files, expected_dim)
    except ValueError as error:
        print(f"데이터 오류: {error}")
        return

    partial_training = args.all_labels or args.pilot
    if active_label_ids and not partial_training:
        selected = np.isin(y, list(active_label_ids))
        X, y = X[selected], y[selected]
        present_ids = set(np.unique(y).astype(int).tolist())
        missing_ids = sorted(active_label_ids - present_ids)
        if missing_ids:
            print(f"활성 라벨 데이터가 부족합니다. 누락된 ID: {missing_ids}")
            print("gesture_config.json의 active_label_ids 전체를 수집한 후 다시 학습하세요.")
            return

    unique_ids = np.unique(y).astype(int).tolist()
    print(f"총 데이터 프레임 수: {len(X)}")
    print(f"학습 라벨 ID: {unique_ids}")
    print(f"특징 수: {X.shape[1]}")
    if len(unique_ids) < 2:
        print("라벨이 2개 이상 있어야 학습할 수 있습니다.")
        return

    try:
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio=0.2, seed=42)
    except ValueError as error:
        print(error)
        return

    model = fit_svm_model(X_train, y_train, kernel="rbf", c_value=10.0, gamma="scale")
    model["config"] = {
        "active_label_ids": unique_ids,
        "control_labels": config.get("control_labels", {}),
        "feature_dim": expected_dim,
    }
    print("\n[성공] scikit-learn SVM 모델 학습 완료")
    print("모델 설정: RBF kernel, C=10.0, gamma=scale, class_weight=balanced")
    if len(X_test):
        inference_started = time.perf_counter()
        predictions, confidences = predict_batch(model, X_test)
        elapsed_ms = (time.perf_counter() - inference_started) * 1000.0
        avg_inference_time_ms = elapsed_ms / len(X_test)
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        print("\n[평가] 분류 리포트")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(classification_report_text(y_test, predictions))
        print(f"\n평균 신뢰도: {np.mean(confidences):.4f}")
        print(f"평균 추론 시간: {avg_inference_time_ms:.4f} ms/frame")

    save_path = PILOT_MODEL_PATH if args.pilot else SAVE_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as file:
        pickle.dump(model, file)
    print(f"\n학습된 모델이 '{save_path.relative_to(ROOT_DIR)}'에 저장되었습니다.")

    if len(X_test):
        metrics = save_metrics_report(
            config=config,
            model=model,
            y_true=y_test,
            y_pred=predictions,
            confidences=confidences,
            avg_inference_time_ms=avg_inference_time_ms,
            save_path=save_path,
        )
        print("\n[성능지표 저장]")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-score: {metrics['macro_f1_score']:.4f}")
        print(f"Confusion Matrix: {REPORT_DIR.relative_to(ROOT_DIR) / 'confusion_matrix.csv'}")
        print(f"라벨별 지표: {REPORT_DIR.relative_to(ROOT_DIR) / 'classification_report.csv'}")


if __name__ == "__main__":
    main()
