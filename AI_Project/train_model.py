import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np

from simple_svm_model import classification_report_text, fit_svm_model, predict_batch, stratified_split


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "gesture_config.json"
SAVE_MODEL_PATH = ROOT_DIR / "sign_language_model.pkl"
PILOT_MODEL_PATH = ROOT_DIR / "pilot_sign_language_model.pkl"
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
        help="현재 존재하는 일부 라벨만 학습해 pilot_sign_language_model.pkl로 저장합니다.",
    )
    parser.add_argument(
        "--collector",
        help="지정한 수집자 폴더의 CSV만 학습합니다. 예: --collector gangwoo",
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
            file_path for file_path in unique_paths if Path(file_path).parent.name == collector_id
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
        predictions, confidences = predict_batch(model, X_test)
        print("\n[평가] 분류 리포트")
        print(classification_report_text(y_test, predictions))
        print(f"\n평균 신뢰도: {np.mean(confidences):.4f}")

    save_path = PILOT_MODEL_PATH if args.pilot else SAVE_MODEL_PATH
    with save_path.open("wb") as file:
        pickle.dump(model, file)
    print(f"\n학습된 모델이 '{save_path.name}'로 저장되었습니다.")


if __name__ == "__main__":
    main()
