import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
    import mediapipe as mp
except ModuleNotFoundError as error:
    missing = getattr(error, "name", "dependency")
    print(f"필수 패키지 '{missing}'가 현재 Python 환경에 없습니다.")
    print("웹캠 평가를 쓰려면 requirements.txt의 패키지를 설치하세요.")
    raise SystemExit(1)

from src.sign_language.angle_calculator import compute_features
from src.sign_language.simple_svm_model import predict_single


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "gesture_config.json"
MODEL_PATH = ROOT_DIR / "models" / "sign_language_model.pkl"
PILOT_MODEL_PATH = ROOT_DIR / "models" / "pilot_sign_language_model.pkl"
REPORT_DIR = ROOT_DIR / "reports" / "metrics"


def parse_args():
    parser = argparse.ArgumentParser(description="실제 웹캠 정적 수화 평가")
    parser.add_argument("--label", type=int, required=True, help="테스트할 정답 라벨 ID")
    parser.add_argument("--samples", type=int, default=50, help="수집할 예측 프레임 수")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 카메라 인덱스")
    parser.add_argument("--warmup", type=float, default=2.0, help="평가 시작 전 준비 시간(초)")
    parser.add_argument("--pilot", action="store_true", help="파일럿 모델로 평가합니다.")
    return parser.parse_args()


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model(path):
    try:
        with path.open("rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise SystemExit(f"'{path.relative_to(ROOT_DIR)}' 모델 파일이 없습니다. train_model.py를 먼저 실행하세요.")
    if not isinstance(model, dict) or model.get("type") != "sklearn_svm":
        raise SystemExit("지원하지 않는 모델 형식입니다. train_model.py로 다시 학습하세요.")
    return model


def label_text(labels, label_id):
    label = labels.get(int(label_id))
    if not label:
        return f"ID {label_id}"
    return f"{label['code']}({label_id})"


def save_report(payload):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"webcam_eval_{timestamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples는 1 이상이어야 합니다.")

    config = load_config()
    labels = {int(label["id"]): label for label in config.get("labels", [])}
    active_ids = {int(label_id) for label_id in config.get("active_label_ids", [])}
    if args.label not in active_ids:
        raise SystemExit(f"라벨 {args.label}은 현재 학습 대상이 아닙니다.")

    model_path = PILOT_MODEL_PATH if args.pilot else MODEL_PATH
    model = load_model(model_path)
    if args.label not in set(model.get("labels", [])):
        raise SystemExit(f"모델에 라벨 {args.label}이 없습니다.")

    feature_dim = int(model.get("feature_dim", config.get("feature_dim", 83)))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    predictions = []
    confidences = []
    inference_times_ms = []
    started_at = time.time() + max(0.0, args.warmup)
    target_text = label_text(labels, args.label)

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while cap.isOpened() and len(predictions) < args.samples:
                ok, image = cap.read()
                if not ok:
                    break
                image = cv2.flip(image, 1)
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                hands_detected = result.multi_hand_landmarks or []
                for hand in hands_detected:
                    drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

                now = time.time()
                status = "warming up"
                if now >= started_at and len(hands_detected) == 1:
                    features = compute_features(hands_detected[0])
                    if len(features) != feature_dim:
                        raise RuntimeError("모델과 특징 수가 다릅니다. 모델을 다시 학습하세요.")
                    predict_started = time.perf_counter()
                    prediction, confidence, _ = predict_single(model, features)
                    inference_times_ms.append((time.perf_counter() - predict_started) * 1000.0)
                    predictions.append(int(prediction))
                    confidences.append(float(confidence))
                    status = f"pred={label_text(labels, prediction)} conf={confidence * 100:.1f}%"
                elif now >= started_at:
                    status = "show one hand"

                cv2.putText(
                    image,
                    f"target={target_text} samples={len(predictions)}/{args.samples}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (80, 255, 180),
                    2,
                )
                cv2.putText(
                    image,
                    status,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    image,
                    "q: quit",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 220, 255),
                    2,
                )
                cv2.imshow("Webcam Evaluation", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not predictions:
        raise SystemExit("저장된 예측이 없습니다.")

    predictions_array = np.asarray(predictions, dtype=np.int32)
    correct = int(np.sum(predictions_array == int(args.label)))
    accuracy = correct / len(predictions)
    mean_confidence = float(np.mean(confidences))
    avg_inference_time_ms = float(np.mean(inference_times_ms))
    report = {
        "target_label_id": int(args.label),
        "target_label": target_text,
        "model_path": str(model_path.relative_to(ROOT_DIR)),
        "samples": int(len(predictions)),
        "correct": correct,
        "accuracy": float(accuracy),
        "mean_confidence": mean_confidence,
        "avg_inference_time_ms": avg_inference_time_ms,
        "predictions": predictions,
        "confidences": confidences,
        "inference_times_ms": inference_times_ms,
        "created_at": datetime.now().isoformat(),
    }
    report_path = save_report(report)

    print("[웹캠 평가 결과]")
    print(f"정답 라벨: {target_text}")
    print(f"샘플 수: {len(predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"평균 신뢰도: {mean_confidence:.4f}")
    print(f"평균 추론 시간: {avg_inference_time_ms:.4f} ms/frame")
    print(f"저장 파일: {report_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
