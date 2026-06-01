import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import cv2
    import mediapipe as mp
except ModuleNotFoundError as error:
    missing = getattr(error, "name", "dependency")
    print(f"필수 패키지 '{missing}'가 현재 Python 환경에 없습니다.")
    print("실시간 인식을 쓰려면 requirements.txt의 패키지를 설치하세요.")
    raise SystemExit(1)

from PIL import Image, ImageDraw, ImageFont

from src.sign_language.angle_calculator import compute_features
from src.sign_language.control_gestures import detect_bilateral_control
from src.sign_language.runtime_state import INPUT, RecognitionSession, StableGestureDetector
from src.sign_language.simple_svm_model import predict_single


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "sign_language_model.pkl"
PILOT_MODEL_PATH = ROOT_DIR / "models" / "pilot_sign_language_model.pkl"
CONFIG_PATH = ROOT_DIR / "gesture_config.json"
STABILITY_WINDOW = 7
MIN_STABLE_RATIO = 0.7
MIN_CONFIDENCE = 0.75
FONT_CANDIDATES = [
    ROOT_DIR / "assets" / "fonts" / "NotoSansKR-Regular.ttf",
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("C:/Windows/Fonts/malgun.ttf"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV 정적 수화 문자 입력 MVP")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 카메라 인덱스")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="부분 라벨로 학습한 models/pilot_sign_language_model.pkl을 사용합니다.",
    )
    return parser.parse_args()


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model(config, model_path, allow_partial=False):
    try:
        with model_path.open("rb") as file:
            payload = pickle.load(file)
    except FileNotFoundError:
        raise SystemExit(f"'{model_path.relative_to(ROOT_DIR)}' 모델 파일이 없습니다. train_model.py를 먼저 실행하세요.")
    if not isinstance(payload, dict) or payload.get("type") != "sklearn_svm":
        raise SystemExit("지원하지 않는 모델 형식입니다. train_model.py로 다시 학습하세요.")
    required = {int(value) for value in config.get("active_label_ids", [])}
    missing = sorted(required - set(payload.get("labels", [])))
    if missing and not allow_partial:
        raise SystemExit(f"모델에 학습 대상 라벨이 없습니다: {missing}. 활성 라벨을 모두 수집해 재학습하세요.")
    return payload


@lru_cache(maxsize=None)
def load_font(size):
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_korean_overlay(image, lines):
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = load_font(25)
    small_font = load_font(20)
    y = 22
    for index, (text, color) in enumerate(lines):
        draw.text((25, y), text, font=font if index < 3 else small_font, fill=color)
        y += 38 if index < 3 else 30
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def label_text(labels, label_id):
    label = labels.get(int(label_id))
    return label["code"] if label else f"ID {label_id}"


def main():
    args = parse_args()
    config = load_config()
    labels = {int(label["id"]): label for label in config["labels"]}
    control_labels = config["control_labels"]
    model_path = PILOT_MODEL_PATH if args.pilot else MODEL_PATH
    model = load_model(config, model_path, allow_partial=args.pilot)
    feature_dim = int(model["feature_dim"])
    session = RecognitionSession(config["labels"], control_labels)
    input_detector = StableGestureDetector(STABILITY_WINDOW, MIN_STABLE_RATIO, MIN_CONFIDENCE)
    control_detector = StableGestureDetector(STABILITY_WINDOW, MIN_STABLE_RATIO, MIN_CONFIDENCE)

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    current_prediction = "파일럿 모델: 수집된 라벨만 예측합니다" if args.pilot else "손을 보여주세요"
    stable_text = "안정화 대기"
    try:
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while cap.isOpened():
                ok, image = cap.read()
                if not ok:
                    break
                image = cv2.flip(image, 1)
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                hands_detected = result.multi_hand_landmarks or []
                for hand in hands_detected:
                    drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

                control_id = detect_bilateral_control(
                    hands_detected, control_labels["start"], control_labels["end"]
                )
                if control_id is not None:
                    input_detector.observe_no_hand()
                    current_prediction = f"제어 후보: {label_text(labels, control_id)} (양손)"
                    stable = control_detector.observe(control_id, 1.0)
                    if stable["stable"]:
                        stable_text = (
                            f"확정 후보: {label_text(labels, stable['label_id'])} "
                            f"({stable['confidence'] * 100:.0f}%)"
                        )
                    else:
                        stable_text = "확정 후보: 안정화 대기"
                    if stable["event_label"] is not None:
                        session.handle(stable["event_label"])
                elif len(hands_detected) == 1 and session.state == INPUT:
                    control_detector.observe_no_hand()
                    hand = hands_detected[0]
                    features = compute_features(hand)
                    if len(features) != feature_dim:
                        raise RuntimeError("모델과 특징 수가 다릅니다. 모델을 다시 학습하세요.")
                    label_id, confidence, _ = predict_single(model, features)
                    current_prediction = f"현재 예측: {label_text(labels, label_id)} ({confidence * 100:.0f}%)"
                    stable = input_detector.observe(label_id, confidence)
                    if stable["stable"]:
                        stable_text = (
                            f"확정 후보: {label_text(labels, stable['label_id'])} "
                            f"({stable['confidence'] * 100:.0f}%)"
                        )
                    else:
                        stable_text = "확정 후보: 안정화 대기"
                    if stable["event_label"] is not None:
                        session.handle(stable["event_label"])
                elif len(hands_detected) == 2:
                    input_detector.observe_no_hand()
                    control_detector.observe_no_hand()
                    current_prediction = "현재 예측: 양손 제어 자세를 유지하세요"
                    stable_text = "확정 후보: 안정화 대기"
                elif hands_detected:
                    input_detector.observe_no_hand()
                    control_detector.observe_no_hand()
                    current_prediction = "현재 예측: 양손 START를 보여주세요"
                    stable_text = "확정 후보: 안정화 대기"
                else:
                    input_detector.observe_no_hand()
                    control_detector.observe_no_hand()
                    current_prediction = "현재 예측: 손 없음"
                    stable_text = "확정 후보: 안정화 대기"

                mode_text = {"IDLE": "대기", "INPUT": "입력 중", "RESULT": "결과"}[session.state]
                text_title = "입력 텍스트" if session.state == INPUT else "결과 텍스트"
                image = draw_korean_overlay(
                    image,
                    [
                        (f"모드: {mode_text}", (80, 255, 180)),
                        (f"{text_title}: {session.text or '-'}", (255, 230, 90)),
                        (current_prediction, (255, 255, 255)),
                        (stable_text, (210, 225, 245)),
                        (session.message, (100, 220, 255)),
                        ("START: 양손 펼침  END: 양손 주먹  Q: 종료  C: 초기화", (185, 190, 205)),
                    ],
                )
                cv2.imshow("Static Sign Recognition MVP", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    session.clear()
                    input_detector.reset()
                    control_detector.reset()
                elif key == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
