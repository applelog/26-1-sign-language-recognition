import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

try:
    import cv2
    import mediapipe as mp
except ModuleNotFoundError as error:
    missing = getattr(error, "name", "dependency")
    print(f"필수 패키지 '{missing}'가 현재 Python 환경에 없습니다.")
    print("CLI 수집기를 쓰려면 requirements.txt의 패키지를 설치하세요.")
    raise SystemExit(1)

from angle_calculator import compute_features


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "gesture_config.json"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"
CONTROL_FEATURES_DIR = ROOT_DIR / "dataset" / "control_features"
SESSIONS_DIR = ROOT_DIR / "dataset" / "sessions"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_label(config, label_id):
    for label in config.get("labels", []):
        if int(label["id"]) == label_id:
            return label
    raise ValueError(f"설정에 없는 라벨 ID입니다: {label_id}")


def slug(value):
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return text.strip("_") or "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV 정적 수화 특징 데이터 수집기")
    parser.add_argument("--label", type=int, required=True, help="gesture_config.json의 라벨 ID")
    parser.add_argument("--collector", required=True, help="수집자 ID")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 카메라 인덱스")
    parser.add_argument("--warmup", type=float, default=3.0, help="촬영 시작 전 대기 초")
    parser.add_argument("--notes", default="", help="세션 메모")
    return parser.parse_args()


def save_metadata(path, metadata):
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    config = load_config()
    label = find_label(config, args.label)
    collection_ids = {int(label_id) for label_id in config.get("collection_label_ids", [])}
    if collection_ids and args.label not in collection_ids:
        raise SystemExit("이 라벨은 수집 대상이 아닙니다.")
    if args.collector not in label.get("assignees", []):
        raise SystemExit("이 라벨은 선택한 수집자에게 배정되지 않았습니다.")
    expected_dim = int(config.get("feature_dim", 83))
    collection_mode = label.get("collection_mode", "right_hand")
    target_frames = int(config.get("target_frames", 180))
    session_id = uuid4().hex[:12]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector_id = slug(args.collector)
    label_slug = f"label_{args.label}_{slug(label['code'])}"

    save_dir = CONTROL_FEATURES_DIR if collection_mode == "both_hands" else FEATURES_DIR
    collector_dir = save_dir / collector_id
    collector_dir.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = collector_dir / f"{label_slug}__{timestamp}__{session_id}.csv"
    metadata_path = SESSIONS_DIR / f"{session_id}.json"
    metadata = {
        "session_id": session_id,
        "collector_id": collector_id,
        "label_id": args.label,
        "label_code": label["code"],
        "label_name": label["name"],
        "target_frames": target_frames,
        "collection_mode": collection_mode,
        "notes": args.notes,
        "created_at": datetime.now().isoformat(),
        "csv_path": str(csv_path.relative_to(ROOT_DIR)),
        "frames_saved": 0,
    }
    save_metadata(metadata_path, metadata)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    print(f"[{label['code']} / {label['name']}] 수집을 {args.warmup:g}초 뒤 시작합니다.")
    time.sleep(max(0.0, args.warmup))

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            with mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            ) as hands:
                while cap.isOpened() and metadata["frames_saved"] < target_frames:
                    ok, image = cap.read()
                    if not ok:
                        break
                    image = cv2.flip(image, 1)
                    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    captured_features = None
                    detected = []
                    if result.multi_hand_landmarks:
                        for index, landmarks in enumerate(result.multi_hand_landmarks):
                            raw_hand = result.multi_handedness[index].classification[0].label.lower()
                            physical_hand = "right" if raw_hand == "left" else "left"
                            detected.append((physical_hand, compute_features(landmarks).tolist()))
                            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
                    if collection_mode == "both_hands" and len(detected) >= 2:
                        detected.sort(key=lambda entry: entry[0])
                        captured_features = detected[0][1] + detected[1][1]
                    elif collection_mode != "both_hands":
                        right_hand = next((features for side, features in detected if side == "right"), None)
                        captured_features = right_hand

                    if captured_features is not None:
                        required_dim = expected_dim * 2 if collection_mode == "both_hands" else expected_dim
                        if len(captured_features) != required_dim:
                            raise ValueError(f"특징 수가 {len(captured_features)}입니다. 기대값은 {required_dim}입니다.")
                        writer.writerow([args.label, *captured_features])
                        metadata["frames_saved"] += 1

                    cv2.putText(
                        image,
                        f"Label: {label['code']}  Frames: {metadata['frames_saved']}/{target_frames}",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )
                    required_text = "Both hands required" if collection_mode == "both_hands" else "Right hand only"
                    cv2.putText(image, required_text, (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
                    cv2.putText(image, "Q: stop", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.imshow("Static Sign Data Collection", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        metadata["ended_at"] = datetime.now().isoformat()
        save_metadata(metadata_path, metadata)

    print(f"저장 완료: {csv_path.relative_to(ROOT_DIR)} ({metadata['frames_saved']} frames)")


if __name__ == "__main__":
    main()
