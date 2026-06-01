import pickle
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
AI_DIR = ROOT_DIR / "AI"
FRONTEND_DIR = ROOT_DIR / "frontend"
sys.path.insert(0, str(AI_DIR))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.sign_language.angle_calculator import compute_features
from src.sign_language.control_gestures import detect_bilateral_control
from src.sign_language.runtime_state import INPUT, RecognitionSession, StableGestureDetector
from src.sign_language.simple_svm_model import predict_single

CONFIG_PATH = AI_DIR / "gesture_config.json"
MODEL_PATH = AI_DIR / "models" / "sign_language_model.pkl"

STABILITY_WINDOW = 7
MIN_STABLE_RATIO = 0.7
MIN_CONFIDENCE = 0.75

rooms = {}

class LandmarkPayload(BaseModel):
    x: float
    y: float
    z: float = 0.0


class HandPayload(BaseModel):
    handedness: str = Field(pattern="^(Right|Left)$")
    landmarks: List[LandmarkPayload]


class PredictPayload(BaseModel):
    hands: List[HandPayload] = []
    timestamp: Optional[float] = None


def load_config():
    import json

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model():
    try:
        with MODEL_PATH.open("rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        raise RuntimeError("AI/models/sign_language_model.pkl 파일이 없습니다. AI/train_model.py를 먼저 실행하세요.")
    if not isinstance(model, dict) or model.get("type") != "sklearn_svm":
        raise RuntimeError("지원하지 않는 모델 형식입니다. AI/train_model.py로 다시 학습하세요.")
    return model


def to_hand_object(hand_payload):
    if len(hand_payload.landmarks) != 21:
        raise HTTPException(status_code=400, detail="손 하나의 landmarks는 21개여야 합니다.")
    return SimpleNamespace(
        landmark=[
            SimpleNamespace(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_payload.landmarks
        ]
    )


def label_payload(labels, label_id):
    if label_id is None:
        return None
    label = labels.get(int(label_id))
    if not label:
        return {"id": int(label_id), "code": str(label_id), "name": ""}
    return {
        "id": int(label_id),
        "code": label["code"],
        "name": label["name"],
        "group": label.get("group", ""),
    }


def label_code(labels, label_id):
    payload = label_payload(labels, label_id)
    return payload["code"] if payload else None


def top_prediction_payloads(labels, model_labels, probabilities, limit=3):
    ranked = sorted(
        zip(model_labels, probabilities),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    payloads = []
    for label_id, probability in ranked[:limit]:
        label = label_payload(labels, int(label_id))
        label["confidence"] = float(probability)
        label["confidence_percent"] = int(round(float(probability) * 100))
        payloads.append(label)
    return payloads


async def broadcast_room(room_code, message, exclude_socket=None):
    users = rooms.get(room_code, [])
    disconnected = []
    for user in list(users):
        socket = user["socket"]
        if socket == exclude_socket:
            continue
        try:
            await socket.send_json(message)
        except Exception:
            disconnected.append(socket)

    if disconnected:
        rooms[room_code] = [
            user for user in rooms.get(room_code, [])
            if user["socket"] not in disconnected
        ]
        if not rooms.get(room_code):
            rooms.pop(room_code, None)


class InferenceState:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.labels = {int(label["id"]): label for label in config["labels"]}
        self.session = RecognitionSession(config["labels"], config["control_labels"])
        self.input_detector = StableGestureDetector(STABILITY_WINDOW, MIN_STABLE_RATIO, MIN_CONFIDENCE)
        self.control_detector = StableGestureDetector(STABILITY_WINDOW, MIN_STABLE_RATIO, MIN_CONFIDENCE)
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.session.clear()
            self.input_detector.reset()
            self.control_detector.reset()
            return self.response_base("Session cleared")

    def response_base(self, prediction):
        return {
            "mode": self.session.state,
            "prediction": prediction,
            "prediction_label": None,
            "top_predictions": [],
            "confidence": 0,
            "stable_label": None,
            "stable": None,
            "stable_text": "안정화 대기",
            "event_label": None,
            "message": self.session.message,
            "text": self.session.text,
            "glosses": list(self.session.composer.tokens),
            "debug": {
                "hands_count": 0,
                "handedness": [],
                "mode": self.session.state,
                "predicted_label_id": None,
                "predicted_code": None,
                "predicted_group": None,
                "stable_ratio": None,
                "stable_confidence": None,
                "rejected_reason": None,
            },
        }

    def predict(self, payload):
        hands = payload.hands
        hand_objects = [to_hand_object(hand) for hand in hands]
        control_labels = self.config["control_labels"]

        with self.lock:
            response = self.response_base("손 없음")
            response["debug"]["hands_count"] = len(hands)
            response["debug"]["handedness"] = [hand.handedness for hand in hands]
            control_id = detect_bilateral_control(
                hand_objects,
                control_labels["start"],
                control_labels["end"],
            )

            if control_id is not None:
                self.input_detector.observe_no_hand()
                response["prediction"] = f"제어 후보: {label_code(self.labels, control_id)} (양손)"
                response["prediction_label"] = label_payload(self.labels, control_id)
                self.apply_prediction_debug(response, control_id, 1.0)
                stable = self.control_detector.observe(control_id, 1.0)
                self.apply_stable_response(response, stable)
                if stable["event_label"] is not None:
                    self.session.handle(stable["event_label"])

            elif len(hands) == 1 and self.session.state == INPUT:
                self.control_detector.observe_no_hand()
                hand_payload = hands[0]
                if hand_payload.handedness != "Right":
                    self.input_detector.observe_no_hand()
                    response["prediction"] = "오른손을 인식 영역에 두세요"
                    response["debug"]["rejected_reason"] = "right_hand_required"
                else:
                    features = compute_features(hand_objects[0])
                    expected_dim = int(self.model.get("feature_dim", self.config.get("feature_dim", 83)))
                    if len(features) != expected_dim:
                        raise HTTPException(status_code=500, detail="모델과 특징 수가 다릅니다.")
                    label_id, confidence, probability_vector = predict_single(self.model, features)
                    response["prediction"] = label_code(self.labels, label_id)
                    response["prediction_label"] = label_payload(self.labels, label_id)
                    response["top_predictions"] = top_prediction_payloads(
                        self.labels,
                        self.model.get("labels", []),
                        probability_vector,
                    )
                    response["confidence"] = int(confidence * 100)
                    self.apply_prediction_debug(response, label_id, confidence)
                    stable = self.input_detector.observe(label_id, confidence)
                    self.apply_stable_response(response, stable)
                    if stable["event_label"] is not None:
                        self.session.handle(stable["event_label"])

            elif len(hands) == 2:
                self.input_detector.observe_no_hand()
                self.control_detector.observe_no_hand()
                response["prediction"] = "양손 제어 자세를 유지하세요"
                response["debug"]["rejected_reason"] = "control_gesture_not_stable"

            elif hands:
                self.input_detector.observe_no_hand()
                self.control_detector.observe_no_hand()
                response["prediction"] = "양손 START를 보여주세요"
                response["debug"]["rejected_reason"] = "input_mode_not_started"

            else:
                self.input_detector.observe_no_hand()
                self.control_detector.observe_no_hand()
                response["prediction"] = "손 없음"
                response["debug"]["rejected_reason"] = "no_hand"

            response["mode"] = self.session.state
            response["debug"]["mode"] = self.session.state
            response["message"] = self.session.message
            response["text"] = self.session.text
            response["glosses"] = list(self.session.composer.tokens)
            return response

    def apply_prediction_debug(self, response, label_id, confidence):
        label = label_payload(self.labels, label_id)
        response["debug"]["predicted_label_id"] = label["id"]
        response["debug"]["predicted_code"] = label["code"]
        response["debug"]["predicted_group"] = label.get("group", "")
        response["debug"]["stable_confidence"] = float(confidence)

    def apply_stable_response(self, response, stable):
        stable_label = label_payload(self.labels, stable["label_id"])
        response["stable"] = {
            "label_id": stable["label_id"],
            "code": stable_label["code"],
            "is_stable": stable["stable"],
            "confidence": stable["confidence"],
            "ratio": stable["ratio"],
        }
        response["debug"]["stable_ratio"] = stable["ratio"]
        response["debug"]["stable_confidence"] = stable["confidence"]
        if not stable["stable"]:
            response["debug"]["rejected_reason"] = "stable_threshold_not_met"
        if stable["stable"]:
            response["stable_label"] = stable_label["code"]
            response["stable_text"] = f"확정 후보: {stable_label['code']} ({stable['confidence'] * 100:.0f}%)"
            response["debug"]["rejected_reason"] = None
        if stable["event_label"] is not None:
            response["event_label"] = label_payload(self.labels, stable["event_label"])

app = FastAPI(title="Static Sign Language MVP Backend")

config = load_config()
model = load_model()

state = InferenceState(config, model)
app.state.inference = state


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_path": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "label_count": len(model.get("labels", [])),
        "feature_dim": int(
            model.get(
                "feature_dim",
                config.get("feature_dim", 83)
            )
        ),
    }


@app.get("/api/config")
def get_config():
    return {
        "labels": config["labels"],
        "control_labels": config["control_labels"],
        "feature_dim": int(config.get("feature_dim", 83)),
        "active_label_ids": config.get("active_label_ids", []),
    }


@app.post("/api/predict")
def predict(payload: PredictPayload):
    return state.predict(payload)


@app.post("/api/reset")
def reset():
    return {"ok": True, **state.reset()}


@app.websocket("/ws/{room_code}/{nickname}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_code: str,
    nickname: str
):
    await websocket.accept()
    print(f"[연결] {nickname} 님이 {room_code} 방에 입장")

    if room_code not in rooms:
        rooms[room_code] = []

    rooms[room_code].append({
        "socket": websocket,
        "nickname": nickname
    })

    join_message = {
        "type": "user_status",
        "user": nickname,
        "status": "joined"
    }

    await broadcast_room(room_code, join_message)

    try:
        while True:
            data = await websocket.receive_json()
            print(f"[수신] {nickname}: ", data)

            await broadcast_room(room_code, data, exclude_socket=websocket)

    except WebSocketDisconnect:
        print(f"[퇴장] {nickname} 연결 종료")

        rooms[room_code] = [
            user for user in rooms.get(room_code, [])
            if user["socket"] != websocket
        ]

        leave_message = {
            "type": "user_status",
            "user": nickname,
            "status": "left"
        }

        await broadcast_room(room_code, leave_message)

        if not rooms.get(room_code):
            rooms.pop(room_code, None)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontend"
)
