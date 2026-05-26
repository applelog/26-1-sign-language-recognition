import argparse
import json
import mimetypes
import pickle
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

# Add parent directory to sys.path to import project modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import project-specific logic from root
from angle_calculator import compute_features
from control_gestures import detect_bilateral_control
from runtime_state import INPUT, RecognitionSession, StableGestureDetector
from simple_svm_model import predict_single

GEMINI_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "sign_language_model.pkl"
PILOT_MODEL_PATH = ROOT_DIR / "pilot_sign_language_model.pkl"
CONFIG_PATH = ROOT_DIR / "gesture_config.json"

# Settings (from real_time_recognition.py)
STABILITY_WINDOW = 7
MIN_STABLE_RATIO = 0.7
MIN_CONFIDENCE = 0.75


class GeminiInferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            self.respond_json({
                "labels": self.server.labels,
                "control_labels": self.server.app_config["control_labels"]
            })
            return
        
        # Serve static files from gemini/ folder
        self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        payload = self.read_json_body()
        if payload is None:
            return

        if parsed.path == "/api/predict":
            self.handle_predict(payload)
            return
        
        if parsed.path == "/api/reset":
            self.server.session.clear()
            self.server.input_detector.reset()
            self.server.control_detector.reset()
            self.respond_json({"ok": True, "message": "Session cleared"})
            return

        self.respond_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

    def serve_static(self, path):
        if path in ("/", ""):
            target = GEMINI_DIR / "index.html"
        else:
            target = (GEMINI_DIR / path.lstrip("/")).resolve()

        if not str(target).startswith(str(GEMINI_DIR.resolve())) or not target.exists() or not target.is_file():
            self.respond_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(target.name)
        mime_type = mime_type or "application/octet-stream"
        with open(target, "rb") as file:
            content = file.read()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_predict(self, payload):
        """
        Payload: {
            "hands": [{"landmarks": [...], "handedness": "Right" | "Left"}],
            "timestamp": float
        }
        """
        raw_hands = payload.get("hands", [])
        
        # 1. Prepare for Bilateral Control Detection
        # detect_bilateral_control expects a list of hand objects with .landmark attribute
        class MockHand:
            def __init__(self, landmarks):
                self.landmark = landmarks # This should be a list of objects with x, y, z

        hands_detected = [MockHand(h["landmarks"]) for h in raw_hands]
        
        control_labels = self.server.app_config["control_labels"]
        labels_map = self.server.labels
        session = self.server.session
        input_detector = self.server.input_detector
        control_detector = self.server.control_detector
        model = self.server.model

        response = {
            "mode": session.state,
            "prediction": "손 없음",
            "confidence": 0,
            "stable_label": None,
            "stable_text": "안정화 대기",
            "message": session.message,
            "text": session.text,
            "glosses": session.composer.tokens
        }

        control_id = detect_bilateral_control(
            hands_detected, control_labels["start"], control_labels["end"]
        )

        if control_id is not None:
            input_detector.observe_no_hand()
            label_obj = labels_map.get(control_id, {})
            response["prediction"] = f"제어 후보: {label_obj.get('code', control_id)} (양손)"
            stable = control_detector.observe(control_id, 1.0)
            if stable["stable"]:
                response["stable_label"] = labels_map.get(stable["label_id"], {}).get("code")
                response["stable_text"] = f"확정 후보: {response['stable_label']} (100%)"
            if stable["event_label"] is not None:
                session.handle(stable["event_label"])
        
        elif len(raw_hands) == 1 and session.state == INPUT:
            control_detector.observe_no_hand()
            # Right hand only for static signs (as per guide)
            right_hand_data = next((h for h in raw_hands if h["handedness"] == "Right"), None)
            if right_hand_data:
                hand = MockHand(right_hand_data["landmarks"])
                features = compute_features(hand)
                label_id, confidence, _ = predict_single(model, features)
                
                label_obj = labels_map.get(label_id, {})
                response["prediction"] = f"{label_obj.get('code', label_id)}"
                response["confidence"] = int(confidence * 100)
                
                stable = input_detector.observe(label_id, confidence)
                if stable["stable"]:
                    response["stable_label"] = labels_map.get(stable["label_id"], {}).get("code")
                    response["stable_text"] = f"확정 후보: {response['stable_label']} ({stable['confidence'] * 100:.0f}%)"
                if stable["event_label"] is not None:
                    session.handle(stable["event_label"])
            else:
                input_detector.observe_no_hand()
                response["prediction"] = "오른손을 인식 영역에 두세요"
        
        elif len(raw_hands) == 2:
            input_detector.observe_no_hand()
            control_detector.observe_no_hand()
            response["prediction"] = "양손 제어 자세를 유지하세요"
        
        elif raw_hands:
            input_detector.observe_no_hand()
            control_detector.observe_no_hand()
            response["prediction"] = "양손 START를 보여주세요"
        
        else:
            input_detector.observe_no_hand()
            control_detector.observe_no_hand()
            response["prediction"] = "손 없음"

        # Sync session state
        response["mode"] = session.state
        response["text"] = session.text
        response["message"] = session.message
        # Assuming RecognitionSession might have been updated to store glosses if we want to show them
        # For now, let's just send the text and let the JS handle the 'glosses' if needed, 
        # or we can mock it if session doesn't have it.
        
        self.respond_json(response)

    def read_json_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            return json.loads(raw.decode("utf-8"))
        except:
            self.respond_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return None

    def respond_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def respond_error(self, status, message):
        self.respond_json({"ok": False, "error": message}, status=status)

    def log_message(self, format, *args):
        return


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model(config, model_path):
    try:
        with model_path.open("rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    config = load_config()
    model = load_model(config, MODEL_PATH)
    if not model:
        # Try pilot model
        model = load_model(config, PILOT_MODEL_PATH)
        if not model:
            print("No model found. Please run train_model.py first.")
            return

    labels = {int(label["id"]): label for label in config["labels"]}
    
    server = ThreadingHTTPServer(("0.0.0.0", args.port), GeminiInferenceHandler)
    server.app_config = config
    server.model = model
    server.labels = labels
    server.session = RecognitionSession(config["labels"], config["control_labels"])
    server.input_detector = StableGestureDetector(STABILITY_WINDOW, MIN_STABLE_RATIO, MIN_CONFIDENCE)
    server.control_detector = StableGestureDetector(STABILITY_WINDOW, MIN_STABLE_RATIO, MIN_CONFIDENCE)

    print(f"Gemini Inference Server running on http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")


if __name__ == "__main__":
    main()
