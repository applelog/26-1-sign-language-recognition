import argparse
import csv
import json
import mimetypes
import socket
import ssl
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parent
UI_DIR = ROOT_DIR / "web_ui"
DATASET_DIR = ROOT_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
CONTROL_FEATURES_DIR = DATASET_DIR / "control_features"
SESSIONS_DIR = DATASET_DIR / "sessions"
CONFIG_PATH = ROOT_DIR / "gesture_config.json"

SESSION_LOCK = threading.Lock()
ACTIVE_SESSIONS = {}

DEFAULT_COLLECTORS = [
    {"id": "gangwoo", "name": "강우"},
    {"id": "youngrae", "name": "영래"},
    {"id": "junghyo", "name": "정효"},
    {"id": "heetae", "name": "희태"},
]

DEFAULT_LABEL_SPECS = [
    ("ㄱ", "기역", "자음", "gangwoo"),
    ("ㄲ", "쌍기역", "자음", "youngrae"),
    ("ㄴ", "니은", "자음", "junghyo"),
    ("ㄷ", "디귿", "자음", "heetae"),
    ("ㄸ", "쌍디귿", "자음", "gangwoo"),
    ("ㄹ", "리을", "자음", "youngrae"),
    ("ㅁ", "미음", "자음", "junghyo"),
    ("ㅂ", "비읍", "자음", "heetae"),
    ("ㅃ", "쌍비읍", "자음", "gangwoo"),
    ("ㅅ", "시옷", "자음", "youngrae"),
    ("ㅆ", "쌍시옷", "자음", "junghyo"),
    ("ㅇ", "이응", "자음", "heetae"),
    ("ㅈ", "지읒", "자음", "gangwoo"),
    ("ㅉ", "쌍지읒", "자음", "youngrae"),
    ("ㅊ", "치읓", "자음", "junghyo"),
    ("ㅋ", "키읔", "자음", "heetae"),
    ("ㅌ", "티읕", "자음", "gangwoo"),
    ("ㅍ", "피읖", "자음", "youngrae"),
    ("ㅎ", "히읗", "자음", "junghyo"),
    ("ㅏ", "아", "모음", "heetae"),
    ("ㅐ", "애", "모음", "gangwoo"),
    ("ㅑ", "야", "모음", "youngrae"),
    ("ㅒ", "얘", "모음", "junghyo"),
    ("ㅓ", "어", "모음", "heetae"),
    ("ㅔ", "에", "모음", "gangwoo"),
    ("ㅕ", "여", "모음", "youngrae"),
    ("ㅖ", "예", "모음", "junghyo"),
    ("ㅗ", "오", "모음", "heetae"),
    ("ㅘ", "와", "모음", "gangwoo"),
    ("ㅙ", "왜", "모음", "youngrae"),
    ("ㅚ", "외", "모음", "junghyo"),
    ("ㅛ", "요", "모음", "heetae"),
    ("ㅜ", "우", "모음", "gangwoo"),
    ("ㅝ", "워", "모음", "youngrae"),
    ("ㅞ", "웨", "모음", "junghyo"),
    ("ㅟ", "위", "모음", "heetae"),
    ("ㅠ", "유", "모음", "gangwoo"),
    ("ㅡ", "으", "모음", "youngrae"),
    ("ㅢ", "의", "모음", "junghyo"),
    ("ㅣ", "이", "모음", "heetae"),
]


def build_default_config():
    labels = []
    for index, (code, name, group, assignee) in enumerate(DEFAULT_LABEL_SPECS):
        labels.append(
            {
                "id": index,
                "code": code,
                "name": name,
                "group": group,
                "description": f"{group} 정적 수화 ({code})",
                "assignees": [assignee],
            }
        )

    labels.extend(
        [
            {
                "id": 40,
                "code": "START",
                "name": "입력 시작",
                "group": "제어",
                "description": "양손 펼친손",
                "assignees": ["gangwoo"],
                "collection_mode": "both_hands",
                "runtime_rule": True,
            },
            {
                "id": 41,
                "code": "END",
                "name": "입력 종료",
                "group": "제어",
                "description": "양손 주먹",
                "assignees": ["gangwoo"],
                "collection_mode": "both_hands",
                "runtime_rule": True,
            },
            {
                "id": 42,
                "code": "DELETE",
                "name": "한 자모 삭제",
                "group": "제어",
                "description": "오른손 엄지 아래",
                "assignees": ["gangwoo"],
                "collection_mode": "right_hand",
            },
        ]
    )

    return {
        "project_name": "정적 수화 데이터 수집기",
        "capture_interval_ms": 220,
        "target_frames": 180,
        "preview_inference_ms": 90,
        "allowed_hand": "right",
        "feature_dim": 83,
        "active_label_ids": list(range(40)) + [42],
        "collection_label_ids": list(range(43)),
        "control_labels": {"start": 40, "end": 41, "delete": 42},
        "collectors": DEFAULT_COLLECTORS,
        "labels": labels,
    }


def safe_slug(value):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def ensure_dirs():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    config = build_default_config()
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            user_config = json.load(file)
        config.update({key: value for key, value in user_config.items() if key not in {"collectors", "labels"}})
        if "collectors" in user_config:
            config["collectors"] = user_config["collectors"]
        if "labels" in user_config:
            config["labels"] = user_config["labels"]

    config.setdefault("collectors", DEFAULT_COLLECTORS)
    config.setdefault("labels", build_default_config()["labels"])
    config.setdefault("capture_interval_ms", 220)
    config.setdefault("target_frames", 180)
    config.setdefault("preview_inference_ms", 90)
    config.setdefault("allowed_hand", "right")
    return config


def dataset_stats():
    stats = {}
    for root_dir in (FEATURES_DIR, CONTROL_FEATURES_DIR):
        for csv_path in root_dir.rglob("*.csv"):
            collector_id = csv_path.parent.name
            label_key = csv_path.stem.split("__", 1)[0]
            row_count = 0
            with open(csv_path, "r", encoding="utf-8") as file:
                for _ in file:
                    row_count += 1
            label_stats = stats.setdefault(collector_id, {}).setdefault(
                label_key, {"frames": 0, "sessions": 0}
            )
            label_stats["frames"] += row_count
            label_stats["sessions"] += 1
    return stats


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def collect_network_urls(bind_host, port, scheme):
    urls = []
    seen = set()

    def add_url(host):
        host = host.strip()
        if not host or host in seen:
            return
        seen.add(host)
        urls.append(f"{scheme}://{host}:{port}")

    if bind_host not in {"0.0.0.0", "::"}:
        add_url(bind_host)
        return urls

    add_url("127.0.0.1")
    add_url("localhost")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            add_url(sock.getsockname()[0])
    except OSError:
        pass

    try:
        host_name = socket.gethostname()
        for family, _, _, _, sockaddr in socket.getaddrinfo(host_name, None, socket.AF_INET):
            if family == socket.AF_INET:
                add_url(sockaddr[0])
    except OSError:
        pass

    return urls


class DatasetHandler(BaseHTTPRequestHandler):
    server_version = "DatasetServer/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            self.respond_json(
                {
                    "config": self.server.app_config,
                    "stats": dataset_stats(),
                    "serverInfo": {
                        "urls": self.server.local_urls,
                        "saveRoot": self.server.save_root,
                        "scheme": self.server.scheme,
                        "cameraSecureHint": self.server.scheme != "https",
                    },
                }
            )
            return

        if parsed.path == "/api/health":
            self.respond_json({"ok": True, "time": time.time()})
            return

        self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        payload = self.read_json_body()
        if payload is None:
            return

        if parsed.path == "/api/session/start":
            self.handle_start_session(payload)
            return

        if parsed.path == "/api/session/frame":
            self.handle_append_frame(payload)
            return

        if parsed.path == "/api/session/end":
            self.handle_end_session(payload)
            return

        self.respond_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

    def serve_static(self, path):
        if path in ("/", ""):
            target = UI_DIR / "index.html"
        else:
            target = (UI_DIR / path.lstrip("/")).resolve()

        if not str(target).startswith(str(UI_DIR.resolve())) or not target.exists() or not target.is_file():
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

    def handle_start_session(self, payload):
        collector_id = safe_slug(payload.get("collectorId") or payload.get("collector") or "unknown")
        collector_name = payload.get("collectorName") or collector_id
        label_id = int(payload.get("labelId"))
        label = next(
            (entry for entry in self.server.app_config["labels"] if int(entry["id"]) == label_id),
            None,
        )
        collection_ids = {
            int(value) for value in self.server.app_config.get("collection_label_ids", [])
        }
        if not label or (collection_ids and label_id not in collection_ids):
            self.respond_error(HTTPStatus.BAD_REQUEST, "Label is not collectable")
            return
        if collector_id not in label.get("assignees", []):
            self.respond_error(HTTPStatus.BAD_REQUEST, "Label is not assigned to this collector")
            return
        label_name = label["name"]
        label_code = label["code"]
        notes = payload.get("notes", "")
        target_frames = int(self.server.app_config.get("target_frames", 180))
        collection_mode = label.get("collection_mode", "right_hand")

        session_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_slug = f"label_{label_id}_{safe_slug(label_code)}"
        save_dir = CONTROL_FEATURES_DIR if collection_mode == "both_hands" else FEATURES_DIR
        collector_dir = save_dir / collector_id
        collector_dir.mkdir(parents=True, exist_ok=True)

        csv_path = collector_dir / f"{label_slug}__{timestamp}__{session_id}.csv"
        metadata_path = SESSIONS_DIR / f"{session_id}.json"

        metadata = {
            "session_id": session_id,
            "collector_id": collector_id,
            "collector_name": collector_name,
            "label_id": label_id,
            "label_name": label_name,
            "label_code": label_code,
            "target_frames": target_frames,
            "collection_mode": collection_mode,
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "csv_path": str(csv_path.relative_to(ROOT_DIR)),
            "frames_saved": 0,
        }
        write_json(metadata_path, metadata)

        with SESSION_LOCK:
            ACTIVE_SESSIONS[session_id] = {
                "csv_path": csv_path,
                "metadata_path": metadata_path,
                "frames_saved": 0,
                "metadata": metadata,
                "expected_row_length": 167 if collection_mode == "both_hands" else 84,
            }

        self.respond_json(
            {
                "sessionId": session_id,
                "csvPath": str(csv_path.relative_to(ROOT_DIR)),
                "message": "session started",
            }
        )

    def handle_append_frame(self, payload):
        session_id = payload.get("sessionId")
        row = payload.get("row")

        if not session_id or not isinstance(row, list) or len(row) < 2:
            self.respond_error(HTTPStatus.BAD_REQUEST, "Invalid frame payload")
            return

        with SESSION_LOCK:
            session = ACTIVE_SESSIONS.get(session_id)

        if not session:
            self.respond_error(HTTPStatus.BAD_REQUEST, "Session not found")
            return
        if len(row) != session["expected_row_length"]:
            self.respond_error(HTTPStatus.BAD_REQUEST, "Invalid feature row length")
            return
        if session["frames_saved"] >= session["metadata"]["target_frames"]:
            self.respond_json({"ok": True, "framesSaved": session["frames_saved"], "complete": True})
            return

        csv_path = session["csv_path"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        with SESSION_LOCK:
            session["frames_saved"] += 1
            session["metadata"]["frames_saved"] = session["frames_saved"]
            write_json(session["metadata_path"], session["metadata"])
            frames_saved = session["frames_saved"]

        self.respond_json(
            {
                "ok": True,
                "framesSaved": frames_saved,
                "complete": frames_saved >= session["metadata"]["target_frames"],
            }
        )

    def handle_end_session(self, payload):
        session_id = payload.get("sessionId")
        if not session_id:
            self.respond_error(HTTPStatus.BAD_REQUEST, "sessionId is required")
            return

        with SESSION_LOCK:
            session = ACTIVE_SESSIONS.pop(session_id, None)

        if not session:
            self.respond_json({"ok": True, "framesSaved": 0, "message": "session already closed"})
            return

        session["metadata"]["ended_at"] = datetime.now().isoformat()
        write_json(session["metadata_path"], session["metadata"])
        self.respond_json(
            {
                "ok": True,
                "framesSaved": session["frames_saved"],
                "csvPath": str(session["csv_path"].relative_to(ROOT_DIR)),
            }
        )

    def read_json_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.respond_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return None

        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
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


def main():
    parser = argparse.ArgumentParser(description="Static sign dataset web server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="Bind port, default: 8000")
    parser.add_argument("--cert-file", default=None, help="HTTPS certificate path")
    parser.add_argument("--key-file", default=None, help="HTTPS private key path")
    args = parser.parse_args()

    ensure_dirs()
    config = load_config()

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), DatasetHandler)
    server.app_config = config
    server.save_root = str(DATASET_DIR.resolve())
    server.scheme = "http"

    if args.cert_file and args.key_file:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
        server.scheme = "https"

    server.local_urls = collect_network_urls(args.host, args.port, server.scheme)

    print(f"[{config['project_name']}] 서버 실행 중")
    for url in server.local_urls:
        print(f" - {url}")
    print(f" - 저장 위치: {server.save_root}")
    if server.scheme != "https":
        print(" - 참고: 다른 기기 브라우저 카메라는 HTTP에서 막힐 수 있습니다. 필요하면 HTTPS로 실행하세요.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
