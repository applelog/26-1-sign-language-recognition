from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_returns_loaded_model_info():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["label_count"] == 32
    assert payload["feature_dim"] == 83


def test_config_returns_labels_and_control_contract():
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["active_label_ids"]) == 32
    assert payload["control_labels"] == {"start": 40, "end": 41, "delete": 42}
    assert any(label["code"] == "ㄱ" for label in payload["labels"])


def test_reset_returns_idle_state():
    response = client.post("/api/reset")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "IDLE"
    assert payload["text"] == ""


def test_predict_rejects_invalid_landmark_count():
    response = client.post(
        "/api/predict",
        json={
            "hands": [
                {
                    "handedness": "Right",
                    "landmarks": [{"x": 0, "y": 0, "z": 0}],
                }
            ]
        },
    )
    assert response.status_code == 400
