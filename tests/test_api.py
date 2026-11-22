from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VTO_PROVIDER", "mock")
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://testserver")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "vto_provider": "mock", "try_on_queue": {"active": 0, "waiting": 0}}


def test_consent_required(client):
    response = client.post("/api/v1/analyses", data={"consent": "false"}, files={"image": ("face.jpg", b"bad", "image/jpeg")})
    assert response.status_code == 400


def test_rejects_non_image(client):
    response = client.post("/api/v1/analyses", data={"consent": "true"}, files={"image": ("face.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_full_flow_with_stubbed_analysis(client, monkeypatch):
    import app.main as main
    result = {
        "season": "spring", "season_ko": "봄 웜톤", "undertone": "warm", "confidence": .8,
        "measured_skin_color": {"hex": "#D7A080", "rgb": (215, 160, 128)},
        "metrics": {"warmth_score": 12.0}, "palette": ["#FF8A65"],
        "recommendations": [
            {"category": "lip", "name": "코랄", "color": "#FF6F61", "description": "추천"},
            {"category": "blush", "name": "살구", "color": "#F6A57A", "description": "추천"},
            {"category": "eye", "name": "베이지", "color": "#C99152", "description": "추천"},
        ],
        "disclaimer": "체험용", "face_box": {"x": 60, "y": 40, "w": 180, "h": 220},
    }
    monkeypatch.setattr(main, "analyze", lambda path: result)
    image = np.full((320, 300, 3), (150, 180, 220), np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    analysis = client.post("/api/v1/analyses", data={"consent": "true"}, files={"image": ("face.jpg", encoded.tobytes(), "image/jpeg")})
    assert analysis.status_code == 201
    aid = analysis.json()["id"]
    products = client.get(f"/api/v1/analyses/{aid}/products")
    assert products.status_code == 200
    assert {item["category"] for item in products.json()} == {"lip", "blush", "eye"}

    tried = client.post("/api/v1/try-ons", json={"analysis_id": aid, "intensity": 55})
    assert tried.status_code == 201
    assert client.get(tried.json()["result_image_url"]).status_code == 200

    shared = client.post(f"/api/v1/analyses/{aid}/share")
    assert shared.status_code == 201
    token = shared.json()["token"]
    assert client.get(f"/api/v1/public/results/{token}").status_code == 200
    qr = client.get(f"/api/v1/public/results/{token}/qr")
    assert qr.status_code == 200
    assert qr.headers["content-type"] == "image/png"

    stats = client.get("/api/v1/admin/stats", headers={"X-Admin-Token": "test-admin-token"})
    assert stats.status_code == 200
    assert stats.json()["analyses"]["completed"] == 1
    assert client.get("/api/v1/admin/stats").status_code == 404

    assert client.delete(f"/api/v1/analyses/{aid}").status_code == 204
    assert client.get(f"/api/v1/analyses/{aid}").status_code == 404
    assert client.get(f"/api/v1/public/results/{token}").status_code == 404



def test_try_on_limit(client, monkeypatch, tmp_path):
    import app.main as main
    from app.db import Database

    settings = get_settings()
    monkeypatch.setattr(settings, "max_try_ons_per_analysis", 2)
    monkeypatch.setattr(settings, "max_total_try_ons", 4)

    db = Database(settings)

    result = {
        "recommendations": [
            {"category": "lip", "color": "#FF6F61"},
            {"category": "blush", "color": "#F6A57A"},
            {"category": "eye", "color": "#C99152"},
        ],
        "face_box": {"x": 60, "y": 40, "w": 180, "h": 220},
    }

    class FakeProvider:
        async def render(self, *args):
            return "test-task"

    monkeypatch.setattr(main, "provider_for", lambda _: FakeProvider())

    for aid in ["user1", "user2"]:
        db.create_analysis(aid, str(tmp_path / f"{aid}.jpg"))
        db.finish_analysis(aid, result)

    def try_on(aid, token=None):
        headers = {"X-Admin-Token": token} if token else {}

        return client.post(
            "/api/v1/try-ons",
            json={"analysis_id": aid, "intensity": 15},
            headers=headers,
        )

    assert try_on("user1").status_code == 201
    assert try_on("user1").status_code == 201
    assert try_on("user1").status_code == 429

    assert try_on("user1", "wrong-token").status_code == 429

    assert try_on("user1", "test-admin-token").status_code == 201

    assert try_on("user2").status_code == 201

    db.delete_analysis("user1")

    assert try_on("user2").status_code == 429
    assert try_on("user2", "test-admin-token").status_code == 429