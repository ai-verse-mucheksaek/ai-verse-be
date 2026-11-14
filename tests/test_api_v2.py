"""새 분석(v2)을 API에 연결한 부분 테스트."""

import math

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.schemas import AnalysisResult
from app.services.analysis_v2.classify import classify
from app.services.analysis_v2.content import build_result_content
from app.services.analysis_v2.features import ColorFeatures, LabColor
from app.services.color_analysis import to_api_result


def _lab(L, C, h):
    return LabColor(L=L, a=round(C * math.cos(math.radians(h)), 1), b=round(C * math.sin(math.radians(h)), 1),
                    C=C, h=h, hex="#E3C0A9")


def _fake_v2_output() -> dict:
    """analyze_rgb()가 돌려주는 모양과 같은 가짜 결과 (얼굴 사진 없이 테스트하려고)."""
    feats = ColorFeatures(skin=_lab(80, 26, 72), lips=_lab(55, 30, 38), iris=_lab(35, 10, 60),
                          contrast=45.0, skin_pixel_count=900)
    c = classify(feats)
    return {**c.to_dict(), "reliability": "high", "warnings": [], "warning_messages": [],
            "features": feats.to_dict(), "lighting": {"method": "card", "gains": [1.0, 1.0, 1.0], "mirrored": False},
            "face_box": [60, 40, 180, 220], "content": build_result_content(c.season, c.season_scores)}


def test_v2_result_fits_api_schema():
    api = to_api_result(_fake_v2_output())
    parsed = AnalysisResult(**api)  # 형식이 안 맞으면 여기서 에러
    assert parsed.analyzer == "v2"
    assert parsed.season == "spring"
    assert {r.category for r in parsed.recommendations} == {"lip", "blush", "eye"}
    assert parsed.content["season_ko"] == "봄 웜톤"
    assert api["face_box"] == {"x": 60, "y": 40, "w": 180, "h": 220}  # 가상 메이크업이 쓰는 형식


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def _make(analyzer: str):
        monkeypatch.setenv("DATABASE_PATH", str(tmp_path / f"{analyzer}.db"))
        monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("VTO_PROVIDER", "mock")
        monkeypatch.setenv("ANALYZER", analyzer)
        monkeypatch.setenv("PUBLIC_BASE_URL", "http://testserver")
        get_settings.cache_clear()
        return TestClient(app)
    yield _make
    get_settings.cache_clear()


def _jpeg(color=(150, 180, 220)) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((400, 400, 3), color, np.uint8))
    assert ok
    return buf.tobytes()


def test_v2_no_face_returns_friendly_422(make_client):
    with make_client("v2") as client:
        r = client.post("/api/v1/analyses", data={"consent": "true"}, files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 422
    assert "얼굴" in r.json()["detail"]


def test_v2_full_flow_and_public_card(make_client, monkeypatch):
    import app.services.analysis_v2.pipeline as pipeline

    monkeypatch.setattr(pipeline, "analyze_rgb", lambda rgb, saved_gains_path=None: _fake_v2_output())
    with make_client("v2") as client:
        created = client.post("/api/v1/analyses", data={"consent": "true"}, files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
        assert created.status_code == 201
        result = created.json()["result"]
        assert result["analyzer"] == "v2" and result["reliability"] == "high"
        assert result["content"]["booth_lip"] == "롬앤 베어 애프리콧"

        aid = created.json()["id"]
        assert client.post("/api/v1/try-ons", json={"analysis_id": aid, "intensity": 55}).status_code == 201
        token = client.post(f"/api/v1/analyses/{aid}/share").json()["token"]
        public = client.get(f"/api/v1/public/results/{token}").json()
        assert public["content"]["season_ko"] == "봄 웜톤"
        assert public["reasons"]


def test_legacy_switch(make_client, monkeypatch):
    import app.services.color_analysis as ca

    called = {}
    def fake_legacy(path):
        called["yes"] = True
        return to_api_result(_fake_v2_output()) | {"analyzer": "legacy"}
    monkeypatch.setattr(ca, "analyze_legacy", fake_legacy)
    with make_client("legacy") as client:
        r = client.post("/api/v1/analyses", data={"consent": "true"}, files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 201 and called["yes"]
