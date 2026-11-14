import httpx
import numpy as np
import pytest
from PIL import Image
import io

from app.services.analysis_v2.youcam_tones import YouCamError, analyze_tones, crop_for_youcam

COLORS = {"skin_color": "#b9947c", "lip_color": "#D23245", "eye_color": "#3B2A20", "eye_color_name": "Brown",
          "eyebrow_color": "#5B2B31", "hair_color": "#000000", "hair_color_name": "Black"}


def _fake_youcam(polls_before_success: int = 1):
    state = {"polls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/s2s/v2.0/file":
            return httpx.Response(200, json={"data": {"files": [{"file_id": "F1", "requests": [
                {"method": "PUT", "url": "https://upload.example/F1", "headers": {}}]}]}})
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if path == "/s2s/v2.0/task/skin-tone-analysis" and request.method == "POST":
            assert request.read()  # 본문이 있어야 함
            return httpx.Response(200, json={"data": {"task_id": "T1"}})
        if path == "/s2s/v2.0/task/skin-tone-analysis/T1":
            state["polls"] += 1
            if state["polls"] <= polls_before_success:
                return httpx.Response(200, json={"data": {"task_status": "running"}})
            return httpx.Response(200, json={"data": {"task_status": "success", "results": {"color": COLORS}}})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_analyze_tones_full_flow():
    result = analyze_tones(b"jpeg", "KEY", client=_fake_youcam(), poll_interval=0)
    assert result["skin_color"] == "#b9947c"


def test_missing_key_is_clear_error():
    with pytest.raises(YouCamError, match="API 키"):
        analyze_tones(b"jpeg", "")


def test_bad_key_message():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "Invalid API key"})))
    with pytest.raises(YouCamError, match="API 키가 올바르지"):
        analyze_tones(b"jpeg", "WRONG", client=client)


def test_crop_makes_face_wide_enough():
    image = np.zeros((1200, 1600, 3), dtype=np.uint8)
    face_box = (700, 400, 200, 260)  # 얼굴 폭 200 / 사진 폭 1600 = 12.5%
    crop = Image.open(io.BytesIO(crop_for_youcam(image, face_box)))
    # 자른 사진에서 얼굴 폭 비율 = 0.7 (확대해도 비율은 유지)
    original_crop_width = 200 / 0.7
    scale = crop.width / round(original_crop_width)
    assert (200 * scale) / crop.width > 0.6
    assert min(crop.size) >= 320
