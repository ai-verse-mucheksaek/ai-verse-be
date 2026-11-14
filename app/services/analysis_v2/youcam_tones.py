"""YouCam AI Facial Color Tones Analyzer 호출 (정확도 검증용).

공식 문서: https://docs.perfectcorp.com/_bundle/reference/ai_skin_tone_analysis.json
  - 계절은 주지 않고 피부·입술·눈·눈썹·머리카락 색(HEX)만 준다.
  - 1회 성공 시 20유닛 소모 (실패하면 소모 안 됨).
  - 얼굴 폭이 이미지 폭의 60%보다 커야 한다 → crop_for_youcam()으로 얼굴만 잘라서 보낸다.

흐름: 파일 등록(/file) → 받은 URL로 직접 업로드 → 작업 시작(/task/skin-tone-analysis) → 끝날 때까지 상태 확인
"""

from __future__ import annotations

import io
import time

import httpx
import numpy as np
from PIL import Image

TASK_PATH = "/s2s/v2.0/task/skin-tone-analysis"
FILE_PATH = "/s2s/v2.0/file"
UNITS_PER_CALL = 20


class YouCamError(RuntimeError):
    pass


def crop_for_youcam(image_rgb: np.ndarray, face_box: tuple[int, int, int, int], face_ratio: float = 0.7) -> bytes:
    """얼굴 폭이 사진 폭의 face_ratio(기본 70%)가 되도록 잘라 JPEG로 돌려준다.

    짧은 변이 320px보다 작으면 YouCam이 거절하므로 키운다.
    """
    x, y, w, h = face_box
    cx, cy = x + w / 2, y + h / 2
    crop_w = w / face_ratio
    crop_h = crop_w * 1.3
    img_h, img_w = image_rgb.shape[:2]
    left = int(max(0, cx - crop_w / 2))
    right = int(min(img_w, cx + crop_w / 2))
    top_ = int(max(0, cy - crop_h / 2))
    bottom = int(min(img_h, cy + crop_h / 2))
    crop = Image.fromarray(image_rgb[top_:bottom, left:right])
    if min(crop.size) < 360:
        scale = 360 / min(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def analyze_tones(
    image_jpeg: bytes,
    api_key: str,
    base_url: str = "https://yce-api-01.makeupar.com",
    client: httpx.Client | None = None,
    poll_interval: float = 2.0,
    timeout: float = 90.0,
    strictness: str = "medium",
) -> dict:
    """YouCam에 사진을 보내 색 결과(dict)를 받는다. 예: {"skin_color": "#b9947c", "lip_color": ..., ...}"""
    if not api_key:
        raise YouCamError("PERFECTCORP_API_KEY가 비어 있습니다. .env에 YouCam API 키를 넣어 주세요.")
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    own_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        # 1) 파일 등록 → 업로드할 주소 받기
        init = client.post(
            f"{base}{FILE_PATH}",
            headers=headers,
            json={"files": [{"content_type": "image/jpeg", "file_name": "face.jpg", "file_size": len(image_jpeg)}]},
        )
        _raise_for(init, "파일 등록")
        file_info = init.json()["data"]["files"][0]
        request = file_info["requests"][0]
        upload_headers = dict(request.get("headers", {}))
        upload_headers["Content-Type"] = "image/jpeg"

        # 2) 받은 주소로 직접 업로드 (이걸 안 하면 404/500이 난다)
        uploaded = client.request(request.get("method", "PUT"), request["url"], headers=upload_headers, content=image_jpeg)
        _raise_for(uploaded, "파일 업로드")

        # 3) 작업 시작
        task = client.post(
            f"{base}{TASK_PATH}",
            headers=headers,
            json={"src_file_id": file_info["file_id"], "face_angle_strictness_level": strictness},
        )
        _raise_for(task, "작업 시작")
        task_id = task.json()["data"]["task_id"]

        # 4) 끝날 때까지 상태 확인 (확인을 안 하면 결과를 잃고 유닛만 빠질 수 있다)
        deadline = time.monotonic() + timeout
        while True:
            status = client.get(f"{base}{TASK_PATH}/{task_id}", headers=headers)
            _raise_for(status, "상태 확인")
            data = status.json()["data"]
            if data["task_status"] == "success":
                return data["results"]["color"] | {"face_quality": data["results"].get("face_quality")}
            if data["task_status"] == "error":
                raise YouCamError(f"YouCam 분석 실패: {data.get('error')} {data.get('error_message') or ''}".strip())
            if time.monotonic() > deadline:
                raise YouCamError("YouCam 응답 대기 시간이 초과됐습니다.")
            time.sleep(poll_interval)
    finally:
        if own_client:
            client.close()


def _raise_for(response: httpx.Response, step: str) -> None:
    if response.status_code < 400:
        return
    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:200]
    if response.status_code == 401:
        raise YouCamError(f"{step}: API 키가 올바르지 않습니다 ({detail})")
    if isinstance(detail, dict) and detail.get("error_code") == "CreditInsufficiency":
        raise YouCamError(f"{step}: YouCam 유닛이 부족합니다")
    raise YouCamError(f"{step} 실패 (HTTP {response.status_code}): {detail}")
