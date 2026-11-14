"""조명 보정: 사진 속 기준 카드(회색 패치)가 '진짜 회색'이 되도록 사진 전체 색을 바로잡는다.

보정 순서
  1) 기준 카드를 찾으면 → 카드의 회색 패치로 보정 (method="card")
  2) 카드가 안 보이면 → 마지막으로 카드를 찾았을 때의 보정값 사용 (method="saved")
  3) 그것도 없으면 → 보정하지 않고 그대로 둔다 (method="none", 결과 신뢰도를 낮게 표시할 것)

카드 없이 하는 자동 보정(배경 평균, 흰자위 기준)은 테스트에서 오히려 색을 틀어지게 만들어 쓰지 않는다.

색 계산은 '선형 RGB'(빛의 실제 세기)에서 한다. 일반 사진 값(sRGB)은 사람 눈에 맞춰
휘어 있어서, 그대로 곱하면 밝은 곳과 어두운 곳이 다르게 틀어지기 때문이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---- 기준 카드 규격 (scripts/make_reference_card.py와 반드시 같아야 함) ----
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
CARD_W_MM, CARD_H_MM = 90.0, 60.0
MARKER_MM = 16.0
MARKER_MARGIN_MM = 4.0
# 마커 번호 → 카드 위 마커 왼쪽 위 모서리 위치(mm). 0=왼쪽위, 1=오른쪽위, 2=오른쪽아래, 3=왼쪽아래
MARKER_ORIGINS_MM = {
    0: (MARKER_MARGIN_MM, MARKER_MARGIN_MM),
    1: (CARD_W_MM - MARKER_MARGIN_MM - MARKER_MM, MARKER_MARGIN_MM),
    2: (CARD_W_MM - MARKER_MARGIN_MM - MARKER_MM, CARD_H_MM - MARKER_MARGIN_MM - MARKER_MM),
    3: (MARKER_MARGIN_MM, CARD_H_MM - MARKER_MARGIN_MM - MARKER_MM),
}
# 회색을 재는 곳: 마커 사이 가운데 영역(mm)
PATCH_X_MM = (MARKER_MARGIN_MM + MARKER_MM + 6, CARD_W_MM - MARKER_MARGIN_MM - MARKER_MM - 6)
PATCH_Y_MM = (10.0, CARD_H_MM - 10.0)
CARD_GRAY = 160  # 카드에 인쇄하는 회색(sRGB)이자, 보정 후 패치가 되어야 할 밝기

MAX_GAIN, MIN_GAIN = 4.0, 0.25  # 말도 안 되는 보정값 방지


@dataclass
class LightingResult:
    image: np.ndarray                  # 보정된 RGB 이미지 (uint8)
    method: str                        # "card" | "saved" | "none"
    gains: tuple[float, float, float]  # 선형 RGB에 곱한 값
    card_quad: np.ndarray | None = None  # 찾은 회색 패치의 네 꼭짓점 (디버그용)
    mirrored: bool = False               # 좌우 반전된 사진(셀카)에서 카드를 찾았는지


# ---------------------------------------------------------------- 색 공간 변환
def srgb_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    c = rgb_u8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


TARGET_LINEAR = float(srgb_to_linear(np.array([CARD_GRAY], dtype=np.uint8))[0])


# ---------------------------------------------------------------- 카드 찾기
def _detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID), params)


def _detect_markers(gray: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None, bool]:
    """마커를 찾는다. 못 찾으면 좌우를 뒤집어 한 번 더 찾는다.

    휴대폰·태블릿 전면 카메라는 사진을 거울처럼 좌우 반전해 저장하는 경우가 많은데,
    마커 무늬는 뒤집히면 인식되지 않는다. 뒤집은 사진에서 찾은 좌표는 원래 사진 좌표로 되돌린다.
    """
    detector = _detector()
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        return list(corners), ids, False
    corners, ids, _ = detector.detectMarkers(cv2.flip(gray, 1))
    if ids is None:
        return [], None, False
    w = gray.shape[1]
    restored = []
    for c in corners:
        c = c.copy()
        c[..., 0] = (w - 1) - c[..., 0]
        restored.append(c)
    return restored, ids, True


def find_card_patch(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """기준 카드를 찾아 (회색 패치 픽셀 (N,3), 패치 꼭짓점 (4,2), 좌우반전 여부)를 돌려준다. 없으면 None.

    마커가 하나만 보여도 카드 위치를 계산할 수 있다(손에 일부 가려져도 동작).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, mirrored = _detect_markers(gray)
    if ids is None:
        return None

    card_pts, img_pts = [], []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) not in MARKER_ORIGINS_MM:
            continue
        ox, oy = MARKER_ORIGINS_MM[int(marker_id)]
        card_pts += [(ox, oy), (ox + MARKER_MM, oy), (ox + MARKER_MM, oy + MARKER_MM), (ox, oy + MARKER_MM)]
        img_pts += marker_corners.reshape(4, 2).tolist()
    if not card_pts:
        return None

    homography, _ = cv2.findHomography(np.float32(card_pts), np.float32(img_pts))
    if homography is None:
        return None

    (x0, x1), (y0, y1) = PATCH_X_MM, PATCH_Y_MM
    patch_mm = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(patch_mm, homography).reshape(4, 2)

    h, w = gray.shape
    if np.any(quad < 0) or np.any(quad[:, 0] >= w) or np.any(quad[:, 1] >= h):
        return None  # 패치가 사진 밖으로 잘림

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
    pixels = image_rgb[mask.astype(bool)]
    if len(pixels) < 50:
        return None  # 카드가 너무 작게 찍힘
    return pixels, quad, mirrored


# ---------------------------------------------------------------- 보정값 계산
def _gains_from_reference(ref_pixels_rgb: np.ndarray) -> tuple[float, float, float] | None:
    """기준 픽셀들이 목표 회색이 되도록 하는 채널별 곱셈값."""
    lin = srgb_to_linear(ref_pixels_rgb.reshape(-1, 3))
    # 빛 반사(너무 밝음)나 노이즈 픽셀을 빼고 가운데 값을 쓴다
    not_clipped = np.all(ref_pixels_rgb.reshape(-1, 3) < 250, axis=1)
    if not_clipped.sum() < 20:
        return None
    ref = np.median(lin[not_clipped], axis=0)
    if np.any(ref <= 1e-4):
        return None
    gains = np.clip(TARGET_LINEAR / ref, MIN_GAIN, MAX_GAIN)
    return float(gains[0]), float(gains[1]), float(gains[2])


def apply_gains(image_rgb: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return linear_to_srgb(srgb_to_linear(image_rgb) * np.float32(gains))


# ---------------------------------------------------------------- 저장된 보정값
def load_saved_gains(path: Path | None) -> tuple[float, float, float] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return tuple(float(g) for g in data["gains"])  # type: ignore[return-value]
    except (ValueError, KeyError, TypeError):
        return None


def save_gains(path: Path | None, gains: tuple[float, float, float]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gains": list(gains)}), encoding="utf-8")


# ---------------------------------------------------------------- 노출 검사
def exposure_warning(image_rgb: np.ndarray, skin_mask: np.ndarray) -> str | None:
    """피부가 너무 밝게(하얗게 날아감) 또는 너무 어둡게 찍혔는지 검사한다.

    너무 밝게 찍힌 색은 보정으로도 되살릴 수 없으므로 재촬영을 안내하는 데 쓴다.
    """
    pixels = image_rgb[skin_mask]
    if len(pixels) == 0:
        return None
    if np.mean(np.any(pixels >= 250, axis=1)) > 0.05:
        return "overexposed"
    if np.median(pixels.max(axis=1)) < 60:
        return "underexposed"
    return None


# ---------------------------------------------------------------- 진입점
def correct_lighting(image_rgb: np.ndarray, saved_gains_path: Path | None = None) -> LightingResult:
    """사진 색을 보정한다. saved_gains_path를 주면 카드가 안 보일 때 마지막 보정값을 쓴다."""
    found = find_card_patch(image_rgb)
    if found is not None:
        pixels, quad, mirrored = found
        gains = _gains_from_reference(pixels)
        if gains is not None:
            save_gains(saved_gains_path, gains)  # 다음에 카드가 가려졌을 때 쓰려고 저장
            return LightingResult(apply_gains(image_rgb, gains), "card", gains, quad, mirrored)

    saved = load_saved_gains(saved_gains_path)
    if saved is not None:
        return LightingResult(apply_gains(image_rgb, saved), "saved", saved)

    return LightingResult(image_rgb.copy(), "none", (1.0, 1.0, 1.0))
