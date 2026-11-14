"""MediaPipe Face Landmarker로 얼굴 부위(피부·입술·눈동자) 마스크를 만든다.

사용 예:
    regions = extract_regions(image_rgb)   # numpy RGB 배열
    regions.masks["lips"]                  # True/False 마스크 (H, W)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"

# ---- 랜드마크 번호 (MediaPipe 478점 기준, "왼쪽/오른쪽"은 사진 속 사람 기준) ----
# 볼: 광대 아래, 콧볼 옆의 평평한 피부. 그림자·잡티가 적어 피부색 측정에 좋다.
LEFT_CHEEK = [117, 118, 100, 126, 142, 203, 206, 207, 187, 147]
RIGHT_CHEEK = [346, 347, 329, 355, 371, 423, 426, 427, 411, 376]
# 이마: 눈썹 위 가운데. 앞머리에 가려지면 자동으로 걸러진다(아래 피부 필터).
FOREHEAD = [69, 108, 151, 337, 299, 296, 336, 9, 107, 66]
# 입술: 바깥 윤곽에서 안쪽(입 벌린 부분, 치아)을 뺀다.
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
              375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
              324, 318, 402, 317, 14, 87, 178, 88, 95]
# 눈 윤곽: 눈동자 원이 눈꺼풀에 가려진 부분을 잘라내는 데 쓴다.
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
# 눈동자: 중심점 + 테두리 4점
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


class FaceNotFoundError(ValueError):
    pass


@dataclass
class FaceRegions:
    landmarks: np.ndarray                      # (478, 2) 픽셀 좌표
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # 부위 이름 → bool 마스크
    face_box: tuple[int, int, int, int] = (0, 0, 0, 0)          # x, y, w, h


@lru_cache(maxsize=1)
def _get_landmarker() -> vision.FaceLandmarker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe 모델 파일이 없습니다: {MODEL_PATH}\n"
            "README의 'MediaPipe 모델 준비' 절차대로 face_landmarker.task를 받아 두세요."
        )
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,  # 2명까지 찾아서 '여러 명' 여부를 판단한다
        min_face_detection_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(image_rgb: np.ndarray) -> np.ndarray:
    """가장 큰 얼굴 하나의 랜드마크를 픽셀 좌표 (478, 2)로 돌려준다."""
    h, w = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        raise FaceNotFoundError("얼굴을 찾지 못했습니다. 정면에서 얼굴이 잘 보이도록 다시 촬영해 주세요.")

    faces = [np.array([(p.x * w, p.y * h) for p in face], dtype=np.float32) for face in result.face_landmarks]
    # 여러 명이 찍혔으면 가장 큰 얼굴을 쓴다
    return max(faces, key=lambda pts: np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def _iris_mask(shape: tuple[int, int], pts: np.ndarray, iris_idx: list[int], eye_idx: list[int]) -> np.ndarray:
    """눈동자 원에서 동공(가운데 검은 부분)을 빼고, 눈꺼풀에 가려진 부분을 잘라낸다."""
    center = pts[iris_idx[0]]
    radius = float(np.mean(np.linalg.norm(pts[iris_idx[1:]] - center, axis=1)))
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    ring = (dist <= radius * 0.85) & (dist >= radius * 0.4)
    return ring & _polygon_mask(shape, pts[eye_idx])


def extract_regions(image_rgb: np.ndarray) -> FaceRegions:
    shape = image_rgb.shape[:2]
    pts = detect_landmarks(image_rgb)

    lips = _polygon_mask(shape, pts[LIPS_OUTER]) & ~_polygon_mask(shape, pts[LIPS_INNER])
    masks = {
        "left_cheek": _polygon_mask(shape, pts[LEFT_CHEEK]),
        "right_cheek": _polygon_mask(shape, pts[RIGHT_CHEEK]),
        "forehead": _polygon_mask(shape, pts[FOREHEAD]),
        "lips": lips,
        "left_iris": _iris_mask(shape, pts, LEFT_IRIS, LEFT_EYE),
        "right_iris": _iris_mask(shape, pts, RIGHT_IRIS, RIGHT_EYE),
    }

    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    face_box = (int(x0), int(y0), int(min(x1, shape[1]) - x0), int(min(y1, shape[0]) - y0))
    return FaceRegions(landmarks=pts, masks=masks, face_box=face_box)


def filter_skin_pixels(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 안 픽셀 중 그림자·하이라이트·머리카락처럼 튀는 값을 빼고 (N, 3) RGB로 돌려준다."""
    pixels = image_rgb[mask]
    if len(pixels) < 20:
        return pixels
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lightness = lab[:, 0]
    lo, hi = np.percentile(lightness, [10, 90])
    keep = (lightness >= lo) & (lightness <= hi)
    return pixels[keep]


# 디버그 그림에서 부위별로 칠할 색 (RGB)
DEBUG_COLORS = {
    "left_cheek": (255, 140, 0),
    "right_cheek": (255, 140, 0),
    "forehead": (0, 200, 255),
    "lips": (255, 0, 120),
    "left_iris": (0, 255, 0),
    "right_iris": (0, 255, 0),
}


def draw_debug(image_rgb: np.ndarray, regions: FaceRegions) -> np.ndarray:
    """각 부위를 반투명하게 칠하고 랜드마크 점을 찍은 RGB 이미지를 돌려준다."""
    out = image_rgb.astype(np.float32).copy()
    for name, mask in regions.masks.items():
        color = np.array(DEBUG_COLORS.get(name, (255, 255, 255)), dtype=np.float32)
        out[mask] = out[mask] * 0.45 + color * 0.55
    out = out.astype(np.uint8)
    for x, y in regions.landmarks:
        cv2.circle(out, (int(x), int(y)), 1, (255, 255, 255), -1)
    x, y, w, h = regions.face_box
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return out
