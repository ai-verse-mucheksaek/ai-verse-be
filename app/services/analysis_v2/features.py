"""색 특징 계산: 부위별 대표색을 CIELAB 값(명도·채도·색상각)으로 바꾼다.

CIELAB(Lab)은 사람 눈이 느끼는 색 차이에 맞춘 색 공간이다.
  L*  명도: 0(검정) ~ 100(흰색)
  a*  +면 빨강, -면 초록
  b*  +면 노랑, -면 파랑
여기서 두 가지를 더 계산한다.
  C*  채도 = √(a² + b²)  : 색이 얼마나 선명한가 (낮으면 탁하고 부드러움)
  h°  색상각 = atan2(b, a): 0°에 가까우면 붉은·분홍빛, 90°에 가까우면 노란빛
피부 색상각은 보통 40~80° 사이에 있고, 클수록 노란 기(웜), 작을수록 붉은·분홍 기(쿨)가 강하다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .landmarks import FaceRegions, filter_skin_pixels
from .lighting import srgb_to_linear

# sRGB(D65) → XYZ 변환 행렬과 D65 백색점
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


@dataclass
class LabColor:
    L: float
    a: float
    b: float
    C: float  # 채도
    h: float  # 색상각(도)
    hex: str

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> "LabColor":
        rgb_u8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        xyz = _RGB_TO_XYZ @ srgb_to_linear(rgb_u8).astype(np.float64) / _WHITE_D65
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        L = 116 * f[1] - 16
        a = 500 * (f[0] - f[1])
        b = 200 * (f[1] - f[2])
        r, g, bl = (int(v) for v in rgb_u8)
        return cls(
            L=round(float(L), 1),
            a=round(float(a), 1),
            b=round(float(b), 1),
            C=round(float(np.hypot(a, b)), 1),
            h=round(float(np.degrees(np.arctan2(b, a))) % 360, 1),
            hex=f"#{r:02X}{g:02X}{bl:02X}",
        )


@dataclass
class ColorFeatures:
    skin: LabColor
    lips: LabColor | None
    iris: LabColor | None
    contrast: float | None  # 피부 명도 - 눈동자 명도 (클수록 또렷한 대비)
    skin_pixel_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _representative(pixels: np.ndarray) -> LabColor | None:
    if len(pixels) < 10:
        return None
    return LabColor.from_rgb(np.median(pixels, axis=0))


def extract_features(image_rgb: np.ndarray, regions: FaceRegions) -> ColorFeatures:
    """조명 보정이 끝난 사진과 부위 마스크로 색 특징을 계산한다."""
    m = regions.masks
    skin_pixels = filter_skin_pixels(image_rgb, m["left_cheek"] | m["right_cheek"] | m["forehead"])
    skin = _representative(skin_pixels)
    if skin is None:
        raise ValueError("피부 영역이 너무 작아 색을 잴 수 없습니다. 얼굴이 크게 나오도록 다시 촬영해 주세요.")

    # 입술: 너무 어두운 틈(입 안 그림자)과 반사광을 빼고 잰다
    lips = _representative(filter_skin_pixels(image_rgb, m["lips"]))

    # 눈동자: 반사광(아주 밝은 점)을 빼고 잰다
    iris_pixels = image_rgb[m["left_iris"] | m["right_iris"]]
    if len(iris_pixels):
        iris_pixels = iris_pixels[iris_pixels.max(axis=1) < 200]
    iris = _representative(iris_pixels)

    contrast = round(skin.L - iris.L, 1) if iris else None
    return ColorFeatures(skin=skin, lips=lips, iris=iris, contrast=contrast, skin_pixel_count=int(len(skin_pixels)))
