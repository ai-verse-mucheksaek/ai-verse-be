"""사계절 분류: 색 특징을 점수로 바꾸고, 두 번의 갈림길로 계절을 정한다.

  갈림길 1) 웜톤인가, 쿨톤인가?          ← 피부 색상각(노란 기 vs 분홍 기) + 입술 색상각
  갈림길 2-웜) 봄(밝고 맑음)인가, 가을(깊고 차분함)인가?   ← 피부 명도 + 채도
  갈림길 2-쿨) 여름(밝고 부드러움)인가, 겨울(깊고 또렷함)인가? ← 피부 명도 + 눈동자 대비 + 채도

각 갈림길은 '확률'처럼 0~1 값으로 계산하고 곱해서 네 계절 점수를 만든다(합계 1).
예: 웜 0.7 × 봄 0.6 = 봄 0.42.  가장 높은 계절이 결과, 그 점수가 신뢰도다.

⚠ THRESHOLDS의 숫자는 시작용 추정값이다. 5단계에서 실제 부스 사진과 YouCam 결과로 다시 맞춘다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .features import ColorFeatures


@dataclass(frozen=True)
class Thresholds:
    # 웜/쿨: 피부 색상각(도). 기준보다 크면 웜 쪽
    skin_hue_center: float = 60.0
    skin_hue_scale: float = 8.0
    # 입술 색상각(도). 코랄·주황빛이면 크고, 핑크·베리빛이면 작다
    lip_hue_center: float = 25.0
    lip_hue_scale: float = 10.0
    lip_weight: float = 0.3  # 웜/쿨 판단에서 입술 비중 (나머지는 피부)
    # 명도: 피부 L*
    skin_L_center: float = 70.0
    skin_L_scale: float = 8.0
    # 채도: 피부 C*
    skin_C_center: float = 20.0
    skin_C_scale: float = 5.0
    # 대비: 피부 L* - 눈동자 L*
    contrast_center: float = 45.0
    contrast_scale: float = 10.0
    # 점수를 확률로 바꿀 때의 기울기 (클수록 단호하게 판정)
    sharpness: float = 1.6


THRESHOLDS = Thresholds()

SEASON_KO = {"spring": "봄 웜톤", "summer": "여름 쿨톤", "autumn": "가을 웜톤", "winter": "겨울 쿨톤"}


@dataclass
class Classification:
    season: str
    season_ko: str
    undertone: str  # "warm" | "cool"
    confidence: float  # 1위 계절 점수 (0~1)
    season_scores: dict[str, float]
    axis_scores: dict[str, float]  # 웜/쿨, 밝기, 선명도, 대비 점수 (-2 ~ +2)
    reasons: list[str]  # 결과 카드에 보여줄 쉬운 설명

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _z(value: float, center: float, scale: float) -> float:
    """기준값에서 얼마나 떨어졌는지를 -2 ~ +2 사이 점수로."""
    return max(-2.0, min(2.0, (value - center) / scale))


def _sigmoid(x: float, sharpness: float) -> float:
    return 1.0 / (1.0 + math.exp(-sharpness * x))


def _angle_diff(h: float, center: float) -> float:
    """색상각 차이 (360° 경계를 넘어가도 올바르게)."""
    return (h - center + 180.0) % 360.0 - 180.0


def classify(features: ColorFeatures, t: Thresholds = THRESHOLDS) -> Classification:
    skin = features.skin

    # ---- 축 점수 ----
    warm_skin = max(-2.0, min(2.0, _angle_diff(skin.h, t.skin_hue_center) / t.skin_hue_scale))
    if features.lips is not None:
        warm_lip = max(-2.0, min(2.0, _angle_diff(features.lips.h, t.lip_hue_center) / t.lip_hue_scale))
        warmth = (1 - t.lip_weight) * warm_skin + t.lip_weight * warm_lip
    else:
        warmth = warm_skin
    brightness = _z(skin.L, t.skin_L_center, t.skin_L_scale)
    clarity = _z(skin.C, t.skin_C_center, t.skin_C_scale)
    contrast = _z(features.contrast, t.contrast_center, t.contrast_scale) if features.contrast is not None else 0.0

    # ---- 갈림길 ----
    p_warm = _sigmoid(warmth, t.sharpness)
    p_spring_if_warm = _sigmoid(0.6 * brightness + 0.4 * clarity, t.sharpness)   # 밝고 맑으면 봄
    p_summer_if_cool = _sigmoid(0.6 * brightness - 0.4 * contrast, t.sharpness)  # 밝고 대비 낮으면 여름

    scores = {
        "spring": p_warm * p_spring_if_warm,
        "autumn": p_warm * (1 - p_spring_if_warm),
        "summer": (1 - p_warm) * p_summer_if_cool,
        "winter": (1 - p_warm) * (1 - p_summer_if_cool),
    }
    season = max(scores, key=scores.get)

    return Classification(
        season=season,
        season_ko=SEASON_KO[season],
        undertone="warm" if season in ("spring", "autumn") else "cool",
        confidence=round(scores[season], 2),
        season_scores={k: round(v, 3) for k, v in scores.items()},
        axis_scores={
            "warmth": round(warmth, 2),
            "brightness": round(brightness, 2),
            "clarity": round(clarity, 2),
            "contrast": round(contrast, 2),
        },
        reasons=_reasons(warmth, brightness, clarity, contrast, features.contrast is not None),
    )


def _reasons(warmth: float, brightness: float, clarity: float, contrast: float, has_iris: bool) -> list[str]:
    reasons = []
    if warmth >= 0.3:
        reasons.append("피부에 노란 기가 도는 편이에요.")
    elif warmth <= -0.3:
        reasons.append("피부에 붉은·분홍 기가 도는 편이에요.")
    else:
        reasons.append("노란 기와 분홍 기가 고르게 섞인 중간 톤이에요.")

    if brightness >= 0.5:
        reasons.append("피부가 밝은 편이라 밝은 색이 잘 받아요.")
    elif brightness <= -0.5:
        reasons.append("피부 톤이 깊은 편이라 진한 색이 잘 받아요.")

    if clarity >= 0.5:
        reasons.append("혈색이 선명한 편이라 맑고 쨍한 색이 어울려요.")
    elif clarity <= -0.5:
        reasons.append("피부 색이 부드러운 편이라 차분하고 톤 다운된 색이 어울려요.")

    if has_iris and contrast >= 0.5:
        reasons.append("피부와 눈동자 대비가 뚜렷해요.")
    elif has_iris and contrast <= -0.5:
        reasons.append("피부와 눈동자 대비가 부드러워요.")
    return reasons
