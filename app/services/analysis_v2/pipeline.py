"""새 분석 전체 흐름: 얼굴 찾기 → 조명 보정 → 색 특징 → 사계절 분류.

    result = analyze_rgb(image_rgb)
    result["season_ko"]   # "여름 쿨톤"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .classify import THRESHOLDS, Thresholds, classify
from .content import build_result_content
from .features import extract_features
from .landmarks import extract_regions
from .lighting import correct_lighting, exposure_warning

WARNING_KO = {
    "no_card": "기준 카드가 보이지 않아 조명 보정 없이 분석했어요. 정확도가 낮을 수 있어요.",
    "saved_gains": "기준 카드가 가려져 직전 보정값으로 분석했어요.",
    "overexposed": "피부가 너무 밝게 찍혔어요. 조명을 줄이고 다시 찍어 주세요.",
    "underexposed": "피부가 너무 어둡게 찍혔어요. 조명을 밝게 하고 다시 찍어 주세요.",
    "no_iris": "눈동자가 잘 보이지 않아 대비는 빼고 분석했어요.",
    "low_confidence": "두 계절의 특징이 비슷하게 나타나요. 진단천으로 직접 비교해 보세요.",
}


def analyze_rgb(
    image_rgb: np.ndarray,
    saved_gains_path: Path | None = None,
    thresholds: Thresholds = THRESHOLDS,
) -> dict:
    # 1) 얼굴 부위 찾기 (보정 전 사진으로 찾아도 위치는 같다)
    regions = extract_regions(image_rgb)

    # 2) 조명 보정
    lit = correct_lighting(image_rgb, saved_gains_path=saved_gains_path)
    warnings: list[str] = []
    if lit.method == "none":
        warnings.append("no_card")
    elif lit.method == "saved":
        warnings.append("saved_gains")
    skin_mask = regions.masks["left_cheek"] | regions.masks["right_cheek"] | regions.masks["forehead"]
    exposure = exposure_warning(lit.image, skin_mask)
    if exposure:
        warnings.append(exposure)

    # 3) 색 특징
    features = extract_features(lit.image, regions)
    if features.iris is None:
        warnings.append("no_iris")

    # 4) 분류
    result = classify(features, thresholds)
    ranked = sorted(result.season_scores.values(), reverse=True)
    if ranked[0] - ranked[1] < 0.1:
        warnings.append("low_confidence")

    # 보정이 없거나 노출이 나쁘면 신뢰도 등급을 낮춘다
    if "no_card" in warnings or exposure:
        reliability = "low"
    elif "low_confidence" in warnings or "saved_gains" in warnings:
        reliability = "medium"
    else:
        reliability = "high"

    return {
        **result.to_dict(),
        "reliability": reliability,
        "warnings": warnings,
        "warning_messages": [WARNING_KO[w] for w in warnings],
        "features": features.to_dict(),
        "lighting": {"method": lit.method, "gains": [round(g, 3) for g in lit.gains], "mirrored": lit.mirrored},
        "face_box": list(regions.face_box),
        "content": build_result_content(result.season, result.season_scores),
    }
