import sys
from pathlib import Path

import cv2
import numpy as np

from app.services.analysis_v2 import lighting as L

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_reference_card import draw_card  # noqa: E402


def _scene_with_card(cast=(1.0, 1.0, 1.0)) -> np.ndarray:
    """회색 배경에 기준 카드를 비스듬히 붙이고, 색 틀어짐(cast)을 입힌 가짜 사진."""
    scene = np.full((800, 800, 3), 120, dtype=np.uint8)
    card = np.array(draw_card())
    ch, cw = card.shape[:2]
    dst = np.float32([[300, 250], [620, 280], [610, 500], [290, 470]])
    h = cv2.getPerspectiveTransform(np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]]), dst)
    warped = cv2.warpPerspective(card, h, (800, 800))
    mask = cv2.warpPerspective(np.ones((ch, cw), np.uint8), h, (800, 800)).astype(bool)
    scene[mask] = warped[mask]
    return L.linear_to_srgb(L.srgb_to_linear(scene) * np.float32(cast))


def test_card_makes_patch_neutral_gray():
    shot = _scene_with_card(cast=(1.3, 1.0, 0.6))
    result = L.correct_lighting(shot)
    assert result.method == "card"
    pixels, _, _ = L.find_card_patch(result.image)
    r, g, b = np.median(pixels, axis=0)
    assert abs(r - L.CARD_GRAY) <= 3 and abs(g - L.CARD_GRAY) <= 3 and abs(b - L.CARD_GRAY) <= 3


def test_saved_gains_used_when_card_missing(tmp_path):
    saved = tmp_path / "gains.json"
    L.correct_lighting(_scene_with_card(cast=(1.3, 1.0, 0.6)), saved_gains_path=saved)
    no_card = np.full((400, 400, 3), 128, dtype=np.uint8)
    result = L.correct_lighting(no_card, saved_gains_path=saved)
    assert result.method == "saved"


def test_no_card_no_saved_leaves_image_unchanged():
    no_card = np.full((400, 400, 3), 128, dtype=np.uint8)
    result = L.correct_lighting(no_card)
    assert result.method == "none"
    assert np.array_equal(result.image, no_card)


def test_mirrored_selfie_card_is_found():
    """전면 카메라처럼 좌우 반전된 사진에서도 카드를 찾고, 보정 결과는 같아야 한다."""
    shot = _scene_with_card(cast=(1.3, 1.0, 0.6))
    mirrored = shot[:, ::-1].copy()
    result = L.correct_lighting(mirrored)
    assert result.method == "card" and result.mirrored
    normal = L.correct_lighting(shot)
    assert np.allclose(result.gains, normal.gains, rtol=0.02)
