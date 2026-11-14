"""포토박스에 붙일 조명 보정용 기준 카드를 만든다 (A4 한 장에 카드 4장).

사용법 (backend 폴더에서):
    uv run python scripts/make_reference_card.py

reference_card.pdf / reference_card.png 가 생긴다. PDF를 인쇄해서 카드 하나를 오려 쓰면 된다.
크기를 줄이거나 늘려 인쇄해도 괜찮다(가로세로 비율만 유지되면 됨).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.analysis_v2.lighting import (  # noqa: E402
    ARUCO_DICT_ID,
    CARD_GRAY,
    CARD_H_MM,
    CARD_W_MM,
    MARKER_MM,
    MARKER_ORIGINS_MM,
)

DPI = 300
QUIET_MM = 2.0  # 마커 둘레 흰 테두리 (인식률을 높여 줌)


def mm(v: float) -> int:
    return int(round(v / 25.4 * DPI))


def draw_card() -> Image.Image:
    card = Image.new("RGB", (mm(CARD_W_MM), mm(CARD_H_MM)), (CARD_GRAY,) * 3)
    draw = ImageDraw.Draw(card)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    for marker_id, (ox, oy) in MARKER_ORIGINS_MM.items():
        draw.rectangle(
            [mm(ox - QUIET_MM), mm(oy - QUIET_MM), mm(ox + MARKER_MM + QUIET_MM), mm(oy + MARKER_MM + QUIET_MM)],
            fill=(255, 255, 255),
        )
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, mm(MARKER_MM))
        card.paste(Image.fromarray(marker).convert("RGB"), (mm(ox), mm(oy)))
    draw.rectangle([0, 0, card.width - 1, card.height - 1], outline=(0, 0, 0), width=2)  # 오리는 선
    return card


def main() -> None:
    page = Image.new("RGB", (mm(210), mm(297)), (255, 255, 255))
    card = draw_card()
    draw = ImageDraw.Draw(page)
    draw.text((mm(15), mm(12)), "AI:VERSE reference card - cut along the black line. Do not bend or cover the gray area.",
              fill=(0, 0, 0))
    for i in range(4):
        x = mm(15) + (i % 2) * mm(CARD_W_MM + 10)
        y = mm(25) + (i // 2) * mm(CARD_H_MM + 15)
        page.paste(card, (x, y))
    page.save("reference_card.png", dpi=(DPI, DPI))
    page.save("reference_card.pdf", resolution=DPI)
    print("reference_card.pdf / reference_card.png 생성 완료")


if __name__ == "__main__":
    main()
