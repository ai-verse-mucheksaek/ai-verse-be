"""조명 보정 결과를 확인한다: 보정 전/후를 나란히 붙인 이미지를 만든다.

사용법 (backend 폴더에서):
    uv run python scripts/debug_lighting.py debug_input

결과는 debug_output/ 폴더에 '원래이름_lighting.jpg'로 저장된다.
왼쪽=보정 전, 오른쪽=보정 후. 기준 카드를 찾았으면 회색 패치에 초록 테두리가 그려진다.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.analysis_v2.landmarks import (  # noqa: E402
    FaceNotFoundError,
    extract_regions,
    filter_skin_pixels,
)
from app.services.analysis_v2.lighting import correct_lighting, exposure_warning  # noqa: E402

OUT_DIR = Path("debug_output")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
METHOD_KO = {"card": "기준 카드로 보정", "saved": "저장된 보정값 사용", "none": "보정 안 함 (카드 없음)"}


def load_rgb(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((1600, 1600))
    return np.array(image)


def skin_hex(rgb: np.ndarray, mask: np.ndarray) -> str:
    pixels = filter_skin_pixels(rgb, mask)
    if len(pixels) == 0:
        return "(없음)"
    r, g, b = np.median(pixels, axis=0).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def run(path: Path) -> None:
    rgb = load_rgb(path)
    result = correct_lighting(rgb)  # 확인용이라 저장값(saved)은 쓰지 않음
    print(f"[{path.name}] {METHOD_KO[result.method]}  보정값(R,G,B)={tuple(round(g, 2) for g in result.gains)}")

    try:
        regions = extract_regions(result.image)
        skin = regions.masks["left_cheek"] | regions.masks["right_cheek"] | regions.masks["forehead"]
        print(f"   피부색 보정 전 {skin_hex(rgb, skin)} → 보정 후 {skin_hex(result.image, skin)}")
        warning = exposure_warning(result.image, skin)
        if warning == "overexposed":
            print("   ⚠ 피부가 너무 밝게 찍혔어요. 조명을 줄이거나 거리를 조절해 다시 찍어 주세요.")
        elif warning == "underexposed":
            print("   ⚠ 피부가 너무 어둡게 찍혔어요. 조명을 밝게 해 주세요.")
    except FaceNotFoundError:
        print("   (얼굴은 못 찾음 — 조명 보정만 확인)")

    before, after = rgb.copy(), result.image.copy()
    if result.card_quad is not None:
        for img in (before, after):
            cv2.polylines(img, [np.round(result.card_quad).astype(np.int32)], True, (0, 255, 0), 3)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{path.stem}_lighting.jpg"
    Image.fromarray(np.hstack([before, after])).save(out_path, quality=90)
    print(f"   → {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    paths = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS) if target.is_dir() else [target]
    if not paths:
        print("사진(JPG/PNG)을 찾지 못했어요. 아이폰 HEIC 사진은 JPG로 바꿔 주세요.")
    for p in paths:
        run(p)


if __name__ == "__main__":
    main()
