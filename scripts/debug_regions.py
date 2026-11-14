"""사진에서 얼굴 부위를 찾아 색칠한 확인용 이미지를 만든다.

사용법 (backend 폴더에서):
    uv run python scripts/debug_regions.py 사진.jpg
    uv run python scripts/debug_regions.py 사진폴더/

결과는 debug_output/ 폴더에 '원래이름_regions.jpg'로 저장된다.
주황=볼, 하늘=이마, 분홍=입술, 초록=눈동자
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.analysis_v2.landmarks import (  # noqa: E402
    FaceNotFoundError,
    draw_debug,
    extract_regions,
    filter_skin_pixels,
)

OUT_DIR = Path("debug_output")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_rgb(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((1600, 1600))
    return np.array(image)


def mean_hex(pixels: np.ndarray) -> str:
    if len(pixels) == 0:
        return "(없음)"
    r, g, b = np.median(pixels, axis=0).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def run(path: Path) -> None:
    rgb = load_rgb(path)
    try:
        regions = extract_regions(rgb)
    except FaceNotFoundError as exc:
        print(f"[실패] {path.name}: {exc}")
        return

    print(f"[완료] {path.name}")
    for name, mask in regions.masks.items():
        pixels = filter_skin_pixels(rgb, mask) if "cheek" in name or name == "forehead" else rgb[mask]
        print(f"   {name:12s} 픽셀 {int(mask.sum()):6d}개  대표색 {mean_hex(pixels)}")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{path.stem}_regions.jpg"
    Image.fromarray(draw_debug(rgb, regions)).save(out_path, quality=92)
    print(f"   → {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    paths = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS) if target.is_dir() else [target]
    for p in paths:
        run(p)


if __name__ == "__main__":
    main()
