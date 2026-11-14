import colorsys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


PALETTES = {
    "spring": {
        "ko": "봄 웜톤",
        "colors": ["#FF8A65", "#FFB74D", "#F48FB1", "#A5D66A", "#FFD54F"],
        "makeup": [("lip", "코랄 피치", "#FF6F61"), ("blush", "살구 코랄", "#F6A57A"), ("eye", "골든 베이지", "#C99152")],
    },
    "summer": {
        "ko": "여름 쿨톤",
        "colors": ["#D98CA3", "#B39DDB", "#90CAF9", "#B0BEC5", "#E6A8C5"],
        "makeup": [("lip", "로즈 핑크", "#C85A7C"), ("blush", "라벤더 핑크", "#D9A0BB"), ("eye", "소프트 모브", "#9A7F9D")],
    },
    "autumn": {
        "ko": "가을 웜톤",
        "colors": ["#A0522D", "#C47A44", "#8D6E63", "#9E8B3D", "#D07A5F"],
        "makeup": [("lip", "브릭 레드", "#A63D40"), ("blush", "테라코타", "#C66B4E"), ("eye", "웜 브라운", "#795548")],
    },
    "winter": {
        "ko": "겨울 쿨톤",
        "colors": ["#B0003A", "#6A1B9A", "#1565C0", "#263238", "#E91E63"],
        "makeup": [("lip", "체리 레드", "#B51246"), ("blush", "쿨 베리", "#B85C7A"), ("eye", "딥 플럼", "#5E315B")],
    },
}


class ImageValidationError(ValueError):
    pass


@dataclass
class FaceRegion:
    x: int
    y: int
    w: int
    h: int


def load_and_validate(path: Path) -> Image.Image:
    try:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:
        raise ImageValidationError("올바른 JPG 또는 PNG 이미지가 아닙니다.") from exc
    width, height = image.size
    if width < 100 or height < 100:
        raise ImageValidationError("이미지는 가로·세로가 각각 100px 이상이어야 합니다.")
    if max(width, height) >= 1920:
        image.thumbnail((1919, 1919), Image.Resampling.LANCZOS)
        if path.suffix.lower() == ".png":
            image.save(path, format="PNG", optimize=True)
        else:
            image.save(path, format="JPEG", quality=94)
    return image


def detect_face(rgb: np.ndarray) -> FaceRegion:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        raise ImageValidationError("정면 얼굴을 찾지 못했습니다. 밝은 곳에서 얼굴 전체가 나오도록 다시 촬영해 주세요.")
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return FaceRegion(int(x), int(y), int(w), int(h))


def _skin_pixels(rgb: np.ndarray, face: FaceRegion) -> np.ndarray:
    x, y, w, h = face.x, face.y, face.w, face.h
    # Forehead + both cheek zones; eyes, brows, hair, and lips are intentionally excluded.
    zones = [
        rgb[y + int(.20*h):y + int(.34*h), x + int(.32*w):x + int(.68*w)],
        rgb[y + int(.48*h):y + int(.67*h), x + int(.10*w):x + int(.33*w)],
        rgb[y + int(.48*h):y + int(.67*h), x + int(.67*w):x + int(.90*w)],
    ]
    pixels = np.concatenate([z.reshape(-1, 3) for z in zones if z.size], axis=0)
    ycrcb = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    mask = (ycrcb[:, 0] > 45) & (ycrcb[:, 1] > 125) & (ycrcb[:, 1] < 180) & (ycrcb[:, 2] > 75) & (ycrcb[:, 2] < 140)
    selected = pixels[mask]
    return selected if len(selected) >= 100 else pixels


def analyze(path: Path) -> dict:
    image = load_and_validate(path)
    rgb = np.asarray(image)
    face = detect_face(rgb)
    pixels = _skin_pixels(rgb, face)
    median = np.median(pixels, axis=0).astype(int)

    # Lab b* expresses blue↔yellow; HSV saturation/value separate muted/clear and light/deep.
    sample = np.uint8([[median]])
    lab = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    h, s, v = colorsys.rgb_to_hsv(*(median / 255.0))
    warmth = float((lab[2] - 128) - 0.35 * (lab[1] - 128))
    warm = warmth >= 9.0
    light = v >= 0.67
    season = ("spring" if light else "autumn") if warm else ("summer" if light else "winter")

    distance = min(abs(warmth - 9.0) / 18.0, 1.0)
    clarity = abs(v - 0.67) / 0.33
    confidence = round(min(0.92, 0.58 + 0.20 * distance + 0.12 * clarity), 2)
    spec = PALETTES[season]
    recs = [
        {"category": cat, "name": name, "color": color, "description": f"{spec['ko']}에 자연스럽게 어울리는 {name} 계열"}
        for cat, name, color in spec["makeup"]
    ]
    return {
        "season": season,
        "season_ko": spec["ko"],
        "undertone": "warm" if warm else "cool",
        "confidence": confidence,
        "measured_skin_color": {"hex": "#%02X%02X%02X" % tuple(median), "rgb": tuple(int(x) for x in median)},
        "metrics": {"lab_a": round(lab[1] - 128, 2), "lab_b": round(lab[2] - 128, 2), "warmth_score": round(warmth, 2), "saturation": round(s, 3), "brightness": round(v, 3)},
        "palette": spec["colors"],
        "recommendations": recs,
        "disclaimer": "조명과 카메라에 따라 결과가 달라질 수 있는 체험용 추천이며 전문 진단을 대체하지 않습니다.",
        "face_box": {"x": face.x, "y": face.y, "w": face.w, "h": face.h},
    }

