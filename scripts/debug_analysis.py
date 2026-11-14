"""새 분석(analysis_v2)을 사진에 돌려서 결과와 근거를 보여준다. 기존 분석 결과도 같이 비교한다.

사용법 (backend 폴더에서):
    uv run python scripts/debug_analysis.py debug_input
"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe 안내 메시지 줄이기

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.analysis_v2.landmarks import FaceNotFoundError  # noqa: E402
from app.services.analysis_v2.pipeline import analyze_rgb  # noqa: E402
from app.services.color_analysis import analyze as analyze_old  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LIGHT_KO = {"card": "기준 카드", "saved": "저장된 보정값", "none": "보정 없음"}


def fmt(c: dict | None) -> str:
    if not c:
        return "(없음)"
    return f"{c['hex']}  명도 {c['L']:5.1f}  채도 {c['C']:5.1f}  색상각 {c['h']:5.1f}°"


def run(path: Path) -> None:
    rgb = np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    try:
        r = analyze_rgb(rgb)
    except (FaceNotFoundError, ValueError) as exc:
        print(f"[{path.name}] 실패: {exc}\n")
        return

    f = r["features"]
    print(f"[{path.name}]  →  {r['season_ko']}  (신뢰도 {r['confidence']:.0%}, 등급 {r['reliability']})")
    print(f"   조명 보정: {LIGHT_KO[r['lighting']['method']]}" + (" (좌우 반전)" if r["lighting"]["mirrored"] else ""))
    print(f"   피부   {fmt(f['skin'])}")
    print(f"   입술   {fmt(f['lips'])}")
    print(f"   눈동자 {fmt(f['iris'])}   대비 {f['contrast']}")
    a = r["axis_scores"]
    print(f"   축 점수: 웜(+)/쿨(-) {a['warmth']:+.2f} | 밝기 {a['brightness']:+.2f} | 선명도 {a['clarity']:+.2f} | 대비 {a['contrast']:+.2f}")
    ranked = sorted(r["season_scores"].items(), key=lambda kv: -kv[1])
    print("   계절 점수: " + "  ".join(f"{k} {v:.0%}" for k, v in ranked))
    for reason in r["reasons"]:
        print(f"   · {reason}")
    c = r["content"]
    print(f"   [결과 카드] 1순위: {c['season_ko']} ({c['score']:.0%})")
    print(f"     {c['summary']}")
    print(f"     어울리는 색: {', '.join(p['name'] for p in c['palette'])}")
    print(f"     옷: {', '.join(c['clothes']['best'])}  /  피할 색: {', '.join(c['clothes']['avoid'])}")
    print(f"     머리색: {', '.join(c['hair'])}  /  액세서리: {c['accessory']['metal_ko']}  /  부스 립: {c['booth_lip']}")
    print(f"     2순위: {c['second_season']['season_ko']} ({c['second_season']['score']:.0%})")
    for msg in r["warning_messages"]:
        print(f"   ⚠ {msg}")
    try:
        old = analyze_old(path)
        print(f"   (비교) 기존 분석 결과: {old['season_ko']}")
    except Exception as exc:  # 기존 분석은 비교용이라 실패해도 넘어간다
        print(f"   (비교) 기존 분석 실패: {exc}")
    print()


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
