"""정확도 검증: 우리 분석과 YouCam 색 분석을 같은 사진으로 비교한다.

사용법 (backend 폴더에서):
    uv run python scripts/validate_youcam.py validation_input --dry-run   # 유닛 안 쓰고 우리 분석만 + 예상 유닛
    uv run python scripts/validate_youcam.py validation_input             # YouCam 호출 (사진당 20유닛)
    uv run python scripts/validate_youcam.py validation_input --limit 5    # 최대 5장만 새로 호출

- YouCam API 키는 .env의 PERFECTCORP_API_KEY를 쓴다.
- 이미 YouCam에 보낸 사진은 validation_output/cache/에 결과가 저장돼 다시 보내지 않는다(유닛 절약).
- 사진 폴더에 labels.csv(파일명,계절)가 있으면 '실제 정답'과도 비교한다. 계절은 봄/여름/가을/겨울 또는 spring/summer/autumn/winter.
- 결과는 validation_output/results.csv (엑셀로 열림)와 화면 요약으로 나온다.

비교 방법
  ① 측정 차이 ΔE: 같은 사진에서 우리가 잰 색과 YouCam이 잰 색의 차이. 2 미만이면 눈으로 구분 어려움, 5 미만이면 비슷.
  ② 분류 일치: YouCam이 잰 색을 우리 분류 규칙에 넣었을 때 같은 계절이 나오는지.
  ③ 정답 비교: labels.csv가 있을 때만.
"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402
from app.services.analysis_v2.classify import SEASON_KO, classify  # noqa: E402
from app.services.analysis_v2.features import ColorFeatures, LabColor  # noqa: E402
from app.services.analysis_v2.landmarks import FaceNotFoundError, extract_regions  # noqa: E402
from app.services.analysis_v2.lighting import correct_lighting  # noqa: E402
from app.services.analysis_v2.pipeline import analyze_rgb  # noqa: E402
from app.services.analysis_v2.youcam_tones import (  # noqa: E402
    UNITS_PER_CALL,
    YouCamError,
    analyze_tones,
    crop_for_youcam,
)

OUT_DIR = Path("validation_output")
CACHE_DIR = OUT_DIR / "cache"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_ALIASES = {"봄": "spring", "여름": "summer", "가을": "autumn", "겨울": "winter"}


def lab_from_hex(hex_color: str | None) -> LabColor | None:
    if not hex_color:
        return None
    h = hex_color.lstrip("#")
    return LabColor.from_rgb(np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64))


def delta_e(c1: dict | LabColor | None, c2: LabColor | None) -> float | None:
    """CIE76 색 차이 (Lab 공간 거리)."""
    if c1 is None or c2 is None:
        return None
    a = c1 if isinstance(c1, dict) else c1.__dict__
    return round(float(np.sqrt((a["L"] - c2.L) ** 2 + (a["a"] - c2.a) ** 2 + (a["b"] - c2.b) ** 2)), 1)


def youcam_season(colors: dict) -> str:
    skin, lips, eye = lab_from_hex(colors.get("skin_color")), lab_from_hex(colors.get("lip_color")), lab_from_hex(colors.get("eye_color"))
    feats = ColorFeatures(skin=skin, lips=lips, iris=eye, contrast=round(skin.L - eye.L, 1) if eye else None, skin_pixel_count=0)
    return classify(feats).season


def load_labels(folder: Path) -> dict[str, str]:
    path = folder / "labels.csv"
    if not path.exists():
        return {}
    labels = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
                season = LABEL_ALIASES.get(row[1].strip(), row[1].strip().lower())
                if season in SEASON_KO:
                    labels[row[0].strip()] = season
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="YouCam을 호출하지 않는다")
    parser.add_argument("--limit", type=int, default=None, help="새로 호출할 최대 사진 수")
    parser.add_argument("--yes", action="store_true", help="유닛 사용 확인 질문을 건너뛴다")
    args = parser.parse_args()

    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        print("사진(JPG/PNG)을 찾지 못했어요.")
        return
    labels = load_labels(args.folder)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 우리 분석 + YouCam에 보낼 사진 준비
    items = []
    for p in paths:
        rgb = np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        try:
            ours = analyze_rgb(rgb)
        except (FaceNotFoundError, ValueError) as exc:
            print(f"[건너뜀] {p.name}: {exc}")
            continue
        lit = correct_lighting(rgb)
        jpeg = crop_for_youcam(lit.image, tuple(extract_regions(lit.image).face_box))
        key = hashlib.sha1(jpeg).hexdigest()[:16]
        items.append({"path": p, "ours": ours, "jpeg": jpeg, "cache": CACHE_DIR / f"{key}.json"})

    todo = [it for it in items if not it["cache"].exists()]
    if args.limit is not None:
        todo = todo[: args.limit]
    cost = len(todo) * UNITS_PER_CALL
    print(f"\n사진 {len(items)}장 중 YouCam 새로 호출 {len(todo)}장 → 약 {cost}유닛 (저장된 결과 {len(items) - len([i for i in items if not i['cache'].exists()])}장 재사용)")

    # 2) YouCam 호출
    if todo and not args.dry_run:
        if not args.yes and input(f"{cost}유닛을 사용할까요? (y/n) ").strip().lower() != "y":
            print("취소했어요. --dry-run으로 우리 분석만 볼 수 있어요.")
            return
        settings = get_settings()
        for it in todo:
            try:
                colors = analyze_tones(it["jpeg"], settings.perfectcorp_api_key, settings.perfectcorp_base_url)
            except YouCamError as exc:
                print(f"[YouCam 실패] {it['path'].name}: {exc}")
                if "키" in str(exc) or "유닛" in str(exc):
                    break
                continue
            it["cache"].write_text(json.dumps(colors, ensure_ascii=False), encoding="utf-8")
            print(f"[YouCam 완료] {it['path'].name}")

    # 3) 비교
    rows = []
    for it in items:
        ours = it["ours"]
        f = ours["features"]
        row = {
            "파일": it["path"].name,
            "조명보정": ours["lighting"]["method"],
            "우리_계절": ours["season"],
            "우리_신뢰도": ours["confidence"],
            "우리_피부": f["skin"]["hex"],
            "우리_입술": f["lips"]["hex"] if f["lips"] else "",
            "우리_눈": f["iris"]["hex"] if f["iris"] else "",
            "정답": labels.get(it["path"].name, ""),
        }
        if it["cache"].exists():
            yc = json.loads(it["cache"].read_text(encoding="utf-8"))
            row |= {
                "유캠_피부": yc.get("skin_color", ""),
                "유캠_입술": yc.get("lip_color", ""),
                "유캠_눈": yc.get("eye_color", ""),
                "유캠_머리": yc.get("hair_color_name", ""),
                "피부_ΔE": delta_e(f["skin"], lab_from_hex(yc.get("skin_color"))),
                "입술_ΔE": delta_e(f["lips"], lab_from_hex(yc.get("lip_color"))),
                "눈_ΔE": delta_e(f["iris"], lab_from_hex(yc.get("eye_color"))),
                "유캠색_계절": youcam_season(yc),
            }
        rows.append(row)

    OUT_DIR.mkdir(exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"\n자세한 표: {OUT_DIR / 'results.csv'}")


def _mean(values: list) -> str:
    vals = [v for v in values if v is not None and v != ""]
    return f"{np.mean(vals):.1f} (사진 {len(vals)}장)" if vals else "-"


def _rate(pairs: list[tuple[str, str]]) -> str:
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return "-"
    hit = sum(a == b for a, b in pairs)
    return f"{hit}/{len(pairs)} ({hit / len(pairs):.0%})"


def print_summary(rows: list[dict]) -> None:
    print("\n===== 검증 요약 =====")
    print("① 측정 차이 ΔE (작을수록 YouCam과 비슷, 5 미만이면 양호)")
    print(f"   피부  평균 {_mean([r.get('피부_ΔE') for r in rows])}")
    print(f"   입술  평균 {_mean([r.get('입술_ΔE') for r in rows])}")
    print(f"   눈    평균 {_mean([r.get('눈_ΔE') for r in rows])}")
    print("② 분류 일치 (우리 색 vs YouCam 색 → 같은 계절?)")
    print(f"   {_rate([(r['우리_계절'], r.get('유캠색_계절', '')) for r in rows])}")
    if any(r["정답"] for r in rows):
        print("③ 정답(진단천 판정)과 비교")
        print(f"   우리 분석      {_rate([(r['우리_계절'], r['정답']) for r in rows])}")
        print(f"   YouCam 색 기반 {_rate([(r.get('유캠색_계절', ''), r['정답']) for r in rows])}")
    no_card = sum(r["조명보정"] == "none" for r in rows)
    if no_card:
        print(f"※ 기준 카드 없이 찍힌 사진 {no_card}장은 조명 보정이 안 돼 결과가 흔들릴 수 있어요.")


if __name__ == "__main__":
    main()
