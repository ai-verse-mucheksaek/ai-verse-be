"""결과 카드에 들어갈 계절별 내용(app/data/season_content.json)을 불러와 분석 결과에 붙인다."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CONTENT_PATH = Path(__file__).resolve().parents[2] / "data" / "season_content.json"
SEASONS = ("spring", "summer", "autumn", "winter")
METAL_KO_TO_KEY = {"골드": "gold", "실버": "silver"}


@lru_cache(maxsize=1)
def load_content() -> dict:
    data = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k in SEASONS}


def content_for(season: str) -> dict:
    return load_content()[season]


def build_result_content(season: str, season_scores: dict[str, float]) -> dict:
    """1위 계절의 내용 + 2순위 계절(점수 차이가 작을 때 참고용)을 묶어 돌려준다."""
    c = content_for(season)
    ranked = sorted(season_scores.items(), key=lambda kv: -kv[1])
    second, second_score = ranked[1]
    return {
        "season": season,
        "season_ko": c["name_ko"],
        "score": season_scores[season],
        "summary": c["summary"],
        "keywords": c["keywords"],
        "palette": c["palette"],
        "makeup": c["makeup"],
        "clothes": {"best": c["clothes"], "avoid": c["clothes_avoid"]},
        "hair": c["hair"],
        "accessory": {"metal": METAL_KO_TO_KEY.get(c["accessory_metal"], c["accessory_metal"]),
                      "metal_ko": c["accessory_metal"], "tip": c["accessory_tip"]},
        "booth_lip": c["booth_lip"],
        "second_season": {"season": second, "season_ko": content_for(second)["name_ko"], "score": second_score},
    }
