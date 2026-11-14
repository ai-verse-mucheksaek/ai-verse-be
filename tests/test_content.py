import re

from app.services.analysis_v2.content import SEASONS, build_result_content, load_content

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
REQUIRED = ["name_ko", "keywords", "summary", "palette", "makeup", "clothes", "clothes_avoid", "hair",
            "accessory_metal", "accessory_tip", "booth_lip"]


def test_all_seasons_have_required_fields():
    data = load_content()
    for season in SEASONS:
        for key in REQUIRED:
            assert key in data[season], f"{season}에 {key}가 없음"


def test_colors_are_valid_hex():
    for season, c in load_content().items():
        for item in c["palette"] + c["makeup"]:
            assert HEX.match(item["hex"]), f"{season}의 {item['name']} 색 형식 오류: {item['hex']}"


def test_makeup_has_lip_blush_eye():
    for season, c in load_content().items():
        assert {m["category"] for m in c["makeup"]} == {"lip", "blush", "eye"}, season


def test_build_result_includes_second_season():
    scores = {"spring": 0.1, "summer": 0.5, "autumn": 0.05, "winter": 0.35}
    result = build_result_content("summer", scores)
    assert result["season_ko"] == "여름 쿨톤"
    assert result["second_season"]["season"] == "winter"
    assert result["accessory"]["metal"] == "silver"
