import pytest

from app.services.analysis_v2.classify import classify
from app.services.analysis_v2.features import ColorFeatures, LabColor


def _lab(L, C, h):
    import math

    a = C * math.cos(math.radians(h))
    b = C * math.sin(math.radians(h))
    return LabColor(L=L, a=round(a, 1), b=round(b, 1), C=C, h=h, hex="#000000")


def _features(skin_L, skin_C, skin_h, lip_h, iris_L):
    return ColorFeatures(
        skin=_lab(skin_L, skin_C, skin_h),
        lips=_lab(55, 30, lip_h),
        iris=_lab(iris_L, 10, 60),
        contrast=round(skin_L - iris_L, 1),
        skin_pixel_count=1000,
    )


@pytest.mark.parametrize(
    "name, feats, expected",
    [
        ("밝고 선명한 노란 피부, 코랄 입술", _features(80, 26, 72, 38, 40), "spring"),
        ("밝고 차분한 분홍 피부, 대비 낮음", _features(82, 14, 48, 12, 45), "summer"),
        ("깊고 차분한 노란 피부", _features(60, 17, 70, 35, 30), "autumn"),
        ("깊은 분홍 피부, 진한 눈동자로 대비 큼", _features(68, 16, 48, 12, 12), "winter"),
    ],
)
def test_typical_profiles(name, feats, expected):
    assert classify(feats).season == expected, name


def test_scores_sum_to_one():
    result = classify(_features(75, 20, 60, 25, 30))
    assert abs(sum(result.season_scores.values()) - 1.0) < 0.01


def test_missing_lips_and_iris_still_works():
    feats = ColorFeatures(skin=_lab(80, 26, 72), lips=None, iris=None, contrast=None, skin_pixel_count=500)
    assert classify(feats).undertone == "warm"
