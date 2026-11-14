import numpy as np
import pytest

from app.services.analysis_v2.landmarks import FaceNotFoundError, extract_regions


def test_no_face_raises():
    blank = np.full((400, 400, 3), 200, dtype=np.uint8)
    with pytest.raises(FaceNotFoundError):
        extract_regions(blank)
