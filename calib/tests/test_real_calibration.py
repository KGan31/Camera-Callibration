"""Real-photo regression tests (skipped when the images are not present)."""
import glob, os
import numpy as np
import pytest
from calib.board import BoardModel
from calib.custom_board import CustomBoardModel
from calib.real_calibrate import detect_views, opencv_calibrate, opencv_charuco_views
from synthetic_calib.joint_refine import calibrate_full

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
KEY = lambda s: int(os.path.basename(s).split("_")[1].split(".")[0])


def _paths(tag):
    p = sorted(glob.glob(os.path.join(ROOT, "data", f"set_{tag}", "*.jpeg")), key=KEY)
    if not p:
        pytest.skip("images not available")
    return p


def test_set_a_matches_opencv():
    """Timeline acceptance on Set A: f within 1%, principal point within 2 px, mean error within 0.05 px."""
    paths = _paths("a")
    views, _, size = detect_views(BoardModel(), paths)
    assert size == (4096, 3072)                       # raw sensor frame: EXIF rotation must be ignored
    ours = calibrate_full(views, size)
    cv_own = opencv_calibrate(opencv_charuco_views(paths), size)
    cv_same = opencv_calibrate(views, size)
    assert abs(ours.K[0, 0] / cv_own["K"][0, 0] - 1) < 0.01
    assert abs(ours.K[1, 1] / cv_own["K"][1, 1] - 1) < 0.01
    assert abs(ours.K[0, 2] - cv_own["K"][0, 2]) < 2 and abs(ours.K[1, 2] - cv_own["K"][1, 2]) < 2
    assert abs(ours.mean_error - cv_same["mean_error"]) < 0.05    # same corners, same model
    assert abs(ours.mean_error - cv_own["mean_error"]) < 0.5      # vs OpenCV's own corners (different detector)


def test_custom_board_set_b_calibrates_and_matches_opencv():
    paths = _paths("b")
    views, _, size = detect_views(CustomBoardModel(), paths)
    assert len(views) == len(paths) and all(len(v["ids"]) == 20 for v in views)
    ours = calibrate_full(views, size)
    cv = opencv_calibrate(views, size)
    assert abs(ours.K[0, 0] / cv["K"][0, 0] - 1) < 0.01
    assert abs(ours.K[0, 2] - cv["K"][0, 2]) < 2 and abs(ours.K[1, 2] - cv["K"][1, 2]) < 2


def test_partial_views_still_used():
    """Set D has the board half out of frame: views with few markers must still contribute."""
    views, skipped, _ = detect_views(CustomBoardModel(), _paths("d"))
    assert len(views) >= 18
    assert min(len(v["ids"]) for v in views) < 20 and min(len(v["ids"]) for v in views) >= 4
