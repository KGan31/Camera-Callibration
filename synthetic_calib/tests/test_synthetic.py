"""Checks for synthetic_calib/synthetic.py.
Run from the repo folder with:   python3 -m pytest synthetic_calib/tests
or without pytest:               python3 -m synthetic_calib.tests.test_synthetic
"""
import numpy as np
import cv2
from synthetic_calib.synthetic import (make_views, board_points, project_points,
                             IMAGE_W, IMAGE_H, BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)


def test_board_grid():
    pts, ids = board_points()
    assert pts.shape == (BOARD_COLS * BOARD_ROWS, 2)          # 54 corners
    assert np.allclose(pts[0], [0, 0])                        # starts at the origin
    assert np.isclose(pts[1, 0] - pts[0, 0], SQUARE_SIZE)     # 35 mm apart
    assert np.array_equal(ids, np.arange(54))


def test_projection_matches_opencv():
    """Our projection formula must agree with OpenCV's, even with lens bending."""
    K, views = make_views(n_views=3, noise_px=0, seed=1)
    pts, _ = board_points()
    k1, k2 = -0.2, 0.05
    for v in views:
        ours = project_points(K, v["R_true"], v["t_true"], pts, k1, k2)
        rvec, _ = cv2.Rodrigues(v["R_true"])
        obj3d = np.column_stack([pts, np.zeros(len(pts))])
        theirs, _ = cv2.projectPoints(obj3d, rvec, v["t_true"], K,
                                      np.array([k1, k2, 0, 0.0]))
        assert np.abs(ours - theirs.reshape(-1, 2)).max() < 1e-6


def test_all_corners_inside_picture():
    _, views = make_views(n_views=30, seed=2)
    for v in views:
        p = v["img_pts"]
        assert p[:, 0].min() > 0 and p[:, 0].max() < IMAGE_W
        assert p[:, 1].min() > 0 and p[:, 1].max() < IMAGE_H


def test_noise_size():
    _, views = make_views(n_views=20, noise_px=0.3, seed=3)
    err = np.concatenate([(v["img_pts"] - v["img_pts_clean"]).ravel() for v in views])
    assert abs(err.std() - 0.3) < 0.03


def test_tilt_in_range():
    _, views = make_views(n_views=30, seed=4, min_tilt_deg=5, max_tilt_deg=40)
    tilts = [v["tilt_deg"] for v in views]
    assert min(tilts) >= 5 and max(tilts) <= 40
    # tilt computed from the pose itself must agree with the stored value
    for v in views:
        assert np.isclose(np.rad2deg(np.arccos(v["R_true"][2, 2])), v["tilt_deg"])


def test_same_seed_gives_same_data():
    _, a = make_views(n_views=5, seed=7)
    _, b = make_views(n_views=5, seed=7)
    _, c = make_views(n_views=5, seed=8)
    assert np.array_equal(a[0]["img_pts"], b[0]["img_pts"])
    assert not np.array_equal(a[0]["img_pts"], c[0]["img_pts"])


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("PASS", f.__name__)
    print(f"\nAll {len(tests)} checks passed.")
