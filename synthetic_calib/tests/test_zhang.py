"""Checks for synthetic_calib/zhang.py.
Run from the repo folder with:   python3 -m pytest synthetic_calib/tests
or without pytest:               python3 -m synthetic_calib.tests.test_zhang
"""
import numpy as np
import cv2
from synthetic_calib.synthetic import make_views, true_camera_matrix, IMAGE_W, IMAGE_H
from synthetic_calib.zhang import zhang_intrinsics, poses_from_homographies, calibrate_closed_form

SIZE = (IMAGE_W, IMAGE_H)
K_TRUE = true_camera_matrix()


def true_homographies(views):
    """The exact homographies: H = K [r1 r2 t] for each view's true pose."""
    return [K_TRUE @ np.column_stack([v["R_true"][:, 0], v["R_true"][:, 1], v["t_true"]]) for v in views]


def test_exact_answer_without_noise():
    _, views = make_views(n_views=10, noise_px=0, seed=1)
    K, poses, _ = calibrate_closed_form(views, SIZE)
    assert np.abs(K - K_TRUE).max() < 1e-6                         # fx, fy, cx, cy all exact
    for (R, t), v in zip(poses, views):
        assert np.abs(R - v["R_true"]).max() < 1e-8                # and so is every board position
        assert np.abs(t - v["t_true"]).max() < 1e-8


def test_exact_from_exact_homographies():
    """Skips the corner-to-homography step entirely: pure Zhang maths."""
    _, views = make_views(n_views=6, noise_px=0, seed=2)
    K = zhang_intrinsics(true_homographies(views), SIZE)
    assert np.abs(K - K_TRUE).max() < 1e-6


def test_image_size_does_not_change_exact_answer():
    """image_size is only used to rescale numbers; it must not change an exact result."""
    _, views = make_views(n_views=6, noise_px=0, seed=3)
    Hs = true_homographies(views)
    assert np.abs(zhang_intrinsics(Hs, SIZE) - zhang_intrinsics(Hs, (4000, 3000))).max() < 1e-6


def test_noisy_answer_is_close_to_truth():
    """0.3 px noise, 30 photos: typical error of the rough closed-form answer is ~1 px."""
    fx_err, fy_err, cx_err, cy_err = [], [], [], []
    for seed in range(20):
        _, views = make_views(n_views=30, noise_px=0.3, seed=100 + seed)
        K, _, _ = calibrate_closed_form(views, SIZE)
        fx_err.append(abs(K[0, 0] - K_TRUE[0, 0])); fy_err.append(abs(K[1, 1] - K_TRUE[1, 1]))
        cx_err.append(abs(K[0, 2] - K_TRUE[0, 2])); cy_err.append(abs(K[1, 2] - K_TRUE[1, 2]))
    assert np.median(fx_err) < 3 and np.median(fy_err) < 3
    assert np.median(cx_err) < 2 and np.median(cy_err) < 2


def test_more_photos_give_better_answers():
    def avg_error(n):
        errs = []
        for seed in range(40):
            _, views = make_views(n_views=n, noise_px=0.3, seed=500 + seed)
            K, _, _ = calibrate_closed_form(views, SIZE)
            errs.append(abs(K[0, 0] - 1050) + abs(K[1, 1] - 1040))
        return np.mean(errs)
    assert avg_error(30) < avg_error(10) < avg_error(3)


def test_agrees_with_opencv():
    """Ours is the rough closed-form answer; OpenCV's is already polished. Expect small gaps."""
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    for seed in range(8):
        _, views = make_views(n_views=15, noise_px=0.3, seed=2000 + seed)
        ours, _, _ = calibrate_closed_form(views, SIZE)
        obj = [np.column_stack([v["obj_pts"], np.zeros(54)]).astype(np.float32) for v in views]
        img = [v["img_pts"].astype(np.float32).reshape(-1, 1, 2) for v in views]
        _, cvK, _, _, _ = cv2.calibrateCamera(obj, img, SIZE, None, None, flags=flags)
        assert abs(ours[0, 0] - cvK[0, 0]) / cvK[0, 0] < 0.01      # focal length within 1%
        assert abs(ours[1, 1] - cvK[1, 1]) / cvK[1, 1] < 0.01
        assert abs(ours[0, 2] - cvK[0, 2]) < 4 and abs(ours[1, 2] - cvK[1, 2]) < 4   # centre within 4 px


def test_poses_are_valid_even_with_noise():
    _, views = make_views(n_views=12, noise_px=0.5, seed=7)
    K, poses, _ = calibrate_closed_form(views, SIZE)
    for R, t in poses:
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)         # a true rotation...
        assert np.isclose(np.linalg.det(R), 1.0)                   # ...not a mirror image
        assert t[2] > 0                                            # board is in front of the camera


def test_sign_of_homography_does_not_matter():
    """A homography is only defined up to scale, including a minus sign."""
    _, views = make_views(n_views=6, noise_px=0, seed=8)
    Hs = true_homographies(views)
    K = zhang_intrinsics([-H for H in Hs], SIZE)
    assert np.abs(K - K_TRUE).max() < 1e-6
    for (R, t), v in zip(poses_from_homographies(K, [-H for H in Hs]), views):
        assert t[2] > 0 and np.abs(t - v["t_true"]).max() < 1e-8


def test_too_few_views_refused():
    _, views = make_views(n_views=2, noise_px=0, seed=9)
    try:
        calibrate_closed_form(views, SIZE)
    except ValueError as e:
        assert "at least 3" in str(e)
    else:
        raise AssertionError("should have been refused")


def test_identical_orientation_is_refused():
    """Board held flat-on in every photo (only moved closer/farther/sideways): the photos
    all say the same thing, so the focal length cannot be found. Must be refused clearly."""
    Hs = [K_TRUE @ np.column_stack([[1, 0, 0], [0, 1, 0], t])
          for t in ([0.0, 0.0, 0.6], [0.05, -0.02, 0.8], [-0.04, 0.03, 0.7], [0.0, 0.05, 0.9])]
    try:
        zhang_intrinsics(Hs, SIZE)
    except ValueError as e:
        assert "too similar" in str(e)
    else:
        raise AssertionError("should have been refused")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("PASS", f.__name__)
    print(f"\nAll {len(tests)} checks passed.")
