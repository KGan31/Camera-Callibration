"""Checks for synthetic_calib/joint_refine.py.
Run from the repo folder with:   python3 -m pytest synthetic_calib/tests
or without pytest:               python3 -m synthetic_calib.tests.test_joint_refine
(This one needs scipy, and takes about a minute because it runs real calibrations.)
"""
import numpy as np
import cv2
from synthetic_calib.synthetic import make_views, true_camera_matrix, IMAGE_W, IMAGE_H
from synthetic_calib.zhang import calibrate_closed_form
from synthetic_calib.joint_refine import (rodrigues, rotation_to_rvec, project, init_distortion,
                                          refine, calibrate_full, corner_weights)

SIZE = (IMAGE_W, IMAGE_H)
K_TRUE = true_camera_matrix()
K1, K2 = -0.18, 0.04                       # the pretend lens bending used in these checks
TRUTH = np.array([K_TRUE[0, 0], K_TRUE[1, 1], K_TRUE[0, 2], K_TRUE[1, 2]])


def intrinsics(K):
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])


def opencv_calibration(views):
    """OpenCV on the same corners: lens bending k1, k2 only (no tangential, no k3)."""
    obj = [np.column_stack([v["obj_pts"], np.zeros(len(v["obj_pts"]))]).astype(np.float32) for v in views]
    img = [v["img_pts"].astype(np.float32).reshape(-1, 1, 2) for v in views]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, SIZE, None, None, flags=flags)
    return rms, K, dist.ravel()


def test_rodrigues_matches_opencv():
    rng = np.random.default_rng(0)
    for angle in [1e-10, 1e-5, 0.3, 1.5, 3.0]:
        for _ in range(20):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * angle
            assert np.abs(rodrigues(v) - cv2.Rodrigues(v)[0]).max() < 1e-9


def test_rotation_round_trip_even_near_half_a_turn():
    rng = np.random.default_rng(1)
    for angle in [1e-10, 1e-5, 0.3, 1.5, 3.0, np.pi - 1e-4, np.pi - 1e-6]:
        for _ in range(20):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * angle
            R = rodrigues(v)
            assert np.abs(rodrigues(rotation_to_rvec(R)) - R).max() < 1e-7


def test_project_matches_opencv_with_lens_bending():
    _, views = make_views(n_views=3, noise_px=0, seed=1)
    for v in views:
        ours = project(K_TRUE, (K1, K2), v["R_true"], v["t_true"], v["obj_pts"])
        rvec, _ = cv2.Rodrigues(v["R_true"])
        obj3d = np.column_stack([v["obj_pts"], np.zeros(len(v["obj_pts"]))])
        theirs, _ = cv2.projectPoints(obj3d, rvec, v["t_true"], K_TRUE, np.array([K1, K2, 0, 0.0]))
        assert np.abs(ours - theirs.reshape(-1, 2)).max() < 1e-6


def test_distortion_first_guess_is_exact_with_true_camera_and_poses():
    _, views = make_views(n_views=8, noise_px=0, seed=2, k1=K1, k2=K2)
    poses = [(v["R_true"], v["t_true"]) for v in views]
    assert np.abs(init_distortion(K_TRUE, poses, views) - [K1, K2]).max() < 1e-8


def test_exact_recovery_with_lens_bending_and_no_noise():
    _, views = make_views(n_views=15, noise_px=0, seed=3, k1=K1, k2=K2)
    r = calibrate_full(views, SIZE)
    assert r.converged
    assert np.abs(intrinsics(r.K) - TRUTH).max() < 1e-3          # camera numbers
    assert np.abs(r.dist - [K1, K2]).max() < 1e-5               # lens bending
    assert r.mean_error < 1e-4                                  # corners land where they were seen


def test_exact_recovery_when_parts_of_the_board_are_hidden():
    """Each photo shows only 14 to 53 of the 54 corners (as with a partly hidden board)."""
    _, views = make_views(n_views=12, noise_px=0, seed=4, k1=K1, k2=K2)
    rng = np.random.default_rng(0)
    partial = []
    for v in views:
        pick = np.sort(rng.choice(54, int(rng.integers(14, 54)), replace=False))
        partial.append(dict(obj_pts=v["obj_pts"][pick], img_pts=v["img_pts"][pick]))
    r = calibrate_full(partial, SIZE)
    assert np.abs(intrinsics(r.K) - TRUTH).max() < 1e-3
    assert np.abs(r.dist - [K1, K2]).max() < 1e-5


def test_agrees_with_opencv_and_meets_the_project_targets():
    """The targets from the project plan: focal length within 1% of OpenCV's, image centre within
    2 px, and the average error within 0.05 px. (We are far inside all three.)"""
    for seed in range(5):
        _, views = make_views(n_views=15, noise_px=0.3, seed=100 + seed, k1=K1, k2=K2)
        ours = calibrate_full(views, SIZE)
        rms, cvK, cvdist = opencv_calibration(views)
        assert abs(ours.K[0, 0] - cvK[0, 0]) / cvK[0, 0] < 0.01
        assert abs(ours.K[1, 1] - cvK[1, 1]) / cvK[1, 1] < 0.01
        assert abs(ours.K[0, 2] - cvK[0, 2]) < 2 and abs(ours.K[1, 2] - cvK[1, 2]) < 2
        assert abs(ours.rms_error - rms) < 0.05
        # in fact much closer than the targets:
        assert abs(ours.K[0, 0] - cvK[0, 0]) / cvK[0, 0] < 0.001
        assert np.abs(ours.dist - cvdist[:2]).max() < 0.005


def test_refining_beats_the_rough_closed_form_answer():
    err_rough, err_refined = [], []
    for seed in range(8):
        _, views = make_views(n_views=15, noise_px=0.3, seed=200 + seed, k1=K1, k2=K2)
        r = calibrate_full(views, SIZE)
        err_rough.append(np.abs(intrinsics(r.K_closed_form) - TRUTH)[:2].mean())
        err_refined.append(np.abs(intrinsics(r.K) - TRUTH)[:2].mean())
    assert np.median(err_refined) < np.median(err_rough) / 3


def test_refining_never_makes_the_fit_worse():
    _, views = make_views(n_views=10, noise_px=0.3, seed=5, k1=K1, k2=K2)
    K0, poses0, _ = calibrate_closed_form(views, SIZE)
    d0 = init_distortion(K0, poses0, views)
    before = np.concatenate([np.linalg.norm(project(K0, d0, R, t, v["obj_pts"]) - v["img_pts"], axis=1)
                             for (R, t), v in zip(poses0, views)])
    r = refine(K0, d0, poses0, views)
    assert np.sqrt(np.mean(r.rms_error ** 2)) < np.sqrt(np.mean(before ** 2))


def test_leaving_out_the_lens_bending_is_clearly_worse_when_it_exists():
    _, views = make_views(n_views=15, noise_px=0.3, seed=6, k1=K1, k2=K2)
    with_k = calibrate_full(views, SIZE)
    without = calibrate_full(views, SIZE, estimate_distortion=False)
    assert np.all(without.dist == 0)
    assert without.mean_error > 1.5 * with_k.mean_error


def test_no_lens_bending_is_found_when_there_is_none():
    for seed in range(3):
        _, views = make_views(n_views=15, noise_px=0.3, seed=300 + seed)
        r = calibrate_full(views, SIZE)
        assert abs(r.dist[0]) < 0.05 and abs(r.dist[1]) < 0.15


def test_staged_and_direct_runs_agree():
    _, views = make_views(n_views=15, noise_px=0.3, seed=7, k1=K1, k2=K2)
    a = calibrate_full(views, SIZE, staged=True)
    b = calibrate_full(views, SIZE, staged=False)
    assert np.abs(intrinsics(a.K) - intrinsics(b.K)).max() < 0.5


def test_robust_loss_resists_wildly_wrong_corners():
    """15 corners (under 2%) are knocked about 35 px off. Plain least squares is dragged away;
    the robust version barely notices."""
    plain_err, robust_err = [], []
    for seed in range(4):
        _, views = make_views(n_views=15, noise_px=0.3, seed=400 + seed, k1=-0.15, k2=0.03)
        rng = np.random.default_rng(seed)
        for v in views[:5]:
            for i in rng.choice(54, 3, replace=False):
                v["img_pts"][i] += rng.choice([-1, 1], 2) * 25
        plain_err.append(np.abs(intrinsics(calibrate_full(views, SIZE, robust=False).K) - TRUTH).mean())
        robust_err.append(np.abs(intrinsics(calibrate_full(views, SIZE, robust=True).K) - TRUTH).mean())
    assert np.median(robust_err) < np.median(plain_err) / 2


def test_robust_loss_changes_nothing_on_clean_data():
    _, views = make_views(n_views=15, noise_px=0.3, seed=8, k1=K1, k2=K2)
    a = calibrate_full(views, SIZE, robust=False)
    b = calibrate_full(views, SIZE, robust=True)
    assert np.abs(intrinsics(a.K) - intrinsics(b.K)).max() < 0.5


def test_corner_weights():
    m = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
    assert np.all(corner_weights(m, "linear", 1.0) == 1)
    h = corner_weights(m, "huber", 1.0)
    assert np.allclose(h, [1, 1, 1, 0.5, 0.1])                   # trusted less as the miss grows past 1 px
    c = corner_weights(m, "cauchy", 1.0)
    assert c[0] == 1 and np.all(np.diff(c) < 0) and c[-1] < 0.02  # far-off corners are nearly ignored


def test_poses_and_errors_are_sensible():
    _, views = make_views(n_views=10, noise_px=0.3, seed=9, k1=K1, k2=K2)
    r = calibrate_full(views, SIZE)
    assert len(r.poses) == 10 and len(r.view_errors) == 10
    for R, t in r.poses:
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-8) and np.isclose(np.linalg.det(R), 1.0)
        assert t[2] > 0
    assert r.rms_error >= r.mean_error > 0
    assert 0.3 < r.mean_error < 0.45                              # about 0.3 px noise in each direction


def test_too_few_views_refused():
    _, views = make_views(n_views=2, noise_px=0, seed=1)
    try:
        calibrate_full(views, SIZE)
    except ValueError as e:
        assert "at least 3" in str(e)
    else:
        raise AssertionError("should have been refused")


if __name__ == "__main__":
    import time
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    t0 = time.time()
    for f in tests:
        f()
        print("PASS", f.__name__)
    print(f"\nAll {len(tests)} checks passed in {time.time() - t0:.0f} s.")
