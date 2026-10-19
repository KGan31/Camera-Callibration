"""Checks for synthetic_calib/homography.py.
Run from the repo folder with:   python3 -m pytest synthetic_calib/tests
or without pytest:               python3 -m synthetic_calib.tests.test_homography
"""
import numpy as np
import cv2
from synthetic_calib.synthetic import make_views
from synthetic_calib.homography import (apply_homography, reprojection_errors, normalize_points,
                                        dlt_homography, estimate_homography, has_four_in_general_position)


def true_homography(K, view):
    """The exact answer: H = K [r1 r2 t] (first two columns of the turn, plus the shift)."""
    R, t = view["R_true"], view["t_true"]
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H / H[2, 2]


def test_normalize_points():
    pts = np.random.default_rng(0).uniform(500, 900, (30, 2))
    T, new = normalize_points(pts)
    assert np.allclose(new.mean(axis=0), 0, atol=1e-12)                       # centred
    assert np.isclose(np.linalg.norm(new, axis=1).mean(), np.sqrt(2))         # average distance sqrt(2)


def test_recovers_exact_answer_without_noise():
    K, views = make_views(n_views=10, noise_px=0, seed=1)
    for v in views:
        H = estimate_homography(v["obj_pts"], v["img_pts"])
        assert np.abs(H - true_homography(K, v)).max() < 1e-6 * np.abs(H).max()
        assert reprojection_errors(H, v["obj_pts"], v["img_pts"]).max() < 1e-8


def test_matches_opencv():
    """Our H and OpenCV's H must send the board to the same places in the photo."""
    _, views = make_views(n_views=10, noise_px=0.3, seed=2)
    for v in views:
        ours = estimate_homography(v["obj_pts"], v["img_pts"])
        theirs, _ = cv2.findHomography(v["obj_pts"], v["img_pts"], 0)
        a = apply_homography(ours, v["obj_pts"])
        b = apply_homography(theirs, v["obj_pts"])
        assert np.abs(a - b).max() < 0.01          # within 0.01 pixel


def test_polishing_never_hurts():
    _, views = make_views(n_views=10, noise_px=0.3, seed=3)
    for v in views:
        before = dlt_homography(v["obj_pts"], v["img_pts"])
        after = estimate_homography(v["obj_pts"], v["img_pts"])
        e0 = np.sqrt(np.mean(reprojection_errors(before, v["obj_pts"], v["img_pts"]) ** 2))
        e1 = np.sqrt(np.mean(reprojection_errors(after, v["obj_pts"], v["img_pts"]) ** 2))
        assert e1 <= e0 + 1e-12


def test_error_matches_noise_level():
    """0.3 px noise sideways AND up-down gives a 2D miss of about 0.3 x 1.41 = 0.42 px.
    A little less is left over (about 0.41) because the fitted H soaks up a bit of the noise."""
    _, views = make_views(n_views=30, noise_px=0.3, seed=4)
    rms = [np.sqrt(np.mean(reprojection_errors(estimate_homography(v["obj_pts"], v["img_pts"]),
                                               v["obj_pts"], v["img_pts"]) ** 2)) for v in views]
    assert 0.37 < np.mean(rms) < 0.45


def test_works_with_only_part_of_the_board():
    """A partly hidden board is fine: 8 random corners still give the exact rule (no noise)."""
    K, views = make_views(n_views=5, noise_px=0, seed=5)
    rng = np.random.default_rng(0)
    for v in views:
        pick = rng.choice(54, 8, replace=False)
        H = estimate_homography(v["obj_pts"][pick], v["img_pts"][pick])
        assert np.abs(H - true_homography(K, v)).max() < 1e-6 * np.abs(H).max()


def test_units_do_not_matter():
    """Measuring the board in millimetres instead of metres must give the same mapping."""
    _, views = make_views(n_views=5, noise_px=0.3, seed=6)
    for v in views:
        Hm = estimate_homography(v["obj_pts"], v["img_pts"])
        Hmm = estimate_homography(v["obj_pts"] * 1000, v["img_pts"])
        a = apply_homography(Hm, v["obj_pts"])
        b = apply_homography(Hmm, v["obj_pts"] * 1000)
        assert np.abs(a - b).max() < 1e-4


def test_bad_input_is_refused():
    raised = 0
    for src, dst in [(np.zeros((3, 2)), np.zeros((3, 2))),                        # too few points
                     (np.column_stack([np.arange(6.0), np.arange(6.0)]),           # all in a line
                      np.column_stack([np.arange(6.0) * 2, np.arange(6.0) * 3])),
                     (np.ones((6, 2)), np.ones((6, 2)))]:                          # all the same point
        try:
            estimate_homography(src, dst)
        except ValueError:
            raised += 1
    assert raised == 3


def test_mostly_collinear_points_are_refused_clearly():
    """Real case found while testing: 5 corners on one line + 1 off it. Not enough to
    pin down a homography, so we must say so clearly (not fail with a maths error)."""
    src = np.array([[0, 0], [0.07, 0], [0.105, 0], [0.175, 0], [0.28, 0], [0.175, 0.035]])
    dst = np.array([[572.5, 270.2], [645.8, 315.0], [683.8, 333.5], [773.2, 367.8],
                    [879.9, 437.5], [751.8, 409.3]])
    assert not has_four_in_general_position(src)
    try:
        estimate_homography(src, dst)
    except ValueError as e:
        assert "general position" in str(e)
    else:
        raise AssertionError("should have been refused")


def test_terrible_data_never_crashes():
    """Few corners + huge noise (far worse than any real detector). Each try must either
    return a finite H or be refused with one of OUR clear messages. Nothing else."""
    rng = np.random.default_rng(1)
    for noise, n_pts in [(0.3, 54), (2.0, 54), (5.0, 54), (0.3, 6), (2.0, 6), (5.0, 6)]:
        _, views = make_views(n_views=300, noise_px=noise, seed=12)
        for v in views:
            pick = np.sort(rng.choice(54, n_pts, replace=False))
            try:
                H = estimate_homography(v["obj_pts"][pick], v["img_pts"][pick])
            except ValueError as e:
                assert "general position" in str(e) or "straight line" in str(e), str(e)
                continue
            assert np.all(np.isfinite(H))


def test_spread_check():
    pts, _ = make_views(n_views=1, seed=0)[1][0]["obj_pts"], None
    assert has_four_in_general_position(pts)                       # the full board
    assert has_four_in_general_position(pts[[0, 1, 9, 10]])        # one small square: 4 corners
    assert not has_four_in_general_position(pts[:9])               # a whole row = one line


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("PASS", f.__name__)
    print(f"\nAll {len(tests)} checks passed.")
