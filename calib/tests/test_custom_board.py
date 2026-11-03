import glob, os
import numpy as np, cv2
import pytest
from calib.custom_board import CustomBoardModel
from calib.predict import predict_corners

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _ids(m):
    return {c for i in m.marker_xy for c in m.square_corner_ids(i)}


def test_marker_count_and_ids():
    m = CustomBoardModel()
    assert sorted(m.marker_xy) == list(range(20))
    assert m.marker_square[0] == (0, 1) and m.marker_square[8] == (6, 1)
    assert m.marker_square[16] == (3, 4) and m.marker_square[19] == (5, 4)


def test_marker_geometry():
    m = CustomBoardModel()
    for xy in m.marker_xy.values():
        assert np.isclose(np.linalg.norm(xy[1] - xy[0]), 0.07)
        assert np.isclose(np.linalg.norm(xy[3] - xy[0]), 0.07)


def test_corner_counts():
    assert len(_ids(CustomBoardModel())) == 20                               # 5 patches x 4 interior corners
    m = CustomBoardModel(diagonal_junctions=True)
    assert len(_ids(m)) == 24                                                # + 4 diagonal junctions
    xy = {tuple(np.round(m.corners[c][:2] / 0.1).astype(int)) for c in _ids(m)}
    assert {(3, 3), (6, 3), (3, 6), (6, 6)} <= xy


def test_corner_validity_examples():
    m = CustomBoardModel()
    assert m.corner_id(1, 1) is not None       # inside top-left patch
    assert m.corner_id(3, 3) is not None       # corner patch meets centre patch diagonally
    assert m.corner_id(4, 1) is None           # inside the top cut-out block
    assert m.corner_id(0, 3) is None and m.corner_id(9, 9) is None   # outer edge


@pytest.mark.parametrize("pattern", ["set_b/B_*.jpeg", "set_d/D_*.jpeg"])
def test_real_images_fit_model(pattern):
    files = sorted(glob.glob(os.path.join(ROOT, "data", pattern)))
    if not files:
        pytest.skip("images not available")
    m = CustomBoardModel()
    det = cv2.aruco.ArucoDetector(m.dictionary, cv2.aruco.DetectorParameters())
    ok = 0
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        c, ids, _ = det.detectMarkers(im)
        if ids is None or len(ids) < 4:
            continue
        src = np.vstack([m.marker_xy[int(i)] for i in ids.flatten()])
        dst = np.vstack([x.reshape(4, 2) for x in c])
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        err = np.linalg.norm(cv2.perspectiveTransform(src[None], H)[0] - dst, axis=1)
        assert np.median(err) < 5.0, (f, np.median(err))
        assert len(predict_corners(m, c, ids)) >= 1
        ok += 1
    assert ok >= len(files) // 2
