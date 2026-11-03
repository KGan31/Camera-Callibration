"""Real-image calibration driver: images -> corners -> our calibration, plus the OpenCV baseline.

    views = detect_views(model, paths)            # our corners (marker prediction + sub-pixel refine)
    ours  = calibrate_full(views, size)           # synthetic_calib: H -> Zhang -> joint refinement
    cv    = opencv_calibrate(views, size)         # calibrateCamera on the SAME corners
    ref   = opencv_charuco_views(paths, ...)      # OpenCV's own ChArUco corners (standard board only)
Both calibrations use the same model: zero skew, k1, k2, no tangential, k3 fixed.
"""
import os
import numpy as np, cv2
from calib.predict import predict_corners
from calib.refine import refine_corner

aruco = cv2.aruco
# Phone photos carry an EXIF rotation tag; cv2.imread would rotate some of them and mix portrait/landscape
# frames, which breaks a single camera matrix. Always read the raw sensor pixels.
READ_FLAGS = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
MAX_SHIFT_PX = 3.0      # reject a refined corner that wandered further than this from its prediction


def auto_window(width):
    """Half-window for sub-pixel refinement, scaled with resolution (12 px at 4096 wide, 6 px at 1920)."""
    return max(5, int(round(width / 340)))


def detect_views(model, paths, w=None, max_shift=MAX_SHIFT_PX, min_corners=4, min_eig=None):
    """One view per usable image: {name, obj_pts (N,2) metres, img_pts (N,2) px, ids, n_pred, n_rejected}."""
    det = aruco.ArucoDetector(model.dictionary, aruco.DetectorParameters())
    views, skipped = [], []
    for p in paths:
        gray = cv2.imread(p, READ_FLAGS)
        mc, mids, _ = det.detectMarkers(gray)
        if mids is None or len(mids) < 1:
            skipped.append((p, "no markers")); continue
        pred = predict_corners(model, mc, mids)
        ww = auto_window(gray.shape[1]) if w is None else w
        g = gray.astype(np.float32)
        Gx, Gy = cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(g, cv2.CV_32F, 0, 1)
        ids, pts, eigs = [], [], []
        for cid, q0 in pred.items():
            q, lam = refine_corner(Gx, Gy, q0, w=ww)
            if np.linalg.norm(q - q0) < max_shift and (min_eig is None or lam > min_eig):
                ids.append(cid); pts.append(q); eigs.append(lam)
        if len(ids) < min_corners:
            skipped.append((p, f"only {len(ids)} corners")); continue
        views.append(dict(name=os.path.basename(p), ids=np.array(ids),
                          obj_pts=model.corners[ids][:, :2].astype(np.float64),
                          img_pts=np.array(pts, dtype=np.float64),
                          n_markers=len(mids), n_pred=len(pred), n_rejected=len(pred) - len(ids)))
    return views, skipped, gray.shape[::-1]


def opencv_charuco_views(paths, squares=(10, 7), square_len=0.035, marker_len=0.026,
                         dict_id=aruco.DICT_5X5_100, min_corners=4):
    d = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard(squares, square_len, marker_len, d)
    det = aruco.CharucoDetector(board)
    allc = board.getChessboardCorners()
    views = []
    for p in paths:
        gray = cv2.imread(p, READ_FLAGS)
        cc, cids, _, _ = det.detectBoard(gray)
        if cids is None or len(cids) < min_corners:
            continue
        ids = cids.flatten()
        views.append(dict(name=os.path.basename(p), ids=ids, obj_pts=allc[ids][:, :2].astype(np.float64),
                          img_pts=cc.reshape(-1, 2).astype(np.float64)))
    return views


def opencv_calibrate(views, image_size):
    """OpenCV calibrateCamera, same model as ours. Returns dict of results."""
    obj = [np.column_stack([v["obj_pts"], np.zeros(len(v["obj_pts"]))]).astype(np.float32) for v in views]
    img = [np.asarray(v["img_pts"], np.float32).reshape(-1, 1, 2) for v in views]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, tuple(image_size), None, None, flags=flags)
    errs = []
    for o, i, r, t in zip(obj, img, rv, tv):
        pr, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(np.linalg.norm(pr.reshape(-1, 2) - i.reshape(-1, 2), axis=1))
    allerr = np.concatenate(errs)
    return dict(K=K, dist=dist.ravel()[:2], mean_error=float(allerr.mean()), rms_error=float(rms),
                view_errors=np.array([e.mean() for e in errs]), rvecs=rv, tvecs=tv)


def summarise(label, K, dist, mean_e, rms_e, n_views, n_pts):
    return dict(label=label, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=dist[0], k2=dist[1],
                mean_error=mean_e, rms_error=rms_e, n_views=n_views, n_points=n_pts)
