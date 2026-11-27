"""Clean 3-Way Camera Calibration Benchmark: OpenCV vs Ours vs Lensboy on Set A.

Each method uses its own native corner detection and calibration pipeline:
1. OpenCV   : OpenCV CharucoDetector -> cv2.calibrateCamera
2. Ours     : Marker Homography + Sub-pixel Refine -> Custom Joint LM Optimizer
3. Lensboy  : Lensboy extract_frames_from_charuco -> Lensboy calibrate_camera (OpenCVConfig)

Reports:
- Focal lengths (fx, fy)
- Principal point (cx, cy)
- Radial distortion coefficients (k1, k2)
- Mean and RMS Reprojection Error (px)
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.board import BoardModel
from calib.predict import predict_corners
from calib.refine import refine_corner
from synthetic_calib.joint_refine import calibrate_full, project

try:
    import lensboy as lb
    HAVE_LENSBOY = True
except ImportError:
    HAVE_LENSBOY = False


READ_FLAGS = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION


def adaptive_drift_threshold(n_markers, base=2.5, scale=15.0, max_thresh=8.0):
    if n_markers <= 0:
        return max_thresh
    return float(min(base + scale / np.sqrt(n_markers), max_thresh))


# -----------------------------------------------------------------------------
# 1. OUR PIPELINE (Detection + Calibration)
# -----------------------------------------------------------------------------
def detect_corners_ours(img_path, model, aruco_detector, w=5):
    """Our custom marker homography prediction + gradient orthogonality sub-pixel refinement."""
    gray = cv2.imread(str(img_path), READ_FLAGS)
    if gray is None:
        return None

    mc, mids, _ = aruco_detector.detectMarkers(gray)
    n_markers = 0 if mids is None else len(mids)
    drift_thresh = adaptive_drift_threshold(n_markers)

    if mids is None or len(mids) == 0:
        return None

    pred = predict_corners(model, mc, mids)
    g = gray.astype(np.float32)
    Gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    Gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)

    ours = {}
    for cid, pt in pred.items():
        q, _ = refine_corner(Gx, Gy, pt, w=w)
        if np.linalg.norm(q - pt) < drift_thresh:
            ours[cid] = q

    if len(ours) < 4:
        return None

    cids = sorted(ours)
    obj_pts = model.corners[cids][:, :2].astype(np.float64)
    img_pts = np.array([ours[c] for c in cids], dtype=np.float64)
    h, ww = gray.shape
    return {
        "name": Path(img_path).name,
        "ids": np.array(cids, dtype=np.int64),
        "obj_pts": obj_pts,
        "img_pts": img_pts,
        "size": (ww, h),
    }


def run_our_calibration(img_paths, model, aruco_detector):
    """Full calibration using our corner detector + our joint optimizer."""
    views = []
    image_size = None
    for p in img_paths:
        v = detect_corners_ours(p, model, aruco_detector)
        if v is not None:
            views.append(v)
            if image_size is None:
                image_size = v["size"]

    res = calibrate_full(views, image_size, estimate_distortion=True, staged=True, robust=True, robust_loss="huber")
    return {
        "Method": "Ours (Proposed)",
        "Corner Detection": "Marker Homography + Sub-pixel Refine",
        "fx": res.K[0, 0],
        "fy": res.K[1, 1],
        "cx": res.K[0, 2],
        "cy": res.K[1, 2],
        "k1": res.dist[0],
        "k2": res.dist[1],
        "Mean Error (px)": res.mean_error,
        "RMS Error (px)": res.rms_error,
        "Views Used": len(views),
    }


# -----------------------------------------------------------------------------
# 2. OPENCV PIPELINE (Detection + Calibration)
# -----------------------------------------------------------------------------
def run_opencv_pipeline(img_paths, model):
    """OpenCV's native CharucoDetector + cv2.calibrateCamera."""
    cv_board = cv2.aruco.CharucoBoard((10, 7), model.sq, 0.026, model.dictionary)
    charuco_det = cv2.aruco.CharucoDetector(cv_board)

    all_obj, all_img = [], []
    image_size = None
    n_views = 0

    for p in img_paths:
        gray = cv2.imread(str(p), READ_FLAGS)
        if gray is None:
            continue
        cc, cids, _, _ = charuco_det.detectBoard(gray)
        if cids is None or len(cids) < 4:
            continue
        obj3d = cv_board.getChessboardCorners()[cids.flatten()]
        img2d = cc.reshape(-1, 1, 2).astype(np.float32)
        all_obj.append(obj3d.astype(np.float32))
        all_img.append(img2d)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        n_views += 1

    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, rv, tv = cv2.calibrateCamera(all_obj, all_img, tuple(image_size), None, None, flags=flags)

    all_errs = []
    for o, i, r, t in zip(all_obj, all_img, rv, tv):
        pr, _ = cv2.projectPoints(o, r, t, K, dist)
        diff = np.linalg.norm(pr.reshape(-1, 2) - i.reshape(-1, 2), axis=1)
        all_errs.append(diff)
    all_errs = np.concatenate(all_errs)

    d = dist.ravel()
    return {
        "Method": "OpenCV Baseline",
        "Corner Detection": "OpenCV CharucoDetector",
        "fx": K[0, 0],
        "fy": K[1, 1],
        "cx": K[0, 2],
        "cy": K[1, 2],
        "k1": d[0],
        "k2": d[1],
        "Mean Error (px)": float(all_errs.mean()),
        "RMS Error (px)": float(rms),
        "Views Used": n_views,
    }


# -----------------------------------------------------------------------------
# 3. LENSBOY PIPELINE (Detection + Calibration with OpenCVConfig)
# -----------------------------------------------------------------------------
def run_lensboy_pipeline(img_paths, model):
    """Lensboy's native extract_frames_from_charuco + calibrate_camera(OpenCVConfig)."""
    if not HAVE_LENSBOY:
        raise RuntimeError("Lensboy is not installed in this environment.")

    cv_board = cv2.aruco.CharucoBoard((10, 7), model.sq, 0.026, model.dictionary)
    images = [cv2.imread(str(p), READ_FLAGS) for p in img_paths]
    h, w = images[0].shape[:2]

    # Lensboy's native corner extraction
    target_points, frames, valid_indices = lb.extract_frames_from_charuco(cv_board, images)

    # Lensboy calibration using OpenCVConfig
    config = lb.OpenCVConfig(image_height=h, image_width=w)
    res = lb.calibrate_camera(
        target_points=target_points,
        frames=frames,
        camera_model_config=config,
    )

    # Compute residuals
    all_residuals = []
    for fi in res.frame_diagnostics:
        if fi is not None:
            all_residuals.append(np.linalg.norm(fi.residuals, axis=1))
    all_res_flat = np.concatenate(all_residuals) if all_residuals else np.array([])
    mean_e = float(np.mean(all_res_flat))
    rms_e = float(np.sqrt(np.mean(all_res_flat ** 2)))

    coeffs = np.array(res.camera_model.distortion_coeffs)
    k1 = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    k2 = float(coeffs[1]) if len(coeffs) > 1 else 0.0

    return {
        "Method": "Lensboy (OpenCVConfig)",
        "Corner Detection": "Lensboy extract_frames_from_charuco",
        "fx": float(res.camera_model.fx),
        "fy": float(res.camera_model.fy),
        "cx": float(res.camera_model.cx),
        "cy": float(res.camera_model.cy),
        "k1": k1,
        "k2": k2,
        "Mean Error (px)": mean_e,
        "RMS Error (px)": rms_e,
        "Views Used": len(valid_indices),
    }


def main():
    set_a_dir = ROOT / "data" / "set_a"
    out_dir = ROOT / "outputs" / "set_a_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        set_a_dir.glob("*.jpeg"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else p.name
    )

    print("=" * 80)
    print("  3-WAY CAMERA CALIBRATION COMPARISON: OpenCV vs. Ours vs. Lensboy")
    print(f"  Dataset : Set A ({len(img_paths)} images, 4096x3072)")
    print("=" * 80)

    model = BoardModel()
    aruco_detector = cv2.aruco.ArucoDetector(model.dictionary, cv2.aruco.DetectorParameters())

    results = []

    # 1. OpenCV Baseline
    print("\n[1/3] Running OpenCV Pipeline (CharucoDetector -> cv2.calibrateCamera)...")
    res_cv = run_opencv_pipeline(img_paths, model)
    results.append(res_cv)

    # 2. Our Pipeline
    print("\n[2/3] Running Our Pipeline (Marker Homography -> Sub-Pixel Refine -> Joint LM)...")
    res_ours = run_our_calibration(img_paths, model, aruco_detector)
    results.append(res_ours)

    # 3. Lensboy Pipeline
    if HAVE_LENSBOY:
        print("\n[3/3] Running Lensboy Pipeline (extract_frames_from_charuco -> OpenCVConfig)...")
        res_lb = run_lensboy_pipeline(img_paths, model)
        results.append(res_lb)
    else:
        print("\n[3/3] Lensboy not available in current environment (run via WSL).")

    df = pd.DataFrame(results)

    # Save CSV
    out_csv = out_dir / "three_way_calibration_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved results to: {out_csv}")

    # Display comparison table
    cols = ["Method", "Corner Detection", "fx", "fy", "cx", "cy", "k1", "k2", "Mean Error (px)", "RMS Error (px)"]
    print("\n" + "=" * 115)
    print("  FINAL 3-WAY CALIBRATION COMPARISON")
    print("=" * 115)
    print(df[cols].to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    main()
