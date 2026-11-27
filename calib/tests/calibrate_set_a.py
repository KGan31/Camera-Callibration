"""End-to-end Camera Calibration on Set A using our custom pipeline.

Pipeline
--------
1. Detect ArUco markers (OpenCV ArucoDetector)
2. Predict inner ChArUco corner positions via per-marker homography  (calib/predict.py)
3. Sub-pixel refinement via gradient orthogonality              (calib/refine.py)
4. Adaptive drift-threshold rejection                           (this script)
5. Zhang's closed-form intrinsics                               (synthetic_calib/zhang.py)
6. Joint Levenberg-Marquardt refinement with k1, k2 distortion  (synthetic_calib/joint_refine.py)

Also runs OpenCV calibrateCamera on the same images for side-by-side comparison.

Outputs
-------
- Printed per-image and aggregate results to stdout
- outputs/set_a_calibration/calibration_report.md
- outputs/set_a_calibration/reprojection_charts.png
- outputs/set_a_calibration/calibration_results.npz  (our K, dist, poses)
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.board import BoardModel
from calib.predict import predict_corners
from calib.refine import refine_corner
from synthetic_calib.joint_refine import calibrate_full
from synthetic_calib.zhang import calibrate_closed_form


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive drift threshold (same formula as benchmark)
# ─────────────────────────────────────────────────────────────────────────────
def adaptive_drift_threshold(n_markers, base=2.5, scale=15.0, max_thresh=8.0):
    if n_markers <= 0:
        return max_thresh
    return float(min(base + scale / np.sqrt(n_markers), max_thresh))


# ─────────────────────────────────────────────────────────────────────────────
# Corner detection: our pipeline
# ─────────────────────────────────────────────────────────────────────────────
def detect_corners_ours(img_path, model, aruco_detector, w=5):
    """Returns (obj_pts, img_pts, n_markers, n_corners, drift_thresh, image_size) or None."""
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)

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
    obj_pts = model.corners[cids][:, :2].astype(np.float64)   # (N, 2) metres
    img_pts = np.array([ours[c] for c in cids], dtype=np.float64)  # (N, 2) pixels
    h, ww = gray.shape
    return obj_pts, img_pts, n_markers, len(ours), drift_thresh, (ww, h)


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV reference calibration
# ─────────────────────────────────────────────────────────────────────────────
def run_opencv_calibration(img_paths, model):
    """Run OpenCV's CharucoDetector + calibrateCamera for comparison."""
    cv_board    = cv2.aruco.CharucoBoard((10, 7), model.sq, 0.026, model.dictionary)
    charuco_det = cv2.aruco.CharucoDetector(cv_board)

    all_obj, all_img = [], []
    image_size = None
    n_views = 0

    for p in img_paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        cc, cids, _, _ = charuco_det.detectBoard(gray)
        if cids is None or len(cids) < 4:
            continue
        obj3d = cv_board.getChessboardCorners()[cids.flatten()]   # (N, 3)
        img2d = cc.reshape(-1, 1, 2).astype(np.float32)
        all_obj.append(obj3d.astype(np.float32))
        all_img.append(img2d)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        n_views += 1

    if n_views < 3:
        return None

    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, image_size, None, None, flags=flags
    )
    return {
        "K": K, "dist": dist.ravel(),
        "rms": rms, "n_views": n_views,
        "rvecs": rvecs, "tvecs": tvecs,
        "image_size": image_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_a_dir = ROOT / "data" / "set_a"
    out_dir   = ROOT / "outputs" / "set_a_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        set_a_dir.glob("*.jpeg"),
        key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else p.name
    )

    print("=" * 70)
    print("  End-to-End Camera Calibration: Our Pipeline vs. OpenCV  (Set A)")
    print(f"  Dataset : {set_a_dir}  ({len(img_paths)} images)")
    print("=" * 70 + "\n")

    model          = BoardModel()
    params         = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(model.dictionary, params)

    # ── 1. Collect views ──────────────────────────────────────────────────────
    views      = []    # list of dicts for synthetic_calib
    skipped    = []
    image_size = None

    print(f"  {'Image':<12} {'Markers':>7} {'Thresh':>8} {'Corners':>9}")
    print("  " + "-" * 42)

    for p in img_paths:
        result = detect_corners_ours(p, model, aruco_detector)
        if result is None:
            skipped.append(p.name)
            print(f"  {p.name:<12}  SKIPPED (too few corners)")
            continue

        obj_pts, img_pts, n_mk, n_cor, thr, sz = result
        views.append({"obj_pts": obj_pts, "img_pts": img_pts, "name": p.name})
        if image_size is None:
            image_size = sz
        print(f"  {p.name:<12}  {n_mk:>5}    {thr:>5.2f}px   {n_cor:>5}")

    print(f"\n  Views accepted : {len(views)} / {len(img_paths)}")
    if skipped:
        print(f"  Skipped        : {skipped}")

    if len(views) < 3:
        print("\n[ERROR] Need at least 3 views for calibration. Aborting.")
        return

    # ── 2. Zhang closed-form ─────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  STEP 1: Zhang Closed-Form Calibration")
    print("-" * 70)
    K_zhang, poses_zhang, _ = calibrate_closed_form(views, image_size)
    print(f"\n  K (Zhang):\n{K_zhang}\n")
    print(f"  fx={K_zhang[0,0]:.2f}  fy={K_zhang[1,1]:.2f}  cx={K_zhang[0,2]:.2f}  cy={K_zhang[1,2]:.2f}")

    # ── 3. Full joint refinement ──────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  STEP 2: Joint LM Refinement  (fx, fy, cx, cy, k1, k2 + all poses)")
    print("-" * 70)
    res = calibrate_full(views, image_size, estimate_distortion=True,
                         staged=True, robust=True, robust_loss="huber", f_scale=1.0)

    print(f"\n  K (LM refined):\n{res.K}\n")
    print(f"  fx={res.K[0,0]:.2f}  fy={res.K[1,1]:.2f}  cx={res.K[0,2]:.2f}  cy={res.K[1,2]:.2f}")
    print(f"  k1={res.dist[0]:.6f}   k2={res.dist[1]:.6f}")
    print(f"\n  Reprojection Error:")
    print(f"    Mean      : {res.mean_error:.4f} px")
    print(f"    RMS       : {res.rms_error:.4f} px")
    print(f"    Converged : {res.converged}   evals: {res.n_evals}")

    # ── 4. OpenCV reference ───────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  STEP 3: OpenCV calibrateCamera (reference)")
    print("-" * 70)
    cv_res = run_opencv_calibration(img_paths, model)
    if cv_res:
        print(f"\n  K (OpenCV):\n{cv_res['K']}\n")
        print(f"  fx={cv_res['K'][0,0]:.2f}  fy={cv_res['K'][1,1]:.2f}  cx={cv_res['K'][0,2]:.2f}  cy={cv_res['K'][1,2]:.2f}")
        print(f"  k1={cv_res['dist'][0]:.6f}   k2={cv_res['dist'][1]:.6f}")
        print(f"\n  RMS Reprojection (OpenCV) : {cv_res['rms']:.4f} px")
        print(f"  Views used by OpenCV      : {cv_res['n_views']}")
    else:
        print("  OpenCV calibration failed.")

    # ── 5. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS COMPARISON")
    print("=" * 70)
    print(f"  {'Parameter':<20} {'Zhang (Ours)':>14} {'LM (Ours)':>14} {'OpenCV':>14}")
    print("  " + "-" * 64)

    def row(name, z, lm, cv):
        cv_s = f"{cv:.4f}" if cv is not None else "  N/A"
        print(f"  {name:<20} {z:>14.4f} {lm:>14.4f} {cv_s:>14}")

    cv_fx = cv_res['K'][0,0] if cv_res else None
    cv_fy = cv_res['K'][1,1] if cv_res else None
    cv_cx = cv_res['K'][0,2] if cv_res else None
    cv_cy = cv_res['K'][1,2] if cv_res else None
    row("fx (px)", K_zhang[0,0], res.K[0,0], cv_fx)
    row("fy (px)", K_zhang[1,1], res.K[1,1], cv_fy)
    row("cx (px)", K_zhang[0,2], res.K[0,2], cv_cx)
    row("cy (px)", K_zhang[1,2], res.K[1,2], cv_cy)
    print(f"  {'k1':<20} {'—':>14} {res.dist[0]:>14.6f} {cv_res['dist'][0]:>14.6f}" if cv_res else
          f"  {'k1':<20} {'—':>14} {res.dist[0]:>14.6f} {'N/A':>14}")
    print(f"  {'k2':<20} {'—':>14} {res.dist[1]:>14.6f} {cv_res['dist'][1]:>14.6f}" if cv_res else
          f"  {'k2':<20} {'—':>14} {res.dist[1]:>14.6f} {'N/A':>14}")
    print(f"  {'RMS reproj (px)':<20} {'—':>14} {res.rms_error:>14.4f} {cv_res['rms']:>14.4f}" if cv_res else
          f"  {'RMS reproj (px)':<20} {'—':>14} {res.rms_error:>14.4f} {'N/A':>14}")
    print("=" * 70)

    # ── 6. Per-view breakdown ─────────────────────────────────────────────────
    print(f"\n  Per-View Reprojection Errors (LM-refined):")
    print(f"  {'Image':<14} {'Corners':>8} {'Mean err (px)':>14}")
    print("  " + "-" * 38)
    for v, err in zip(views, res.view_errors):
        nc = len(v["obj_pts"])
        print(f"  {v['name']:<14} {nc:>8} {err:>14.4f}")

    # ── 7. Save .npz ─────────────────────────────────────────────────────────
    npz_path = out_dir / "calibration_results.npz"
    save_dict = dict(
        K_zhang=K_zhang,
        K=res.K,
        dist=res.dist,
        mean_error=res.mean_error,
        rms_error=res.rms_error,
        view_errors=res.view_errors,
        K_opencv=cv_res['K'] if cv_res else np.full((3,3), np.nan),
        dist_opencv=cv_res['dist'] if cv_res else np.full(5, np.nan),
        rms_opencv=np.float64(cv_res['rms']) if cv_res else np.float64(np.nan),
    )
    np.savez(str(npz_path), **save_dict)
    print(f"\n[Saved] calibration_results.npz -> {npz_path}")

    # ── 8. Charts ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Set A End-to-End Calibration — Our Pipeline vs OpenCV", fontsize=14, fontweight="bold")

    # Per-view reprojection error
    ax1 = axes[0]
    names  = [v["name"].replace(".jpeg", "") for v in views]
    errors = res.view_errors
    colors = ["#e74c3c" if e > 1.0 else "#f39c12" if e > 0.5 else "#2ecc71" for e in errors]
    ax1.bar(names, errors, color=colors, edgecolor="white", linewidth=0.4)
    ax1.axhline(res.mean_error, color="navy",   linestyle="--", lw=1.5, label=f"Our mean: {res.mean_error:.3f} px")
    ax1.axhline(res.rms_error,  color="crimson",linestyle=":",  lw=1.5, label=f"Our RMS:  {res.rms_error:.3f} px")
    if cv_res:
        ax1.axhline(cv_res['rms'], color="#27ae60", linestyle="-.", lw=1.5, label=f"OpenCV RMS: {cv_res['rms']:.3f} px")
    ax1.set_title("Per-View Reprojection Error (Our LM Pipeline)", fontweight="bold")
    ax1.set_xlabel("Image")
    ax1.set_ylabel("Mean Reprojection Error (px)")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Intrinsics comparison
    ax2 = axes[1]
    param_names = ["fx", "fy", "cx", "cy"]
    z_vals  = [K_zhang[0,0], K_zhang[1,1], K_zhang[0,2], K_zhang[1,2]]
    lm_vals = [res.K[0,0],   res.K[1,1],   res.K[0,2],   res.K[1,2]]
    cv_vals = ([cv_res['K'][0,0], cv_res['K'][1,1], cv_res['K'][0,2], cv_res['K'][1,2]]
                if cv_res else [0]*4)
    x = np.arange(4)
    bw = 0.26
    ax2.bar(x - bw, z_vals,  bw, label="Zhang (Ours)",  color="#3498db", edgecolor="white", lw=0.4)
    ax2.bar(x,      lm_vals, bw, label="LM (Ours)",     color="#e67e22", edgecolor="white", lw=0.4)
    if cv_res:
        ax2.bar(x + bw, cv_vals, bw, label="OpenCV",    color="#2ecc71", edgecolor="white", lw=0.4)
    ax2.set_title("Intrinsic Parameters", fontweight="bold")
    ax2.set_xlabel("Parameter")
    ax2.set_ylabel("Value (pixels)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_names, fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = out_dir / "reprojection_charts.png"
    plt.savefig(str(chart_path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Saved] reprojection_charts.png -> {chart_path}")

    # ── 9. Markdown report ────────────────────────────────────────────────────
    report_path = out_dir / "calibration_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Camera Calibration Report — Set A\n\n")
        f.write("## Pipeline\n\n")
        f.write("1. ArUco detection\n")
        f.write("2. Per-marker homography corner prediction (`calib/predict.py`)\n")
        f.write("3. Sub-pixel gradient orthogonality refinement (`calib/refine.py`)\n")
        f.write("4. Adaptive drift-threshold rejection\n")
        f.write("5. Zhang closed-form (`synthetic_calib/zhang.py`)\n")
        f.write("6. Joint LM refinement with k1, k2 (`synthetic_calib/joint_refine.py`)\n\n")
        f.write("## Results\n\n")
        f.write("| Parameter | Zhang (Ours) | LM Refined (Ours) | OpenCV |\n")
        f.write("| :--- | ---: | ---: | ---: |\n")
        cv_v = lambda k: f"{k:.4f}" if cv_res else "N/A"
        f.write(f"| fx (px) | {K_zhang[0,0]:.4f} | {res.K[0,0]:.4f} | {cv_v(cv_res['K'][0,0]) if cv_res else 'N/A'} |\n")
        f.write(f"| fy (px) | {K_zhang[1,1]:.4f} | {res.K[1,1]:.4f} | {cv_v(cv_res['K'][1,1]) if cv_res else 'N/A'} |\n")
        f.write(f"| cx (px) | {K_zhang[0,2]:.4f} | {res.K[0,2]:.4f} | {cv_v(cv_res['K'][0,2]) if cv_res else 'N/A'} |\n")
        f.write(f"| cy (px) | {K_zhang[1,2]:.4f} | {res.K[1,2]:.4f} | {cv_v(cv_res['K'][1,2]) if cv_res else 'N/A'} |\n")
        f.write(f"| k1      | —     | {res.dist[0]:.6f} | {cv_v(cv_res['dist'][0]) if cv_res else 'N/A'} |\n")
        f.write(f"| k2      | —     | {res.dist[1]:.6f} | {cv_v(cv_res['dist'][1]) if cv_res else 'N/A'} |\n")
        f.write(f"| RMS reproj (px) | — | {res.rms_error:.4f} | {cv_v(cv_res['rms']) if cv_res else 'N/A'} |\n\n")
        f.write("## Per-View Reprojection (Our Pipeline)\n\n")
        f.write("| Image | Corners | Mean Error (px) |\n| :--- | :---: | :---: |\n")
        for v, err in zip(views, res.view_errors):
            f.write(f"| `{v['name']}` | {len(v['obj_pts'])} | {err:.4f} |\n")

    print(f"[Saved] calibration_report.md -> {report_path}")
    print("\nDone!")
    return res, cv_res


if __name__ == "__main__":
    main()
