"""Run the corner detection benchmark on set_new_a."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import calib.tests.test_corner_check_set_a as bench

def run():
    set_dir  = ROOT / "data" / "set_a_new"
    out_dir  = ROOT / "outputs" / "set_a_new_corner_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)

    img_paths = sorted(
        list(set_dir.glob("*.jpg")) + list(set_dir.glob("*.jpeg")),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else p.name
    )
    print("=" * 68)
    print("  ChArUco Corner Detection Benchmark: Ours vs OpenCV (set_new_a)")
    print(f"  Dataset : {set_dir}  ({len(img_paths)} images)")
    print("=" * 68)

    from calib.board import BoardModel
    model           = BoardModel()
    cv_board        = cv2.aruco.CharucoBoard((10, 7), model.sq, 0.026, model.dictionary)
    params          = cv2.aruco.DetectorParameters()
    aruco_detector  = cv2.aruco.ArucoDetector(model.dictionary, params)
    charuco_detector= cv2.aruco.CharucoDetector(cv_board)

    results, all_details, total_dists = [], {}, []

    for idx, p in enumerate(img_paths, 1):
        metrics, details = bench.evaluate_single_image(
            p, model, cv_board, aruco_detector, charuco_detector, adaptive=True, w=5
        )
        results.append(metrics)
        all_details[p.name] = details
        total_dists.extend(details["dists"])

        mean_s = f"{metrics['mean_dist']:.3f} px" if not np.isnan(metrics["mean_dist"]) else "   N/A"
        rms_s  = f"{metrics['rms_dist']:.3f} px"  if not np.isnan(metrics["rms_dist"])  else "   N/A"
        thr_s  = f"{metrics['drift_thresh_used']:.2f}px"
        name_s = p.name[:32]
        print(f"[{idx:2d}/{len(img_paths)}] {name_s:<32} mk:{metrics['n_markers']:2d} thr:{thr_s:<7}"
              f" ours:{metrics['n_ours']:2d} cv:{metrics['n_cv']:2d}"
              f" common:{metrics['n_common']:2d} rej:{metrics['drift_rejected']:2d}"
              f" mean:{mean_s:>9} rms:{rms_s:>9}")

    df = pd.DataFrame(results)
    csv_path = out_dir / "set_new_a_corner_metrics.csv"
    df.to_csv(csv_path, index=False)

    tot_ours     = int(df["n_ours"].sum())
    tot_cv       = int(df["n_cv"].sum())
    tot_common   = int(df["n_common"].sum())
    tot_possible = len(df) * 54
    arr = np.array(total_dists)

    print()
    print("=" * 68)
    print("  DATASET-WIDE SUMMARY (set_new_a)")
    print("=" * 68)
    print(f"  Images evaluated         : {len(df)}")
    print(f"  Total corners possible   : {tot_possible}  ({len(df)} x 54)")
    print(f"  Detected (Ours)          : {tot_ours}  ({tot_ours/tot_possible*100:.2f}%)")
    print(f"  Detected (OpenCV)        : {tot_cv}  ({tot_cv/tot_possible*100:.2f}%)")
    print(f"  Common corners           : {tot_common}  ({tot_common/max(tot_cv,1)*100:.2f}% of OpenCV)")
    print(f"  Total drift rejected     : {int(df['drift_rejected'].sum())}  (mean {df['drift_rejected'].mean():.1f}/img)")
    if len(arr) > 0:
        print(f"  Mean localization error  : {np.mean(arr):.4f} px")
        print(f"  Median localization error: {np.median(arr):.4f} px")
        print(f"  RMS  localization error  : {np.sqrt(np.mean(arr**2)):.4f} px")
        print(f"  95th percentile error    : {np.percentile(arr, 95):.4f} px")
        print(f"  Max  localization error  : {np.max(arr):.4f} px")
        mean_pred = df["pred_mean_dist"].mean()
        mean_ref  = df["mean_dist"].mean()
        gain_pct  = df["refinement_gain_pct"].mean()
        if not np.isnan(mean_pred):
            print(f"  Prediction error (raw)   : {mean_pred:.4f} px")
            print(f"  Refined error            : {mean_ref:.4f} px  (refinement gain {gain_pct:.1f}%)")
    print("=" * 68)
    print(f"\n  [Saved] CSV -> {csv_path}")

if __name__ == "__main__":
    run()
