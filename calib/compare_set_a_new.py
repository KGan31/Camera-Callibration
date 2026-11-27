"""Calibrate on Set A New (1920x1440) and compare against Custom Board (Set B, Set D) and Set A.

Investigates whether camera intrinsics (fx, fy, cx, cy, k1, k2) match between the standard board
captured with the new app setting (set_a_new at 1920x1440) and the custom board (set_b / set_d at 1920x1440).
"""

import glob
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.board import BoardModel
from calib.custom_board import CustomBoardModel
from calib.real_calibrate import detect_views, opencv_charuco_views, opencv_calibrate
from synthetic_calib.joint_refine import calibrate_full

def get_paths(dataset_name):
    patterns = [f"data/{dataset_name}/*.jpg", f"data/{dataset_name}/*.jpeg"]
    files = []
    for p in patterns:
        files.extend(glob.glob(str(ROOT / p)))
    # Sort naturally
    return sorted(files)
    def sort_key(s):
        name = os.path.basename(s).split(".")[0]
        parts = name.split("_")
        # Handles A_1, B_1, D_1 as well as A_new_1
        try:
            return int(parts[-1])
        except ValueError:
            return s
    return sorted(files, key=sort_key)

def main():
    print("=" * 80)
    print("  CALIBRATION RUN: Set A New vs. Custom Board (Set B & Set D)")
    print("=" * 80)

    # 1. Standard Board on Set A New (1920x1440)
    std_model = BoardModel()
    paths_a_new = get_paths("set_a_new")
    print(f"\n[1/3] Processing set_a_new ({len(paths_a_new)} images)...")
    views_a_new_ours, sk_a_new, size_a_new = detect_views(std_model, paths_a_new)
    views_a_new_cv = opencv_charuco_views(paths_a_new)
    print(f"  Accepted views: Ours = {len(views_a_new_ours)}, OpenCV = {len(views_a_new_cv)}, Size = {size_a_new}")

    res_a_new_ours = calibrate_full(views_a_new_ours, size_a_new)
    res_a_new_cv_own = opencv_calibrate(views_a_new_cv, size_a_new)
    res_a_new_cv_same = opencv_calibrate(views_a_new_ours, size_a_new)

    # 2. Custom Board on Set B (1920x1440)
    custom_model = CustomBoardModel(diagonal_junctions=False)
    paths_b = get_paths("set_b")
    print(f"\n[2/3] Processing set_b ({len(paths_b)} images)...")
    views_b, _, size_b = detect_views(custom_model, paths_b)
    print(f"  Accepted views: Ours = {len(views_b)}, Size = {size_b}")
    res_b_ours = calibrate_full(views_b, size_b)
    res_b_cv = opencv_calibrate(views_b, size_b)

    # 3. Custom Board on Set D (1920x1440) & Combined B+D
    paths_d = get_paths("set_d")
    print(f"\n[3/3] Processing set_d ({len(paths_d)} images)...")
    views_d, _, size_d = detect_views(custom_model, paths_d)
    print(f"  Accepted views: Ours = {len(views_d)}, Size = {size_d}")
    res_d_ours = calibrate_full(views_d, size_d)
    res_bd_ours = calibrate_full(views_b + views_d, size_b)
    res_bd_cv = opencv_calibrate(views_b + views_d, size_b)

    # 4. Also load Set A (4096x3072) for full reference
    paths_a = get_paths("set_a")
    views_a, _, size_a = detect_views(std_model, paths_a)
    res_a_ours = calibrate_full(views_a, size_a)

    # Compile Summary Table
    rows = [
        {
            "Dataset": "Set A (Old)",
            "Board Type": "Standard ChArUco (10x7)",
            "Method": "Ours",
            "Resolution": f"{size_a[0]}x{size_a[1]}",
            "fx (px)": res_a_ours.K[0, 0],
            "fy (px)": res_a_ours.K[1, 1],
            "cx (px)": res_a_ours.K[0, 2],
            "cy (px)": res_a_ours.K[1, 2],
            "k1": res_a_ours.dist[0],
            "k2": res_a_ours.dist[1],
            "Mean Err (px)": res_a_ours.mean_error,
            "RMS Err (px)": res_a_ours.rms_error,
        },
        {
            "Dataset": "Set A New",
            "Board Type": "Standard ChArUco (10x7)",
            "Method": "Ours",
            "Resolution": f"{size_a_new[0]}x{size_a_new[1]}",
            "fx (px)": res_a_new_ours.K[0, 0],
            "fy (px)": res_a_new_ours.K[1, 1],
            "cx (px)": res_a_new_ours.K[0, 2],
            "cy (px)": res_a_new_ours.K[1, 2],
            "k1": res_a_new_ours.dist[0],
            "k2": res_a_new_ours.dist[1],
            "Mean Err (px)": res_a_new_ours.mean_error,
            "RMS Err (px)": res_a_new_ours.rms_error,
        },
        {
            "Dataset": "Set A New",
            "Board Type": "Standard ChArUco (10x7)",
            "Method": "OpenCV (own corners)",
            "Resolution": f"{size_a_new[0]}x{size_a_new[1]}",
            "fx (px)": res_a_new_cv_own["K"][0, 0],
            "fy (px)": res_a_new_cv_own["K"][1, 1],
            "cx (px)": res_a_new_cv_own["K"][0, 2],
            "cy (px)": res_a_new_cv_own["K"][1, 2],
            "k1": res_a_new_cv_own["dist"][0],
            "k2": res_a_new_cv_own["dist"][1],
            "Mean Err (px)": res_a_new_cv_own["mean_error"],
            "RMS Err (px)": res_a_new_cv_own["rms_error"],
        },
        {
            "Dataset": "Set B",
            "Board Type": "Custom Lab Board",
            "Method": "Ours",
            "Resolution": f"{size_b[0]}x{size_b[1]}",
            "fx (px)": res_b_ours.K[0, 0],
            "fy (px)": res_b_ours.K[1, 1],
            "cx (px)": res_b_ours.K[0, 2],
            "cy (px)": res_b_ours.K[1, 2],
            "k1": res_b_ours.dist[0],
            "k2": res_b_ours.dist[1],
            "Mean Err (px)": res_b_ours.mean_error,
            "RMS Err (px)": res_b_ours.rms_error,
        },
        {
            "Dataset": "Set B",
            "Board Type": "Custom Lab Board",
            "Method": "OpenCV (same corners)",
            "Resolution": f"{size_b[0]}x{size_b[1]}",
            "fx (px)": res_b_cv["K"][0, 0],
            "fy (px)": res_b_cv["K"][1, 1],
            "cx (px)": res_b_cv["K"][0, 2],
            "cy (px)": res_b_cv["K"][1, 2],
            "k1": res_b_cv["dist"][0],
            "k2": res_b_cv["dist"][1],
            "Mean Err (px)": res_b_cv["mean_error"],
            "RMS Err (px)": res_b_cv["rms_error"],
        },
        {
            "Dataset": "Set B+D",
            "Board Type": "Custom Lab Board",
            "Method": "Ours",
            "Resolution": f"{size_b[0]}x{size_b[1]}",
            "fx (px)": res_bd_ours.K[0, 0],
            "fy (px)": res_bd_ours.K[1, 1],
            "cx (px)": res_bd_ours.K[0, 2],
            "cy (px)": res_bd_ours.K[1, 2],
            "k1": res_bd_ours.dist[0],
            "k2": res_bd_ours.dist[1],
            "Mean Err (px)": res_bd_ours.mean_error,
            "RMS Err (px)": res_bd_ours.rms_error,
        },
        {
            "Dataset": "Set B+D",
            "Board Type": "Custom Lab Board",
            "Method": "OpenCV (same corners)",
            "Resolution": f"{size_b[0]}x{size_b[1]}",
            "fx (px)": res_bd_cv["K"][0, 0],
            "fy (px)": res_bd_cv["K"][1, 1],
            "cx (px)": res_bd_cv["K"][0, 2],
            "cy (px)": res_bd_cv["K"][1, 2],
            "k1": res_bd_cv["dist"][0],
            "k2": res_bd_cv["dist"][1],
            "Mean Err (px)": res_bd_cv["mean_error"],
            "RMS Err (px)": res_bd_cv["rms_error"],
        },
    ]

    df = pd.DataFrame(rows)
    out_csv = ROOT / "outputs" / "set_a_new_comparison_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved results to: {out_csv}")

    print("\n" + "=" * 95)
    print("  CALIBRATION INTRINSICS COMPARISON")
    print("=" * 95)
    print(df[["Dataset", "Board Type", "Method", "Resolution", "fx (px)", "fy (px)", "cx (px)", "cy (px)", "Mean Err (px)"]].to_string(index=False))
    print("=" * 95)

    # Difference computation between Set A New (Ours) and Set B (Ours)
    fx_an = res_a_new_ours.K[0, 0]
    fy_an = res_a_new_ours.K[1, 1]
    cx_an = res_a_new_ours.K[0, 2]
    cy_an = res_a_new_ours.K[1, 2]

    fx_b = res_b_ours.K[0, 0]
    fy_b = res_b_ours.K[1, 1]
    cx_b = res_b_ours.K[0, 2]
    cy_b = res_b_ours.K[1, 2]

    print("\n--- Direct Comparison: Set A New (Standard) vs. Set B (Custom) [Same 1920x1440 Resolution] ---")
    print(f"fx: Set A New = {fx_an:.2f} px  |  Set B = {fx_b:.2f} px  |  Diff = {abs(fx_an - fx_b):.2f} px ({abs(fx_an - fx_b)/fx_an*100:.2f}%)")
    print(f"fy: Set A New = {fy_an:.2f} px  |  Set B = {fy_b:.2f} px  |  Diff = {abs(fy_an - fy_b):.2f} px ({abs(fy_an - fy_b)/fy_an*100:.2f}%)")
    print(f"cx: Set A New = {cx_an:.2f} px  |  Set B = {cx_b:.2f} px  |  Diff = {abs(cx_an - cx_b):.2f} px")
    print(f"cy: Set A New = {cy_an:.2f} px  |  Set B = {cy_b:.2f} px  |  Diff = {abs(cy_an - cy_b):.2f} px")
    print("-" * 80)

if __name__ == "__main__":
    main()
