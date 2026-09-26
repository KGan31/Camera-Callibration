# Camera Calibration with ChArUco Boards

This project implements a full camera calibration pipeline from scratch and compares it with OpenCV using the same image sets.

## Goals

1. Start with a **standard ChArUco board** and implement:
   - ChArUco corner detection
   - One homography estimation per view
   - Intrinsic recovery with **Zhang’s planar calibration method**
   - Joint refinement of intrinsics, per-image poses, and radial distortion by minimizing reprojection error
2. Compare all estimated parameters against OpenCV calibration on the same images.
3. Extend the same approach to the **custom ChArUco board** used in the lab, which contains missing regions and large cut-outs.
4. Study parameter stability as the number of images increases from **3 to 30**.
5. Repeat calibration with mostly fronto-parallel views and explain why focal length becomes weakly constrained despite low reprojection error.

## Pipeline Requirements

### 1) Standard ChArUco Board Calibration

- Detect ArUco markers and interpolate ChArUco corners.
- Build 2D–3D correspondences using known board geometry.
- Estimate one homography \(H_i\) for each view.
- Recover initial intrinsics from all homographies using Zhang’s method.
- Solve for extrinsics \((R_i, t_i)\) per image.
- Refine:
  - camera intrinsics \((f_x, f_y, c_x, c_y)\),
  - distortion parameters (at least radial terms),
  - all per-image poses,
  by minimizing total reprojection error with nonlinear optimization.

### 2) Custom ChArUco Board Calibration

Because the custom board has cut-outs and missing cells:

- Use **marker IDs** and the **known board layout** to recover valid correspondences.
- Do not assume a fully visible regular grid.
- Ensure partial board visibility still contributes correctly to calibration.

## Evaluation and Reporting

For both standard and custom boards, report:

- Focal lengths: \(f_x, f_y\)
- Principal point: \(c_x, c_y\)
- Distortion coefficients (radial terms at minimum)
- Mean reprojection error
- OpenCV baseline results for direct comparison

### Image Count Study (3 to 30)

- Calibrate with increasing image counts (e.g., 3, 5, 10, 15, 20, 25, 30).
- Plot or tabulate how intrinsics/distortion/reprojection error change with more views.
- Discuss convergence and stability trends.

### Fronto-Parallel View Study

- Repeat calibration using mostly fronto-parallel images.
- Compare parameters and error with the diverse-view setup.
- Explain why fronto-parallel geometry provides poor depth/perspective diversity, making focal length weakly observable even when reprojection error remains small.

## Expected Deliverables

- Calibration implementation from scratch (standard + custom board handling)
- OpenCV comparison using identical image sets
- Quantitative report/tables/plots for 3-to-30 image experiments
- Short analysis of fronto-parallel degeneracy effects on focal length estimation
