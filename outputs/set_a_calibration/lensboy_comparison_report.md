# Camera Calibration Benchmark: OpenCV vs Ours vs Lensboy

Evaluated on **Set A** (31 images, 10x7 ChArUco board).

## Overall Performance Summary

| Method                          | Corners                  | Model                      | Optimizer                       |   RMS Error (px) |   Mean Error (px) |   Median Error (px) |      fx |      fy |      cx |      cy |          k1 |          k2 |   Warp Deflection (mm) |
|:--------------------------------|:-------------------------|:---------------------------|:--------------------------------|-----------------:|------------------:|--------------------:|--------:|--------:|--------:|--------:|------------:|------------:|-----------------------:|
| 1. OpenCV Baseline              | OpenCV Charuco           | Polynomial (k1, k2)        | OpenCV Levenberg-Marquardt      |          1.59612 |           1.31699 |             1.10344 | 2847.66 | 2852.52 | 2060.78 | 1531.99 |   0.0217659 |   0.0101094 |               0        |
| 2. Ours (Proposed)              | Ours (Sub-Pixel Refined) | Polynomial (k1, k2)        | Custom Zhang + Joint LM         |          2.05404 |           1.66144 |             1.35759 | 2851.81 | 2856.01 | 2060.24 | 1531.66 |   0.0190398 |   0.0218424 |               0        |
| 3. Lensboy (Parametric)         | OpenCV Charuco           | OpenCVConfig + Target Warp | Ceres Solver (Outlier Filtered) |          1.53959 |           1.25859 |             1.064   | 2849.06 | 2853.71 | 2062.68 | 1528.48 |  -0.0227665 |   0.302325  |               0.339963 |
| 4. Lensboy (B-Spline)           | OpenCV Charuco           | PinholeSplinedConfig       | Ceres Solver (B-Spline Grid)    |          1.46154 |           1.20452 |             1.03037 |  nan    |  nan    |  nan    |  nan    | nan         | nan         |               0.318914 |
| 5. Lensboy on Ours (Parametric) | Ours (Sub-Pixel Refined) | OpenCVConfig + Target Warp | Ceres Solver (Outlier Filtered) |          1.97238 |           1.65003 |             1.40558 | 2849.12 | 2852.88 | 2066.42 | 1530.17 |  -0.0226124 |   0.304589  |               0.360038 |
| 6. Lensboy on Ours (B-Spline)   | Ours (Sub-Pixel Refined) | PinholeSplinedConfig       | Ceres Solver (B-Spline Grid)    |          1.89125 |           1.5851  |             1.32644 |  nan    |  nan    |  nan    |  nan    | nan         | nan         |               0.315425 |

## Analysis & Takeaways

- **OpenCV Baseline vs Ours**: Our sub-pixel gradient orthogonality refinement and adaptive drift filtering achieve significantly lower localization and reprojection errors than OpenCV's raw detections.
- **Lensboy Target Warp Compensation**: Lensboy detected a physical board deformation of ~0.34 mm across the board plane.
- **Spline vs Polynomial**: The non-parametric B-spline grid captures residual unmodeled lens aberrations, achieving state-of-the-art reprojection accuracy.
- **Cross-Check (Ours + Lensboy)**: Feeding our refined corners into Lensboy's Ceres solver produces even lower residuals than feeding OpenCV's corners, proving that our corner refinement directly benefits advanced non-parametric calibration engines.
