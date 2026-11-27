# End-to-end calibration on Set A: ours vs OpenCV

31 images (4096x3072), 30 used by both pipelines (>= 12 corners each). Error numbers are computed identically for all rows. Held-out = 5-fold, pose-only fit on unseen views, evaluated on corners found by both detectors.

## Configurations

| configuration | n_views | n_corners | fx | fy | cx | cy | k1 | k2 | mean_err | rms_err | median_err | p95_err | max_err | ho_mean | ho_rms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ours_corners+ours_solver | 30 | 1457 | 2851 | 2856 | 2060 | 1532 | 0.02005 | 0.0176 | 1.538 | 1.896 | 1.252 | 3.801 | 6.64 | 1.593 | 1.889 |
| cv_corners+cv_solver | 30 | 1570 | 2848 | 2852 | 2061 | 1532 | 0.022 | 0.009386 | 1.316 | 1.596 | 1.104 | 3.027 | 7.011 | 1.35 | 1.639 |
| ours_corners+cv_solver | 30 | 1457 | 2847 | 2851 | 2061 | 1533 | 0.02369 | 0.00474 | 1.571 | 1.86 | 1.354 | 3.551 | 5.446 | 1.597 | 1.889 |
| cv_corners+ours_solver | 30 | 1570 | 2852 | 2857 | 2060 | 1531 | 0.01734 | 0.02521 | 1.285 | 1.633 | 1.009 | 3.307 | 8.676 | 1.347 | 1.641 |

## Differences

| comparison | d_fx_pct | d_fy_pct | d_cx_px | d_cy_px | d_k1 | d_k2 | d_mean_err_px | within_thresholds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| END TO END  ours vs OpenCV | 0.1361 | 0.1211 | -0.7781 | -0.007003 | -0.001953 | 0.008211 | 0.2216 | NO: mean_err |
| solver only (our corners): ours vs cv solver | 0.1557 | 0.1587 | -1.1 | -0.6316 | -0.003636 | 0.01286 | -0.03285 | yes |
| solver only (cv corners):  ours vs cv solver | 0.1626 | 0.1624 | -0.7601 | -0.5567 | -0.004661 | 0.01583 | -0.03131 | yes |
| corners only (cv solver):  ours vs cv corners | -0.01959 | -0.03754 | 0.3222 | 0.6246 | 0.001683 | -0.004646 | 0.2544 | NO: mean_err |

Thresholds: focal 1 %, principal point 2 px, mean error 0.05 px.

## Images not used

| image | n_ours | n_cv | used |
| :--- | ---: | ---: | ---: |
| A_5.jpeg | 2 | 8 | only 2 corners (< 12) |
