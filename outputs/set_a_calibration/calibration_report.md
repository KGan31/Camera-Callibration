# Camera Calibration Report — Set A

## Pipeline

1. ArUco detection
2. Per-marker homography corner prediction (`calib/predict.py`)
3. Sub-pixel gradient orthogonality refinement (`calib/refine.py`)
4. Adaptive drift-threshold rejection
5. Zhang closed-form (`synthetic_calib/zhang.py`)
6. Joint LM refinement with k1, k2 (`synthetic_calib/joint_refine.py`)

## Results

| Parameter | Zhang (Ours) | LM Refined (Ours) | OpenCV |
| :--- | ---: | ---: | ---: |
| fx (px) | 3109.5900 | 3158.2481 | 3121.5873 |
| fy (px) | 3080.5258 | 3122.3459 | 3101.5603 |
| cx (px) | 1732.2952 | 1735.7708 | 1823.4188 |
| cy (px) | 1633.8304 | 1635.4501 | 1663.3863 |
| k1      | —     | -0.179530 | -0.1352 |
| k2      | —     | 0.217415 | 0.0800 |
| RMS reproj (px) | — | 10.5592 | 9.6845 |

## Per-View Reprojection (Our Pipeline)

| Image | Corners | Mean Error (px) |
| :--- | :---: | :---: |
| `A_1.jpeg` | 54 | 8.5782 |
| `A_2.jpeg` | 54 | 9.6514 |
| `A_3.jpeg` | 53 | 9.8019 |
| `A_4.jpeg` | 54 | 4.2778 |
| `A_5.jpeg` | 14 | 7.3171 |
| `A_6.jpeg` | 54 | 3.4179 |
| `A_7.jpeg` | 54 | 4.3724 |
| `A_8.jpeg` | 54 | 13.0136 |
| `A_9.jpeg` | 54 | 10.3346 |
| `A_10.jpeg` | 54 | 6.2316 |
| `A_11.jpeg` | 54 | 10.2111 |
| `A_12.jpeg` | 48 | 4.3899 |
| `A_13.jpeg` | 48 | 3.2610 |
| `A_14.jpeg` | 54 | 2.9685 |
| `A_15.jpeg` | 54 | 9.2851 |
| `A_16.jpeg` | 54 | 10.6156 |
| `A_17.jpeg` | 54 | 8.2786 |
| `A_18.jpeg` | 54 | 12.1134 |
| `A_19.jpeg` | 54 | 4.0222 |
| `A_20.jpeg` | 52 | 3.0018 |
| `A_21.jpeg` | 53 | 3.5287 |
| `A_22.jpeg` | 54 | 4.6696 |
| `A_23.jpeg` | 54 | 11.2046 |
| `A_24.jpeg` | 53 | 6.1323 |
| `A_25.jpeg` | 54 | 10.7239 |
| `A_26.jpeg` | 54 | 5.0115 |
| `A_27.jpeg` | 54 | 9.0018 |
| `A_28.jpeg` | 54 | 8.2507 |
| `A_29.jpeg` | 54 | 11.9946 |
| `A_30.jpeg` | 50 | 6.2690 |
| `A_31.jpeg` | 54 | 5.0812 |
