# ChArUco Corner Detection and Refinement

This module implements a custom pipeline for estimating ChArUco chessboard corner locations from detected ArUco markers and refining the corner positions to sub-pixel accuracy.

The implementation is designed as part of the camera calibration pipeline, where accurate 2D image coordinates of ChArUco corners are required for homography estimation and Zhang's camera calibration method.

## Pipeline

The current implementation follows:

```text
Input Image
     │
     ▼
ArUco Marker Detection
     │
     ▼
Detected Marker IDs + Marker Corners
     │
     ▼
Board Geometry Lookup
     │
     ▼
Local Homography Estimation
     │
     ▼
Predict ChArUco Corner Locations
     │
     ▼
Gradient-Based Sub-Pixel Refinement
     │
     ▼
Refined ChArUco Corners
     │
     ├──────────────► Compare with OpenCV
     │
     ▼
Homography Residual Analysis
```

---

## Project Structure

The relevant project structure is:

```text
Camera-Callibration/
│
├── calib/
│   ├── __init__.py
│   ├── board.py
│   ├── predict.py
│   ├── refine.py
│   │
│   └── tests/
│       ├── __init__.py
│       └── test_real.py
│
├── data/
│   └── set_a/
│       └── A_18.jpeg
│
└── outputs/
    ├── real_compare.png
    └── residuals.png
```

### Main files

- `board.py` — Defines the known ChArUco board geometry.
- `predict.py` — Predicts chessboard corner locations using detected ArUco markers.
- `refine.py` — Performs gradient-based sub-pixel corner refinement.
- `tests/test_real.py` — Runs the complete pipeline on a real photograph and compares the result with OpenCV.

---

# 1. Board Model

`board.py` contains the `BoardModel` class.

The default board configuration is:

```python
BoardModel(
    squares=(10, 7),
    square_len=0.035,
    marker_len=0.026,
    dict_id=aruco.DICT_5X5_100
)
```

This represents a ChArUco board with:

- 10 squares horizontally
- 7 squares vertically
- 35 mm square length
- 26 mm marker length
- `DICT_5X5_100` ArUco dictionary

The board contains:

\[
(10-1)(7-1)=54
\]

inner ChArUco corners.

The model stores:

### ChArUco corner coordinates

```python
self.corners
```

These are the known 3D board coordinates:

```text
(X, Y, Z)
```

Since the board is planar:

\[
Z=0
\]

### ArUco marker coordinates

```python
self.marker_xy
```

This maps each marker ID to the known coordinates of its four marker corners in the board coordinate system.

---

# 2. Mapping Marker IDs to ChArUco Corners

The method:

```python
square_corner_ids(marker_id)
```

determines which ChArUco chessboard corners surround a particular ArUco marker.

For example:

```text
      ChArUco corner
             ●
             │
      ┌──────┼──────┐
      │      │      │
      │   ArUco     │
      │   marker    │
      │      │      │
      └──────┼──────┘
             │
             ●
```

The marker's known board coordinates are used to determine which board square contains the marker.

The corresponding surrounding ChArUco corner IDs are then obtained from the board geometry.

---

# 3. ChArUco Corner Prediction

`predict.py` implements:

```python
predict_corners(model, marker_corners, marker_ids)
```

The function takes:

- detected ArUco marker corners
- detected marker IDs
- the known board model

For every detected marker, a local homography is estimated:

$$
\mathbf{x} \sim H\mathbf{X}
$$

where:

- \(\mathbf{X}\) is a point in the board coordinate system
- \(\mathbf{x}\) is its image coordinate
- \(H\) is the planar homography

The homography is estimated using the four known corners of the detected marker:

```python
H, _ = cv2.findHomography(
    model.marker_xy[int(mid)],
    c.reshape(4, 2),
    0
)
```

The nearby ChArUco corners are then projected into the image using:

```python
cv2.perspectiveTransform(...)
```

If a ChArUco corner can be predicted from multiple detected markers, all predictions are averaged.

This reduces the effect of individual marker localization errors.

---

# 4. Sub-Pixel Corner Refinement

`refine.py` implements:

```python
refine_corner(Gx, Gy, q, w=5)
```

where:

- `Gx` is the horizontal image gradient
- `Gy` is the vertical image gradient
- `q` is the initial corner estimate

The initial prediction from the marker-based homography is treated as the starting point.

A local window is extracted around the corner using:

```python
cv2.getRectSubPix(...)
```

This allows the algorithm to operate at non-integer pixel positions.

The local gradients are used to construct the structure matrix:

$$
A =
\begin{bmatrix}
\sum wG_x^2 & \sum wG_xG_y \\
\sum wG_xG_y & \sum wG_y^2
\end{bmatrix}
$$


The corner position is iteratively updated by solving:

$$
A\Delta q = b
$$

until the correction becomes sufficiently small.

The smallest eigenvalue of the structure matrix is also returned as a measure of the local corner strength.

The refinement therefore produces:

```text
Initial prediction
       │
       ▼
Local gradient window
       │
       ▼
Structure matrix
       │
       ▼
Estimate sub-pixel correction
       │
       ▼
Update corner position
       │
       ▼
Repeat until convergence
       │
       ▼
Refined corner
```

---

# 5. Real Image Test

The script:

```text
calib/tests/test_real.py
```

tests the implementation on a photograph of the printed ChArUco board.

The image is loaded using:

```python
gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

The current test image is:

```text
data/set_a/A_18.jpeg
```

The board configuration must match the physical printed board.

---

# 6. ArUco Marker Detection

The current implementation uses OpenCV's ArUco detector:

```python
mc, mids, _ = aruco.ArucoDetector(
    model.dictionary,
    params
).detectMarkers(gray)
```

This provides:

- marker corner locations
- marker IDs

These detected markers are then passed to the custom corner prediction stage.

---

# 7. Our Corner Detection Pipeline

The detected markers are passed to:

```python
pred = predict_corners(model, mc, mids)
```

The predicted corners are then refined:

```python
for cid, p in pred.items():
    q, lam = refine_corner(Gx, Gy, p, w=5)

    if np.linalg.norm(q - p) < 3.0:
        ours[cid] = q
```

The `3.0` pixel condition rejects cases where refinement moves too far from the initial prediction.

This prevents unstable refinement from producing unreasonable corner locations.

---

# 8. OpenCV Reference

For comparison, OpenCV's ChArUco detector is run on the same image:

```python
cv_board = aruco.CharucoBoard(
    (10, 7),
    model.sq,
    0.026,
    model.dictionary
)

det = aruco.CharucoDetector(cv_board)

cc, cids, _, _ = det.detectBoard(gray)
```

The detected corners are stored using their ChArUco IDs.

This allows a direct comparison between:

```text
Our method
    vs.
OpenCV
```

using the corner IDs rather than relying on their ordering.

---

# 9. Corner Localization Error

Only corners detected by both methods are compared:

```python
common = sorted(set(ours) & set(ref))
```

For every common corner:

\[
e_i =
\sqrt{
(x_i^{ours}-x_i^{opencv})^2+
(y_i^{ours}-y_i^{opencv})^2
}
\]

The script reports:

- mean distance
- median distance
- 95th percentile
- maximum distance
- mean `(dx, dy)` offset

Example output:

```text
ours 42 | opencv 45 | common 40
distance px: mean 0.XXX median 0.XXX 95% 0.XXX max 0.XXX
mean offset (dx, dy): [...]
```

These measurements indicate how closely our corner locations agree with OpenCV's reference implementation.

---

# 10. Visualization

The script generates:

```text
real_compare.png
```

The visualization uses:

- **Green circles** — OpenCV ChArUco corners
- **Red dots** — corners produced by our method

Ideally, the red dots should lie very close to the green circles.

Large separations indicate localization errors.

---

# 11. Homography Residual Analysis

The script also evaluates how well the detected corners conform to a single planar homography.

For the detected board points:

```python
X = model.corners[ids_][:, :2]
```

and image points:

```python
x = np.array([ours[c] for c in ids_])
```

a homography is estimated:

```python
H, _ = cv2.findHomography(X, x, 0)
```

The predicted image points are:

\[
\hat{x}_i = H X_i
\]

The residual is:

\[
r_i = x_i-\hat{x}_i
\]

and its magnitude is:

\[
\|r_i\|_2
\]

The script reports the mean and maximum residual.

The same analysis is performed for OpenCV's corners.

---

# 12. Running the Test

The project should be run from the **project root**, not from inside the `calib` directory.

From:

```text
Camera-Callibration/
```

run:

```bash
python -m calib.tests.test_real
```

The package structure includes `__init__.py` files so that Python can resolve:

```python
from calib.board import BoardModel
from calib.predict import predict_corners
from calib.refine import refine_corner
```

correctly.

---

# 13. Path Handling

The test should preferably use `pathlib` so that input/output paths do not depend on the directory from which Python is launched.

For example:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

path = ROOT / "data" / "set_a" / "A_18.jpeg"

gray = cv2.imread(
    str(path),
    cv2.IMREAD_GRAYSCALE
)
```

Outputs can similarly be stored in:

```text
outputs/
```

rather than being written into the current working directory.

---

# 14. Relation to Camera Calibration

The current module is the **corner localization stage** of the larger camera calibration pipeline.

The complete pipeline will eventually be:

```text
ChArUco Board Images
        │
        ▼
ArUco Marker Detection
        │
        ▼
ChArUco Corner Localization
        │
        ▼
Sub-Pixel Corner Refinement
        │
        ▼
2D–3D Correspondences
        │
        ▼
Planar Homographies
        │
        ▼
Zhang's Calibration
        │
        ▼
Initial Camera Intrinsics
        │
        ▼
Camera Poses
        │
        ▼
Radial Distortion + Nonlinear Refinement
        │
        ▼
Final Camera Matrix + Distortion
```

The output of this module is therefore a set of accurate correspondences:

\[
(X_i,Y_i,0)
\leftrightarrow
(u_i,v_i)
\]

which can subsequently be used for homography estimation and camera calibration.