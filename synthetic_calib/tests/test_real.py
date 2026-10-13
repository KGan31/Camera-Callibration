import cv2, numpy as np, sys
from calib.board import BoardModel
from calib.predict import predict_corners
from calib.refine import refine_corner
aruco = cv2.aruco

path = "data/set_a/A_18.jpeg" # printed board, photographed with a phone
gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
model = BoardModel()          # must match the printed board's layout and dictionary

# --- ours ---
params = aruco.DetectorParameters()
mc, mids, _ = aruco.ArucoDetector(model.dictionary, params).detectMarkers(gray)
if mids is None:
    sys.exit("no markers found")
pred = predict_corners(model, mc, mids)

g = gray.astype(np.float32)
Gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
Gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
ours, ours_pred = {}, pred
# ours = {}
for cid, p in pred.items():
    q, lam = refine_corner(Gx, Gy, p, w=5)
    if np.linalg.norm(q - p) < 3.0:        # reject if it wandered off the prediction
        ours[cid] = q

# --- OpenCV reference ---
cv_board = aruco.CharucoBoard((10, 7), model.sq, 0.026, model.dictionary)
det = aruco.CharucoDetector(cv_board)
cc, cids, _, _ = det.detectBoard(gray)
ref = {} if cids is None else {int(i): c.ravel() for i, c in zip(cids.flatten(), cc)}

# --- compare by ID ---
common = sorted(set(ours) & set(ref))
d = np.array([ours[c] - ref[c] for c in common])
dist = np.linalg.norm(d, axis=1)
print(f"ours {len(ours)} | opencv {len(ref)} | common {len(common)}")
print(f"distance px: mean {dist.mean():.3f}  median {np.median(dist):.3f}  "
      f"95% {np.percentile(dist, 95):.3f}  max {dist.max():.3f}")
print(f"mean offset (dx, dy): {d.mean(axis=0).round(3)}")

# --- overlay ---
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for c in common:
    cv2.circle(vis, tuple(np.round(ref[c]).astype(int)), 7, (0, 200, 0), 1)    # OpenCV: green ring
    cv2.circle(vis, tuple(np.round(ours[c]).astype(int)), 2, (0, 0, 255), -1)  # ours: red dot
cv2.imwrite("real_compare.png", vis)


ids_ = sorted(ours)
X = model.corners[ids_][:, :2]
x = np.array([ours[c] for c in ids_])
H, _ = cv2.findHomography(X, x, 0)
res = np.linalg.norm(cv2.perspectiveTransform(X.reshape(-1, 1, 2), H).reshape(-1, 2) - x, axis=1)
print(f"homography residual px: mean {res.mean():.3f}  max {res.max():.3f}")
