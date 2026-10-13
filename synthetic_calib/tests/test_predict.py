import cv2, numpy as np
from calib.board import BoardModel
from calib.predict import predict_corners
from calib.refine import refine_corner
aruco = cv2.aruco

model = BoardModel()
board = aruco.CharucoBoard((10, 7), model.sq, 0.026, model.dictionary)
img = board.generateImage((4000, 2800), marginSize=0, borderBits=1)   # 400 px / square
h, w = img.shape

src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst = np.float32([[300, 250], [1500, 330], [1450, 1000], [350, 900]])
H_true = cv2.getPerspectiveTransform(src, dst)
canvas = cv2.warpPerspective(img, H_true, (1800, 1250), borderValue=200)
canvas = cv2.GaussianBlur(canvas, (0, 0), 1.0)

params = aruco.DetectorParameters()
# params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
corners, ids, _ = aruco.ArucoDetector(model.dictionary, params).detectMarkers(canvas)

pred = predict_corners(model, corners, ids)


g = img.astype(np.float32)
Gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
Gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
ours, ours_pred = {}, pred
# ours = {}
cids = sorted(pred)
for cid, p in pred.items():
    q, lam = refine_corner(Gx, Gy, p, w=5)
    if np.linalg.norm(q - p) < 3.0:        # reject if it wandered off the prediction
        ours[cid] = q

# Ground truth: board coords -> source pixels (pixel centres at integers, hence -0.5) -> warp

xy = model.corners[cids][:, :2] / model.sq * 400 - 0.5
truth = cv2.perspectiveTransform(xy.reshape(-1, 1, 2).astype(np.float64),
                                 H_true.astype(np.float64)).reshape(-1, 2)
err = np.linalg.norm(np.array([ours[c] for c in cids]) - truth, axis=1)
print(f"{len(cids)} corners | error px: mean {err.mean():.3f}  max {err.max():.3f}")

vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
for p, t in zip(np.array([ours[c] for c in cids]), truth):
    cv2.circle(vis, tuple(np.round(t).astype(int)), 6, (0, 200, 0), 1)   # truth: green
    cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (0, 0, 255), -1)  # predicted: red
cv2.imwrite("synthetic_predict.png", vis)