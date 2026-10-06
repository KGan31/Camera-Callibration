import cv2, numpy as np
aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
board = aruco.CharucoBoard((10, 7), 0.035, 0.026, dictionary)
# board = aruco.CharucoBoard((8, 5), 0.030, 0.022, dictionary)   # 240 x 150 mm

# img = cv2.imread("charuco_8x5.png", cv2.IMREAD_GRAYSCALE)
# h, w = img.shape

# Place the board on a larger canvas with a known perspective warp
# src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
# dst = np.float32([[300, 250], [1500, 330], [1450, 1000], [350, 900]])
# H_true = cv2.getPerspectiveTransform(src, dst)
# canvas = cv2.warpPerspective(img, H_true, (1800, 1250), borderValue=200)
canvas = cv2.imread("data/set_c/C_1.jpeg", cv2.IMREAD_GRAYSCALE )

# Detect markers and draw their IDs
detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(canvas)
print("markers found:", 0 if ids is None else len(ids), "of", len(board.getIds()))

vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
aruco.drawDetectedMarkers(vis, corners, ids)
cv2.imwrite("synthetic_detect.png", vis)