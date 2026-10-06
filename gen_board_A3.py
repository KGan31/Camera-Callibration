import cv2
from PIL import Image

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
board = aruco.CharucoBoard((10, 7), 0.035, 0.026, dictionary)  # squares, square_len(m), marker_len(m)

dpi = 300
w_px = round(0.350 / 0.0254 * dpi)   # 350 mm wide
h_px = round(0.245 / 0.0254 * dpi)   # 245 mm tall
img = board.generateImage((w_px, h_px), marginSize=0, borderBits=1)
cv2.imwrite("charuco_10x7.png", img)
pil = Image.fromarray(img).convert("1", dither=Image.Dither.NONE)
pil.save("charuco_10x7.pdf", resolution=dpi)