"""Pretend camera + pretend board.

We invent a camera whose numbers (focal length etc.) WE choose, "photograph"
a flat board with it, and hand back the corner positions. Because we know the
true answer, we can later check that our calibration code finds it.

Data format for one view (same format Kavan's real detector should produce):
    obj_pts : (N, 2)  corner positions on the board, in metres (board is flat, Z = 0)
    img_pts : (N, 2)  where those corners appear in the picture, in pixels
    ids     : (N,)    which corner each row is (so rows can be matched up)
"""
import numpy as np

# ---- the pretend board (same size as Kavan's printed 10x7-square board) ----
BOARD_COLS = 9         # inner corners along the long side (10 squares -> 9 corners)
BOARD_ROWS = 6         # inner corners along the short side (7 squares -> 6 corners)
SQUARE_SIZE = 0.035    # metres (35 mm)

# ---- the pretend camera picture size ----
IMAGE_W, IMAGE_H = 1280, 960


def true_camera_matrix():
    """The 'answer key': the camera numbers we want to recover later."""
    fx, fy = 1050.0, 1040.0      # focal lengths in pixels
    cx, cy = 645.0, 470.0        # image centre in pixels
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])


def board_points(cols=BOARD_COLS, rows=BOARD_ROWS, square=SQUARE_SIZE):
    """Corner grid on the board. Returns (N, 2) in metres, plus (N,) ids."""
    xs, ys = np.meshgrid(np.arange(cols) * square, np.arange(rows) * square)
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    ids = np.arange(len(pts))
    return pts, ids


def rotation_about_axis(axis, angle):
    """Rotation matrix for turning by `angle` (radians) around `axis`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def project_points(K, R, t, pts_xy, k1=0.0, k2=0.0):
    """Where do board points show up in the picture?

    R, t : where the board is relative to the camera (turn, then shift)
    k1,k2: lens bending (0 = perfect lens)
    """
    X = np.column_stack([pts_xy, np.zeros(len(pts_xy))])   # board is flat: Z = 0
    Xc = X @ R.T + t                                        # points in camera's view
    xn = Xc[:, :2] / Xc[:, 2:3]                             # divide by depth
    r2 = np.sum(xn ** 2, axis=1, keepdims=True)
    xd = xn * (1 + k1 * r2 + k2 * r2 ** 2)                  # lens bending
    return xd @ K[:2, :2].T + K[:2, 2]                      # to pixels


def random_pose(rng, K, pts_xy, min_tilt_deg=5.0, max_tilt_deg=40.0,
                min_dist=0.5, max_dist=0.9):
    """Pick a random board position where the whole board is visible.

    Returns R, t, tilt_deg.  tilt_deg = how far the board is turned away from
    facing the camera head-on (0 = perfectly flat-on).
    """
    board_w = pts_xy[:, 0].max()
    board_h = pts_xy[:, 1].max()
    centre = np.array([board_w / 2, board_h / 2, 0.0])
    margin = 10  # pixels of safety from the picture edge

    for _ in range(10000):
        tilt = np.deg2rad(rng.uniform(min_tilt_deg, max_tilt_deg))
        direction = rng.uniform(0, 2 * np.pi)               # which way it leans
        R_tilt = rotation_about_axis([np.cos(direction), np.sin(direction), 0], tilt)
        roll = np.deg2rad(rng.uniform(-30, 30))             # spin like a hand-held board
        R = R_tilt @ rotation_about_axis([0, 0, 1], roll)

        depth = rng.uniform(min_dist, max_dist)
        u = K[0, 2] + rng.uniform(-150, 150)                # roughly centred in picture
        v = K[1, 2] + rng.uniform(-100, 100)
        target = depth * np.array([(u - K[0, 2]) / K[0, 0],
                                   (v - K[1, 2]) / K[1, 1],
                                   1.0])
        t = target - R @ centre

        px = project_points(K, R, t, pts_xy)
        in_front = np.all((pts_xy @ R[:, :2].T + t)[:, 2] > 0)
        inside = (px[:, 0].min() > margin and px[:, 0].max() < IMAGE_W - margin and
                  px[:, 1].min() > margin and px[:, 1].max() < IMAGE_H - margin)
        if in_front and inside:
            return R, t, np.rad2deg(tilt)
    raise RuntimeError("could not place the board in view; relax the settings")


def make_views(n_views=15, noise_px=0.3, seed=0, k1=0.0, k2=0.0,
               min_tilt_deg=5.0, max_tilt_deg=40.0):
    """Make a pretend photo set.

    Returns (K_true, views). Each view is a dict with:
        obj_pts, img_pts, ids          -> what the calibration code gets to see
        img_pts_clean                  -> the same corners with NO noise
        R_true, t_true, tilt_deg       -> the answer key for this view
    """
    rng = np.random.default_rng(seed)
    K = true_camera_matrix()
    pts, ids = board_points()
    views = []
    for _ in range(n_views):
        R, t, tilt = random_pose(rng, K, pts, min_tilt_deg, max_tilt_deg)
        clean = project_points(K, R, t, pts, k1, k2)
        noisy = clean + rng.normal(0.0, noise_px, clean.shape)
        views.append(dict(obj_pts=pts.copy(), img_pts=noisy, ids=ids.copy(),
                          img_pts_clean=clean, R_true=R, t_true=t, tilt_deg=tilt))
    return K, views
