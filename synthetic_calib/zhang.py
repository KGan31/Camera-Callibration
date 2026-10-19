"""Zhang's method: find the camera's own numbers from several homographies.

Every photo gives a homography H = K [r1 r2 t] (up to scale), where K holds the camera
numbers and (r1, r2, t) say where the board stands. The board's two edge directions
r1 and r2 are at right angles and equal in length, which gives TWO equations about K
alone, per photo. Stack them over several photos and solve them all at once.

    zhang_intrinsics            homographies -> camera matrix K
    poses_from_homographies     K + homographies -> where the board stood in each photo
    calibrate_closed_form       points of several views -> K and poses (calls both above)

(Camera model: zero skew, no lens distortion. Distortion is added in the next step.)
"""
import numpy as np
from synthetic_calib.homography import estimate_homography


def _v(H, i, j):
    """The 6 numbers that build the equation linking columns i and j of H."""
    hi, hj = H[:, i], H[:, j]
    return np.array([hi[0] * hj[0],
                     hi[0] * hj[1] + hi[1] * hj[0],
                     hi[1] * hj[1],
                     hi[2] * hj[0] + hi[0] * hj[2],
                     hi[2] * hj[1] + hi[1] * hj[2],
                     hi[2] * hj[2]])


def zhang_intrinsics(Hs, image_size, zero_skew=True):
    """Camera matrix K from a list of homographies (board -> photo, in pixels).

    image_size = (width, height) of the photos. Only used to rescale the numbers
    so the maths stays well-behaved (the answer does not depend on it).
    """
    if len(Hs) < 3:
        raise ValueError("need at least 3 views")
    w, h = image_size
    s = (w + h) / 2.0
    N = np.array([[1 / s, 0, -w / 2 / s],        # moves the picture centre to (0,0) and shrinks pixels
                  [0, 1 / s, -h / 2 / s],
                  [0, 0, 1.0]])

    rows = []
    for H in Hs:
        Hn = N @ H
        Hn = Hn / np.linalg.norm(Hn)             # so no single photo shouts louder than the others
        rows.append(_v(Hn, 0, 1))                # r1 and r2 are at right angles
        rows.append(_v(Hn, 0, 0) - _v(Hn, 1, 1))  # r1 and r2 have equal length
    if zero_skew:
        rows.append(np.array([0, 1.0, 0, 0, 0, 0]))   # "B12 = 0": the camera has no skew
    V = np.array(rows)

    _, sing, Vt = np.linalg.svd(V)
    if sing[-2] < 1e-8 * sing[0]:
        raise ValueError("views are too similar (not enough variety in tilt): cannot find K")
    b = Vt[-1]
    if b[0] < 0:                                  # B must be "positive": fix the arbitrary sign
        b = -b
    B11, B12, B22, B13, B23, B33 = b

    det = B11 * B22 - B12 ** 2
    if det <= 0:
        raise ValueError("no valid camera found (the equations gave an impossible answer)")
    v0 = (B12 * B13 - B11 * B23) / det
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    if lam / B11 <= 0 or lam * B11 / det <= 0:
        raise ValueError("no valid camera found (negative focal length)")
    alpha = np.sqrt(lam / B11)                    # fx (in rescaled units)
    beta = np.sqrt(lam * B11 / det)               # fy
    if zero_skew:
        gamma = 0.0
        u0 = -B13 * alpha ** 2 / lam
    else:
        gamma = -B12 * alpha ** 2 * beta / lam
        u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam
    Kn = np.array([[alpha, gamma, u0],
                   [0, beta, v0],
                   [0, 0, 1.0]])
    return np.linalg.inv(N) @ Kn                  # back to pixels


def poses_from_homographies(K, Hs):
    """For each H, where the board stood: returns a list of (R, t)."""
    Kinv = np.linalg.inv(K)
    poses = []
    for H in Hs:
        M = Kinv @ H                              # = scale * [r1 r2 t]
        scale = 1.0 / np.linalg.norm(M[:, 0])
        if (scale * M[2, 2]) < 0:                 # the board must be IN FRONT of the camera
            scale = -scale
        r1, r2, t = scale * M[:, 0], scale * M[:, 1], scale * M[:, 2]
        Q = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(Q)               # nearest true rotation (noise makes Q slightly off)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        poses.append((R, t))
    return poses


def calibrate_closed_form(views, image_size, zero_skew=True):
    """Points of several views -> (K, poses, Hs).

    views : list of dicts with 'obj_pts' (board, metres) and 'img_pts' (photo, pixels)
    """
    Hs = [estimate_homography(v["obj_pts"], v["img_pts"]) for v in views]
    K = zhang_intrinsics(Hs, image_size, zero_skew)
    return K, poses_from_homographies(K, Hs), Hs
