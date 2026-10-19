"""Homography: the single rule that maps a flat board to its photo.

A homography H is a 3x3 grid of numbers. For a spot (x, y) on the board:
    1. write it as (x, y, 1)
    2. multiply by H           -> (a, b, w)
    3. divide by w             -> (u, v)  = where the spot lands in the photo

We find H from many matching pairs (board spot, photo spot) in three parts:
    normalize_points    rescale the numbers so the maths stays well-behaved
    dlt_homography      solve for H in one go (the "Direct Linear Transform")
    refine_homography   polish H with a few small nudges that cut leftover error
estimate_homography() does all three and is the one to call.
"""
import numpy as np


def apply_homography(H, pts):
    """Send (N, 2) board spots through H. Returns (N, 2) photo spots."""
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def reprojection_errors(H, src, dst):
    """How far (in pixels) each mapped spot lands from where it was actually seen."""
    return np.linalg.norm(apply_homography(H, src) - dst, axis=1)


def normalize_points(pts):
    """Shift the points so their centre is (0, 0), then scale them so the
    average distance from the centre is sqrt(2).

    Returns (T, new_pts): T is the 3x3 shift-and-scale grid, new_pts the result.
    """
    pts = np.asarray(pts, dtype=float)
    centre = pts.mean(axis=0)
    mean_dist = np.linalg.norm(pts - centre, axis=1).mean()
    if mean_dist < 1e-12:
        raise ValueError("all points are identical; cannot find a homography")
    s = np.sqrt(2) / mean_dist
    T = np.array([[s, 0, -s * centre[0]],
                  [0, s, -s * centre[1]],
                  [0, 0, 1.0]])
    return T, (pts - centre) * s


def has_four_in_general_position(pts, tol=1e-3):
    """True if the points include 4 with no 3 in a straight line.

    A homography has 8 unknown numbers and each point pair gives 2 equations, so it
    needs 4 points -- but only if they are spread out. Five points on one line plus
    one more is NOT enough, however many points that is.
    (A quick greedy search, not a full proof; `tol` is a fraction of the points' spread.)
    """
    _, s = normalize_points(pts)

    def dist_to_line(p, q):
        d = q - p
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        return np.abs((s - p) @ n)

    a = s[0]
    b = s[np.argmax(np.linalg.norm(s - a, axis=1))]            # farthest from a
    d_ab = dist_to_line(a, b)
    if d_ab.max() < tol:
        return False                                           # everything is on one line
    c = s[np.argmax(d_ab)]                                     # farthest from the line a-b
    nearest_line = np.minimum(np.minimum(d_ab, dist_to_line(a, c)), dist_to_line(b, c))
    return bool(nearest_line.max() > tol)                      # a 4th point off all three lines


def dlt_homography(src, dst):
    """Find H in one go (no polishing). src, dst: (N, 2) matching spots, N >= 4."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    if n < 4 or len(dst) != n:
        raise ValueError("need at least 4 matching point pairs")
    if not has_four_in_general_position(src):
        raise ValueError("need 4 board points with no 3 in a straight line (general position)")

    Ts, s = normalize_points(src)
    Td, d = normalize_points(dst)

    # Each pair (x, y) -> (u, v) gives two equations that are linear in the 9 numbers of H.
    P = np.column_stack([s, np.ones(n)])             # rows (x, y, 1)
    A = np.zeros((2 * n, 9))
    A[0::2, 0:3] = P
    A[0::2, 6:9] = -d[:, 0:1] * P
    A[1::2, 3:6] = P
    A[1::2, 6:9] = -d[:, 1:2] * P

    # The best H is the direction that A "squashes" the most: the last row of Vt.
    _, sing, Vt = np.linalg.svd(A)
    if sing[-2] < 1e-9 * sing[0]:
        raise ValueError("points are (nearly) in a straight line; cannot find a unique H")
    Hn = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(Td) @ Hn @ Ts                  # undo the rescaling
    return H / H[2, 2]


def refine_homography(H, src, dst, iters=10):
    """Polish H by repeatedly nudging it to cut the pixel error (Gauss-Newton).

    Works in rescaled numbers for stability. A nudge is kept only if it helps.
    """
    Ts, s = normalize_points(src)
    Td, d = normalize_points(dst)
    Hn = Td @ H @ np.linalg.inv(Ts)
    Hn = Hn / Hn[2, 2]
    h = Hn.ravel()[:8].copy()                        # 8 free numbers (the 9th is fixed at 1)
    P = np.column_stack([s, np.ones(len(s))])

    def residual(h):
        Hm = np.append(h, 1.0).reshape(3, 3)
        return (apply_homography(Hm, s) - d).ravel()

    with np.errstate(all="ignore"):                  # a broken H may divide by ~0; we handle that below
        cost = np.sum(residual(h) ** 2)
    if not np.isfinite(cost):
        return H                                     # the starting H is unusable; polishing cannot help
    for _ in range(iters):
        Hm = np.append(h, 1.0).reshape(3, 3)
        q = P @ Hm.T
        w = q[:, 2]
        if np.any(np.abs(w) < 1e-12):
            break                                    # a point maps to "infinity": stop polishing here
        u, v = q[:, 0] / w, q[:, 1] / w
        J = np.zeros((2 * len(s), 8))                # how the error changes if each of the 8 numbers moves
        J[0::2, 0:3] = P / w[:, None]
        J[0::2, 6:8] = -(u / w)[:, None] * P[:, :2]
        J[1::2, 3:6] = P / w[:, None]
        J[1::2, 6:8] = -(v / w)[:, None] * P[:, :2]
        step = np.linalg.lstsq(J, -residual(h), rcond=None)[0]
        scale = 1.0
        while scale > 1e-4:                          # shrink the nudge until it really helps
            with np.errstate(all="ignore"):
                new_cost = np.sum(residual(h + scale * step) ** 2)
            if new_cost < cost:               # (a nan/inf cost is never "better", so it is skipped)
                h, cost = h + scale * step, new_cost
                break
            scale /= 2
        else:
            break                                    # no helpful nudge left: we are done
        if np.linalg.norm(scale * step) < 1e-12:
            break

    Hn = np.append(h, 1.0).reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def estimate_homography(src, dst, refine=True):
    """The one to call. src: (N, 2) board spots, dst: (N, 2) photo spots."""
    H = dlt_homography(src, dst)
    return refine_homography(H, src, dst) if refine else H
