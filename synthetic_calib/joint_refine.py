"""Joint refinement: polish EVERYTHING together, including the lens bending.

Zhang's method (zhang.py) gives a good first answer, but it ignores lens bending and it
finds each part separately. Here we adjust all the unknowns at the same time:

    * the camera numbers      fx, fy, cx, cy
    * the lens bending        k1, k2
    * every photo's board position (a turn and a shift per photo)

until the total miss -- how far each corner's PREDICTED photo position is from where it was
actually SEEN -- is as small as possible. That total miss is called the reprojection error.

    rodrigues / rotation_to_rvec   turn <-> a compact 3-number form of a turn
    project                        where a board spot lands in the photo (with lens bending)
    init_distortion                a first guess for k1, k2 (a simple straight-line fit)
    refine                         the polishing itself
    calibrate_full                 the whole pipeline: corners in, calibration out
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from synthetic_calib.zhang import calibrate_closed_form


# ---------------------------------------------------------------- turns
def rodrigues(rvec):
    """3 numbers -> 3x3 turn matrix. The direction of rvec is the axis, its length the angle."""
    rvec = np.asarray(rvec, dtype=float)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = rvec / theta
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)


def rotation_to_rvec(R):
    """3x3 turn matrix -> 3 numbers (the reverse of rodrigues)."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * skew                                    # almost no turn
    if np.pi - theta > 1e-3:
        return skew / (2 * np.sin(theta)) * theta            # the usual case
    S = (R + R.T) / 2                                        # almost half a turn: the usual formula
    B = (S - cos * np.eye(3)) / (1 - cos)                    # breaks, so rebuild axis*axis^T exactly
    i = np.argmax(np.diag(B))
    axis = B[:, i] / np.sqrt(B[i, i])
    if axis @ skew < 0:
        axis = -axis
    return axis * theta


# ---------------------------------------------------------------- the camera model
def project(K, dist, R, t, pts_xy):
    """Board spots (N, 2) in metres -> photo spots (N, 2) in pixels, with lens bending."""
    X = np.column_stack([pts_xy, np.zeros(len(pts_xy))])
    Xc = X @ R.T + t                                         # into the camera's point of view
    xn = Xc[:, :2] / Xc[:, 2:3]                              # farther things look smaller
    r2 = np.sum(xn ** 2, axis=1, keepdims=True)
    xd = xn * (1 + dist[0] * r2 + dist[1] * r2 ** 2)         # lens bending (k1, k2)
    return xd @ K[:2, :2].T + K[:2, 2]                       # to pixels


def init_distortion(K, poses, views):
    """A first guess for (k1, k2) with K and every pose held fixed.

    Without lens bending a corner would land at an "ideal" pixel; with it, it lands somewhere
    else. That gap is (ideal - centre) * (k1 r^2 + k2 r^4), which is a straight-line problem
    in k1 and k2, solved here by least squares.
    """
    A, b = [], []
    for (R, t), v in zip(poses, views):
        ideal = project(K, (0.0, 0.0), R, t, v["obj_pts"])
        Xc = np.column_stack([v["obj_pts"], np.zeros(len(ideal))]) @ R.T + t
        r2 = np.sum((Xc[:, :2] / Xc[:, 2:3]) ** 2, axis=1)
        d = ideal - K[:2, 2]
        gap = v["img_pts"] - ideal
        for axis in (0, 1):
            A.append(np.column_stack([d[:, axis] * r2, d[:, axis] * r2 ** 2]))
            b.append(gap[:, axis])
    k, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    return k


# ---------------------------------------------------------------- the result
@dataclass
class CalibrationResult:
    K: np.ndarray            # camera matrix (fx, fy, cx, cy)
    dist: np.ndarray         # lens bending (k1, k2)
    poses: list              # [(R, t), ...] one per photo
    mean_error: float        # average miss per corner, in pixels
    rms_error: float         # root-mean-square miss (what OpenCV reports), in pixels
    view_errors: np.ndarray  # average miss for each photo
    cost: float              # the number being minimised (lower = better fit)
    converged: bool
    n_evals: int
    K_closed_form: np.ndarray = None   # Zhang's rough answer, kept for comparison


def _errors(K, dist, poses, views):
    """Per-corner miss distances (pixels), one array per photo."""
    return [np.linalg.norm(project(K, dist, R, t, v["obj_pts"]) - v["img_pts"], axis=1)
            for (R, t), v in zip(poses, views)]


def _result(K, dist, poses, views, opt, K_closed_form=None):
    errs = _errors(K, dist, poses, views)
    allerr = np.concatenate(errs)
    return CalibrationResult(K=K, dist=np.asarray(dist, dtype=float), poses=poses,
                             mean_error=float(allerr.mean()), rms_error=float(np.sqrt(np.mean(allerr ** 2))),
                             view_errors=np.array([e.mean() for e in errs]), cost=float(opt.cost),
                             converged=bool(opt.success), n_evals=int(opt.nfev), K_closed_form=K_closed_form)


# ---------------------------------------------------------------- the polishing
def corner_weights(misses, loss, f_scale):
    """How much each corner is trusted, from how far it misses (in pixels).

    "linear" : every corner counts fully (ordinary least squares)
    "huber"  : corners missing by more than f_scale count less and less (weight f_scale / miss)
    "cauchy" : misses much larger than f_scale are almost ignored (weight 1 / (1 + (miss/f_scale)^2))
    """
    if loss == "linear":
        return np.ones_like(misses)
    if loss == "huber":
        return np.where(misses <= f_scale, 1.0, f_scale / np.maximum(misses, 1e-12))
    if loss == "cauchy":
        return 1.0 / (1.0 + (misses / f_scale) ** 2)
    raise ValueError("loss must be 'linear', 'huber' or 'cauchy'")


def refine(K0, dist0, poses0, views, fix_distortion=False, loss="linear", f_scale=1.0,
           max_evals=300, max_rounds=10):
    """Adjust everything at once to shrink the total miss.

    fix_distortion : keep (k1, k2) fixed at dist0 instead of adjusting them
    loss           : "linear" counts every corner equally. "huber" / "cauchy" trust corners
                     that miss by a lot less (see corner_weights), so a few wildly wrong
                     corners cannot drag the answer. Done by "reweighting": fit, re-score every
                     corner's trust from its miss, fit again, and so on until nothing changes.
    """
    n = len(views)
    n_cam = 4 if fix_distortion else 6                       # unknowns shared by all photos
    pts = [v["obj_pts"] for v in views]
    obs = [v["img_pts"] for v in views]

    def unpack(p):
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
        dist = np.asarray(dist0, dtype=float) if fix_distortion else p[4:6]
        poses = [(rodrigues(p[n_cam + 6 * i: n_cam + 6 * i + 3]), p[n_cam + 6 * i + 3: n_cam + 6 * i + 6])
                 for i in range(n)]
        return K, dist, poses

    def raw_residuals(p):
        K, dist, poses = unpack(p)
        return np.concatenate([(project(K, dist, R, t, X) - u).ravel() for (R, t), X, u in zip(poses, pts, obs)])

    x = [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]]
    if not fix_distortion:
        x += list(dist0)
    for R, t in poses0:
        x += list(rotation_to_rvec(R)) + list(t)
    x = np.array(x, dtype=float)

    # Each photo's misses depend only on the shared numbers and on THAT photo's own pose.
    # Telling the optimiser this makes it much faster. (Photos may have different numbers of
    # corners, for example when part of the board is hidden.)
    sizes = [2 * len(X) for X in pts]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    S = lil_matrix((int(starts[-1]), len(x)), dtype=int)
    for i in range(n):
        S[starts[i]:starts[i + 1], :n_cam] = 1
        S[starts[i]:starts[i + 1], n_cam + 6 * i: n_cam + 6 * i + 6] = 1

    w = np.ones(int(starts[-1]) // 2)                        # one trust value per corner
    for round_ in range(1 if loss == "linear" else max_rounds):
        sqrt_w = np.repeat(np.sqrt(w), 2)
        opt = least_squares(lambda p: raw_residuals(p) * sqrt_w, x, jac_sparsity=S, x_scale="jac", max_nfev=max_evals)
        x = opt.x
        misses = np.linalg.norm(raw_residuals(x).reshape(-1, 2), axis=1)
        new_w = corner_weights(misses, loss, f_scale)
        change = np.abs(new_w - w).max()
        w = new_w
        if loss == "linear" or change < 1e-3:
            break
    K, dist, poses = unpack(x)
    return _result(K, dist, poses, views, opt)


def calibrate_full(views, image_size, estimate_distortion=True, staged=True, robust=True,
                   robust_loss="huber", f_scale=1.0):
    """The whole pipeline: views of corner points in, finished calibration out.

    1. a homography per photo, then Zhang's rough camera and board positions
    2. (if staged) polish with the lens bending held at zero; this settles the rest first
       and stops the lens bending from being confused with the focal length
    3. a first guess for the lens bending, then polish everything with plain least squares
    4. (if robust) a final polish with the robust loss, so a few wildly wrong corners cannot
       drag the answer. This comes LAST on purpose: at the start, the unmodelled lens bending
       makes honest corners near the picture edge look like outliers, and the robust loss
       would wrongly ignore them.
    """
    K0, poses0, _ = calibrate_closed_form(views, image_size)
    if not estimate_distortion:
        res = refine(K0, (0.0, 0.0), poses0, views, fix_distortion=True, loss="linear")
        if robust:
            res = refine(res.K, (0.0, 0.0), res.poses, views, fix_distortion=True, loss=robust_loss, f_scale=f_scale)
    else:
        K, poses = K0, poses0
        if staged:
            first = refine(K0, (0.0, 0.0), poses0, views, fix_distortion=True, loss="linear")
            K, poses = first.K, first.poses
        res = refine(K, init_distortion(K, poses, views), poses, views, loss="linear")
        if robust:
            res = refine(res.K, res.dist, res.poses, views, loss=robust_loss, f_scale=f_scale)
    res.K_closed_form = K0
    return res
