import numpy as np, cv2

def refine_corner(Gx, Gy, q, w=5, iters=30, eps=1e-3):
    """
    Sub-pixel corner refinement by gradient orthogonality.
    Gx, Gy : float32 gradient images of the whole picture
    q      : initial (x, y) guess, e.g. the predicted corner
    w      : half window size (window is (2w+1) x (2w+1) pixels)
    Returns (refined (x, y), smallest eigenvalue of the structure matrix).
    """
    xs = np.arange(-w, w + 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs)
    W = np.exp(-(X**2 + Y**2) / (2 * (w / 2) ** 2))      # Gaussian weight
    q = np.array(q, dtype=np.float64)

    for _ in range(iters):
        gx = cv2.getRectSubPix(Gx, (2*w + 1, 2*w + 1), (float(q[0]), float(q[1])))
        gy = cv2.getRectSubPix(Gy, (2*w + 1, 2*w + 1), (float(q[0]), float(q[1])))
        a, b, c = W * gx * gx, W * gx * gy, W * gy * gy
        A = np.array([[a.sum(), b.sum()],
                      [b.sum(), c.sum()]])
        rhs = np.array([(a * X + b * Y).sum(),
                        (b * X + c * Y).sum()])
        if np.linalg.det(A) < 1e-9:                       # flat region, no corner
            break
        dq = np.linalg.solve(A, rhs)                      # offset from current q
        q += dq
        if np.linalg.norm(dq) < eps:
            break

    return q, np.linalg.eigvalsh(A)[0]