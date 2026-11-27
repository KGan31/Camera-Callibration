import numpy as np
from synthetic_calib.homography import estimate_homography, apply_homography

def predict_corners(model, marker_corners, marker_ids):
    """Returns {corner_id: (u, v)} averaged over every detected marker that sees it."""
    """
    model          : BoardModel instance
    marker_corners : list of detected marker corners
    marker_ids     : list of marker IDs

    Returns a dict mapping corner_id -> (u, v) in pixels.
    """

    acc = {}
    for c, mid in zip(marker_corners, marker_ids.flatten()):
        H = estimate_homography(model.marker_xy[int(mid)], c.reshape(4, 2))
        cids = model.square_corner_ids(int(mid))
        X = model.corners[cids][:, :2]          # (N, 2)
        p = apply_homography(H, X)              # (N, 2)
        for cid, pt in zip(cids, p):
            acc.setdefault(cid, []).append(pt)
    return {cid: np.mean(v, axis=0) for cid, v in acc.items()}