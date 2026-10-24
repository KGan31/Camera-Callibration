import numpy as np, cv2

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
        H, _ = cv2.findHomography(model.marker_xy[int(mid)], c.reshape(4, 2), 0)
        cids = model.square_corner_ids(int(mid))
        X = model.corners[cids][:, :2].reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(X, H).reshape(-1, 2)
        for cid, pt in zip(cids, p):
            acc.setdefault(cid, []).append(pt)
    return {cid: np.mean(v, axis=0) for cid, v in acc.items()}