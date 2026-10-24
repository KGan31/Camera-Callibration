"""The "how many photos do we need?" experiment.

The project plan asks for a sweep from 3 to 30 images, with 10 random subsets for every count,
showing each camera number and how much it wobbles as the count grows.

This file does the experiment. It works on ANY pool of views (made-up ones now, real photos later):
each view is a dict with  obj_pts  (board points, metres)  and  img_pts  (the corners seen, pixels).

    records = run_sweep(views, image_size)          # our own calibration on every subset
    cv      = run_sweep(views, image_size, method="opencv")   # OpenCV on the very same subsets
    table   = summarise(records)                    # mean and spread for every count

Failures are KEPT, not hidden: a subset that cannot be calibrated gets a row with ok=False and
the reason, as the project plan asks ("failure cases are reported rather than filtered out").
"""
import csv
import numpy as np
import cv2

from synthetic_calib.joint_refine import calibrate_full

PARAMS = ["fx", "fy", "cx", "cy", "k1", "k2"]
FIELDS = ["count", "subset", "ok", "reason", "converged"] + PARAMS + ["mean_error", "rms_error"]


def pick_subset(n_pool, count, subset_index, seed=0):
    """Which `count` views (out of n_pool) subset number `subset_index` uses.
    Depends only on (seed, count, subset_index), so our run and OpenCV's run get the SAME subsets,
    and running part of a sweep gives the same answer as running all of it."""
    rng = np.random.default_rng([seed, count, subset_index])
    return np.sort(rng.choice(n_pool, count, replace=False))


def _blank(count, subset, ok, reason=""):
    row = {f: np.nan for f in FIELDS}
    row.update(count=count, subset=subset, ok=ok, reason=reason, converged=False)
    return row


def _opencv_calibrate(views, image_size):
    obj = [np.column_stack([v["obj_pts"], np.zeros(len(v["obj_pts"]))]).astype(np.float32) for v in views]
    img = [np.asarray(v["img_pts"], np.float32).reshape(-1, 1, 2) for v in views]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, tuple(image_size), None, None, flags=flags)
    errs = []
    for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        errs.append(np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1))
    errs = np.concatenate(errs)
    return K, dist.ravel(), float(errs.mean()), float(rms)


def calibrate_subset(views, image_size, method="ours", **kwargs):
    """One calibration. Returns (K, (k1,k2), mean_error, rms_error, converged)."""
    if method == "ours":
        r = calibrate_full(views, image_size, **kwargs)
        return r.K, r.dist, r.mean_error, r.rms_error, r.converged
    if method == "opencv":
        K, dist, mean_e, rms = _opencv_calibrate(views, image_size)
        return K, dist[:2], mean_e, rms, True
    raise ValueError("method must be 'ours' or 'opencv'")


def run_one(views, image_size, count, subset, seed=0, method="ours", **kwargs):
    """One row of the sweep: calibrate subset number `subset` of size `count`."""
    idx = pick_subset(len(views), count, subset, seed)
    row = _blank(count, subset, True)
    try:
        K, dist, mean_e, rms, conv = calibrate_subset([views[i] for i in idx], image_size, method, **kwargs)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as e:    # cannot be calibrated: keep the reason
        return _blank(count, subset, False, f"{type(e).__name__}: {e}"[:120])
    row.update(converged=bool(conv), fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
               k1=dist[0], k2=dist[1], mean_error=mean_e, rms_error=rms)
    return row


def _job(args):
    return run_one(*args[:-1], **args[-1])


def run_sweep(views, image_size, counts=range(3, 31), n_subsets=10, seed=0, method="ours",
              workers=1, **kwargs):
    """Calibrate n_subsets random subsets for every count. Returns a list of row dicts."""
    counts = [c for c in counts if c <= len(views)]
    jobs = [(views, image_size, c, s, seed, method, kwargs) for c in counts for s in range(n_subsets)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_job, jobs, chunksize=4))
    return [_job(j) for j in jobs]


def summarise(records):
    """One row per count: how many subsets worked, and mean / spread (standard deviation) of
    every parameter over the ones that did."""
    out = []
    for c in sorted({r["count"] for r in records}):
        rows = [r for r in records if r["count"] == c]
        good = [r for r in rows if r["ok"]]
        s = {"count": c, "n_subsets": len(rows), "n_ok": len(good),
             "n_not_converged": sum(not r["converged"] for r in good)}
        for p in PARAMS + ["mean_error", "rms_error"]:
            vals = np.array([r[p] for r in good], float)
            s[p + "_mean"] = vals.mean() if len(vals) else np.nan
            s[p + "_std"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        out.append(s)
    return out


def disagreements(ours, theirs, focal_pct=1.0, centre_px=2.0, error_px=0.05):
    """Subsets where our answer and OpenCV's differ by more than the project's thresholds
    (focal length 1 %, principal point 2 px, mean error 0.05 px). Both lists must come from the
    same sweep settings, so row i of one is the same subset as row i of the other."""
    bad = []
    for a, b in zip(ours, theirs):
        assert (a["count"], a["subset"]) == (b["count"], b["subset"])
        if not (a["ok"] and b["ok"]):
            if a["ok"] != b["ok"]:
                bad.append((a["count"], a["subset"], "only one of the two could calibrate"))
            continue
        why = []
        for p in ("fx", "fy"):
            if abs(a[p] - b[p]) / b[p] * 100 > focal_pct:
                why.append(f"{p} differs by {abs(a[p] - b[p]) / b[p] * 100:.2f} %")
        for p in ("cx", "cy"):
            if abs(a[p] - b[p]) > centre_px:
                why.append(f"{p} differs by {abs(a[p] - b[p]):.1f} px")
        if abs(a["mean_error"] - b["mean_error"]) > error_px:
            why.append(f"mean error differs by {abs(a['mean_error'] - b['mean_error']):.3f} px")
        if why:
            bad.append((a["count"], a["subset"], "; ".join(why)))
    return bad


def save_csv(records, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {"count": int(r["count"]), "subset": int(r["subset"]), "ok": r["ok"] == "True",
                   "reason": r["reason"], "converged": r["converged"] == "True"}
            for k in PARAMS + ["mean_error", "rms_error"]:
                row[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
            rows.append(row)
    return rows
