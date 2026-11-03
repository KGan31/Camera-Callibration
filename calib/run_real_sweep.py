"""Real-photo experiments: image-count sweep (3 to 30) and the fronto-parallel study.
Run from the repo root:  python -m calib.run_real_sweep      (writes results/real_sweep/)

  * sweep: A (standard board) and B (custom board), 10 random subsets per count, ours and OpenCV on
    the same corners and the same subsets.
  * fronto-parallel: Set C (board held nearly parallel to the sensor) against Set A, same board and
    camera. For each view we report the board tilt (angle between the board normal and the optical axis)
    from our own calibration, then the spread of the focal length over subsets next to the reprojection error.
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calib.board import BoardModel
from calib.custom_board import CustomBoardModel
from calib.real_calibrate import detect_views
from synthetic_calib.joint_refine import calibrate_full
from synthetic_calib.sweep import run_sweep, summarise, save_csv, disagreements, PARAMS

OUT = "results/real_sweep"
KEY = lambda s: int(os.path.basename(s).split("_")[1].split(".")[0])
paths = lambda t: sorted(glob.glob(f"data/set_{t}/*.jpeg"), key=KEY)


def tilts(views, size):
    """Board tilt per view in degrees, from our full calibration (0 = board exactly parallel to the sensor)."""
    r = calibrate_full(views, size)
    return r, np.array([np.degrees(np.arccos(min(1.0, abs(R[2, 2])))) for R, _ in r.poses])


def sweep_pair(name, views, size, counts, n_subsets=10):
    ours = run_sweep(views, size, counts=counts, n_subsets=n_subsets, workers=2)
    cv = run_sweep(views, size, counts=counts, n_subsets=n_subsets, method="opencv", workers=2)
    save_csv(ours, f"{OUT}/sweep_{name}_ours.csv"); save_csv(cv, f"{OUT}/sweep_{name}_opencv.csv")
    return ours, cv


def mask_full_pool(tab, n_views):
    """With every view used there is only one possible subset, so its 'spread' is 0 by construction: blank it."""
    out = []
    for r in tab:
        r = dict(r)
        if r["count"] >= n_views:
            for k in list(r):
                if k.endswith("_std"):
                    r[k] = np.nan
        out.append(r)
    return out


def plot_sweeps(tab):
    fig, ax = plt.subplots(2, 4, figsize=(17, 7))
    for j, p in enumerate(["fx", "cx", "cy", "k1"]):
        for name, t, c in tab:
            x = np.array([r["count"] for r in t]); m = np.array([r[p + "_mean"] for r in t]); s = np.array([r[p + "_std"] for r in t])
            ax[0, j].plot(x, m, color=c, label=name); ax[0, j].fill_between(x, m - s, m + s, color=c, alpha=.2)
            ax[1, j].plot(x, s, color=c, label=name)
        ax[0, j].set_title(f"{p}: mean ± 1 std over subsets"); ax[1, j].set_title(f"{p}: spread (std)")
        ax[1, j].set_yscale("log"); ax[1, j].set_xlabel("images used")
    ax[0, 0].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_real_sweep.png", dpi=130); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    std, cus = BoardModel(), CustomBoardModel()
    va, _, sa = detect_views(std, paths("a"))
    vb, _, sb = detect_views(cus, paths("b"))
    vc, skc, sc = detect_views(std, paths("c"))
    print("views A/B/C", len(va), len(vb), len(vc), "skipped C", [(os.path.basename(p), r) for p, r in skc])

    # ---------------- image-count sweep
    oa, ca = sweep_pair("A", va, sa, range(3, 32))
    ob, cb = sweep_pair("B", vb, sb, range(3, 26))
    ta, tb = mask_full_pool(summarise(oa), len(va)), mask_full_pool(summarise(ob), len(vb))
    plot_sweeps([("A standard", ta, "tab:blue"), ("B custom", tb, "tab:orange")])
    for name, o, c in (("A", oa, ca), ("B", ob, cb)):
        bad = disagreements(o, c)
        print(f"{name}: {len(bad)} of {len(o)} subsets disagree with OpenCV beyond thresholds")
        with open(f"{OUT}/disagreements_{name}.txt", "w") as f:
            f.write("\n".join(f"count {c} subset {s}: {w}" for c, s, w in bad) + "\n")

    # ---------------- fronto-parallel: C vs A
    ra, tilt_a = tilts(va, sa)
    rows = []
    try:
        rc, tilt_c = tilts(vc, sc)
    except Exception as e:                                            # keep the failure, as the plan says
        rc, tilt_c = None, np.array([])
        print("C full calibration failed:", e)
    with open(f"{OUT}/tilt_per_view.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "image", "tilt_deg", "mean_error_px"])
        for v, t, e in zip(va, tilt_a, ra.view_errors): w.writerow(["A", v["name"], f"{t:.1f}", f"{e:.3f}"])
        if rc is not None:
            for v, t, e in zip(vc, tilt_c, rc.view_errors): w.writerow(["C", v["name"], f"{t:.1f}", f"{e:.3f}"])
    counts = [c for c in (3, 4, 5, 6, 8, 10, 12, 15) if c <= len(vc)]
    oc, cc = sweep_pair("C", vc, sc, counts)
    a_matched = mask_full_pool(summarise([r for r in oa if r["count"] in counts]), len(va))
    c_tab = mask_full_pool(summarise(oc), len(vc))
    with open(f"{OUT}/fronto_parallel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["count", "A_ok", "A_fx_mean", "A_fx_std", "A_fx_std_pct", "A_mean_err", "C_ok", "C_fx_mean", "C_fx_std", "C_fx_std_pct", "C_mean_err"])
        for x, y in zip(a_matched, c_tab):
            w.writerow([x["count"], x["n_ok"], f"{x['fx_mean']:.1f}", f"{x['fx_std']:.1f}", f"{100*x['fx_std']/x['fx_mean']:.2f}", f"{x['mean_error_mean']:.3f}",
                        y["n_ok"], f"{y['fx_mean']:.1f}", f"{y['fx_std']:.1f}", f"{100*y['fx_std']/y['fx_mean']:.2f}" if np.isfinite(y["fx_std"]) else "nan (single subset)", f"{y['mean_error_mean']:.3f}"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(tilt_a, bins=np.arange(0, 75, 5), alpha=.6, label="A (varied)"); ax[0].hist(tilt_c, bins=np.arange(0, 75, 5), alpha=.6, label="C (fronto-parallel)")
    ax[0].set_xlabel("board tilt (deg)"); ax[0].legend(); ax[0].set_title("Tilt distribution")
    for tab, lab in ((a_matched, "A"), (c_tab, "C")):
        ax[1].plot([r["count"] for r in tab], [100 * r["fx_std"] / r["fx_mean"] for r in tab], "o-", label=lab)
        ax[2].plot([r["count"] for r in tab], [r["mean_error_mean"] for r in tab], "o-", label=lab)
    ax[1].set_title("fx spread over subsets (% of fx)"); ax[2].set_title("mean reprojection error (px)")
    for a in ax[1:]: a.set_xlabel("images used"); a.legend()
    ax[1].set_yscale("log"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_fronto_parallel.png", dpi=130)
    print("tilt A mean/median %.1f/%.1f | C mean/median %.1f/%.1f" % (tilt_a.mean(), np.median(tilt_a), tilt_c.mean() if len(tilt_c) else np.nan, np.median(tilt_c) if len(tilt_c) else np.nan))
    if rc is not None:
        print("A full: fx %.1f err %.3f | C full: fx %.1f fy %.1f cx %.1f cy %.1f k1 %.4f err %.3f" % (ra.K[0,0], ra.mean_error, rc.K[0,0], rc.K[1,1], rc.K[0,2], rc.K[1,2], rc.dist[0], rc.mean_error))
    for x, y in zip(a_matched, c_tab):
        print(f"count {x['count']:2d}  A fx {x['fx_mean']:.1f}±{x['fx_std']:.1f} ({x['n_ok']} ok) err {x['mean_error_mean']:.2f} | C fx {y['fx_mean']:.1f}±{y['fx_std']:.1f} ({y['n_ok']} ok) err {y['mean_error_mean']:.2f}")


if __name__ == "__main__":
    main()
