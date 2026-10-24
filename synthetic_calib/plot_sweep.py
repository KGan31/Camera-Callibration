"""Pictures for the "how many photos do we need?" experiment (see sweep.py).

    plot_parameters(ours, opencv, "p.png", truth={"fx": 1050, ...})   # each number vs photo count
    plot_spread(ours, opencv, "s.png", fresh=...)                      # how much each number wobbles
    plot_error(ours, opencv, "e.png")                                  # the error the library reports

`ours`, `opencv`, `fresh` are lists of rows from sweep.run_sweep (or sweep.load_csv).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from synthetic_calib.sweep import summarise

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
LABELS = {"fx": "focal length fx (px)", "fy": "focal length fy (px)", "cx": "image centre cx (px)",
          "cy": "image centre cy (px)", "k1": "lens bending k1", "k2": "lens bending k2"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)


def _col(table, key):
    return np.array([t[key] for t in table], float)


def plot_parameters(ours, opencv, path, truth=None, params=("fx", "fy", "cx", "cy", "k1", "k2")):
    """One small chart per camera number: mean over the subsets (line), the spread (band, one
    standard deviation) and every single subset (dots), for us and for OpenCV."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, p in zip(axes.ravel(), params):
        _style(ax)
        for rows, colour, name, dash in ((opencv, ORANGE, "OpenCV", "--"), (ours, BLUE, "ours", "-")):
            if rows is None:
                continue
            ok = [r for r in rows if r["ok"]]
            ax.scatter([r["count"] for r in ok], [r[p] for r in ok], s=6, color=colour, alpha=0.25, lw=0)
            t = summarise(rows)
            c, m, s = _col(t, "count"), _col(t, p + "_mean"), _col(t, p + "_std")
            ax.fill_between(c, m - s, m + s, color=colour, alpha=0.15, lw=0)
            ax.plot(c, m, dash, color=colour, lw=1.8, label=name)
        if truth and p in truth:
            ax.axhline(truth[p], color=INK, lw=1, ls=":", label="true value")
        ax.set_title(LABELS[p], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("number of photos", fontsize=8, color=MUTED)
        if p in ("fx", "fy", "cx", "cy") and truth and p in truth:     # zoom to where the action is
            allv = [r[p] for rows in (ours, opencv) if rows for r in rows
                    if r["ok"] and r["count"] >= 5]
            lo, hi = np.percentile(allv + [truth[p]], [1, 99])
            pad = 0.2 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Each camera number against the number of photos used (10 random subsets per count)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_spread(ours, opencv, path, fresh=None, params=("fx", "cx", "cy", "k1")):
    """Spread (standard deviation over the 10 subsets) of each number against photo count, log scale.
    `fresh` (optional): the same thing from brand-new independent photo sets, the honest reference."""
    fig, axes = plt.subplots(1, len(params), figsize=(11, 3.4), constrained_layout=True)
    for ax, p in zip(axes, params):
        _style(ax)
        for rows, colour, name, dash in ((opencv, ORANGE, "OpenCV", "--"), (ours, BLUE, "ours", "-"),
                                         (fresh, GREY, "new photo sets", ":")):
            if rows is None:
                continue
            t = summarise(rows)
            ax.plot(_col(t, "count"), _col(t, p + "_std"), dash, color=colour, lw=1.8,
                    marker="o" if rows is fresh else None, ms=4, label=name)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"spread of {p}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("number of photos", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_error(ours, opencv, path):
    """The mean reprojection error the library reports, against photo count."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    _style(ax)
    for rows, colour, name, dash in ((opencv, ORANGE, "OpenCV", "--"), (ours, BLUE, "ours", "-")):
        t = summarise(rows)
        c, m, s = _col(t, "count"), _col(t, "mean_error_mean"), _col(t, "mean_error_std")
        ax.fill_between(c, m - s, m + s, color=colour, alpha=0.15, lw=0)
        ax.plot(c, m, dash, color=colour, lw=1.8, label=name)
    ax.set_xlabel("number of photos", fontsize=8, color=MUTED)
    ax.set_ylabel("mean reprojection error (px)", fontsize=8, color=MUTED)
    ax.set_title("The error the library reports", fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
