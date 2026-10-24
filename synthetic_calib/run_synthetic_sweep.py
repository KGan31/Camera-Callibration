"""Run the 3-to-30 photo sweep on made-up photos and save the numbers and pictures.

    python -m synthetic_calib.run_synthetic_sweep          (takes about 5 to 10 minutes)

Results go to  results/sweep_synthetic/ . For REAL photos later, use the same functions from
sweep.py and plot_sweep.py on a pool of real views instead of the made-up pool made here.
"""
import os
import time
from synthetic_calib.synthetic import make_views, true_camera_matrix, IMAGE_W, IMAGE_H
from synthetic_calib import sweep, plot_sweep

SIZE = (IMAGE_W, IMAGE_H)
OUT = os.path.join("results", "sweep_synthetic")
K1, K2, NOISE = -0.18, 0.04, 0.3


def main(workers=2):
    os.makedirs(OUT, exist_ok=True)
    K, pool = make_views(n_views=40, noise_px=NOISE, seed=2024, k1=K1, k2=K2)
    t = time.time()
    ours = sweep.run_sweep(pool, SIZE, workers=workers)
    print(f"ours: {time.time() - t:.0f} s", flush=True)
    cv = sweep.run_sweep(pool, SIZE, method="opencv", workers=workers)
    # honest reference: brand-new photo sets (not subsets of one pool), 10 per count
    fresh = []
    for c in [3, 5, 8, 10, 15, 20, 25, 30]:
        for s in range(10):
            _, v = make_views(n_views=c, noise_px=NOISE, seed=9000 + 100 * c + s, k1=K1, k2=K2)
            fresh.append(sweep.run_one(v, SIZE, c, s))
    sweep.save_csv(ours, os.path.join(OUT, "sweep_ours.csv"))
    sweep.save_csv(cv, os.path.join(OUT, "sweep_opencv.csv"))
    sweep.save_csv(fresh, os.path.join(OUT, "sweep_fresh.csv"))
    truth = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=K1, k2=K2)
    plot_sweep.plot_parameters(ours, cv, os.path.join(OUT, "fig_params.png"), truth=truth)
    plot_sweep.plot_spread(ours, cv, os.path.join(OUT, "fig_spread.png"), fresh=fresh)
    plot_sweep.plot_error(ours, cv, os.path.join(OUT, "fig_error.png"))
    bad = sweep.disagreements(ours, cv)
    print(f"failed subsets (ours): {sum(not r['ok'] for r in ours)}")
    print(f"subsets where we and OpenCV disagree beyond the project thresholds: {len(bad)}")
    for b in bad:
        print("  ", b)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
