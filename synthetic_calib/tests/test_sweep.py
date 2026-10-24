"""Checks for synthetic_calib/sweep.py  (the "how many photos do we need?" experiment).
Run from the repo folder with:   python -m synthetic_calib.tests.test_sweep
(needs scipy; takes under a minute)
"""
import os
import tempfile
import numpy as np
from synthetic_calib.synthetic import make_views, IMAGE_W, IMAGE_H
from synthetic_calib import sweep

SIZE = (IMAGE_W, IMAGE_H)


def fake_row(count, subset, ok=True, **vals):
    r = sweep._blank(count, subset, ok, "" if ok else "boom")
    if ok:
        r.update(converged=True, fx=1000.0, fy=1000.0, cx=640.0, cy=480.0, k1=-0.1, k2=0.0,
                 mean_error=0.3, rms_error=0.35)
    r.update(vals)
    return r


def test_subsets_are_repeatable_and_valid():
    a = sweep.pick_subset(40, 10, 3, seed=7)
    assert np.array_equal(a, sweep.pick_subset(40, 10, 3, seed=7))      # same question, same answer
    assert len(a) == 10 and len(set(a)) == 10 and np.all(np.diff(a) > 0)
    assert a.min() >= 0 and a.max() < 40
    sweep.pick_subset(40, 5, 0, seed=7)                                 # asking something else first...
    assert np.array_equal(a, sweep.pick_subset(40, 10, 3, seed=7))      # ...changes nothing


def test_different_subsets_really_differ():
    subs = {tuple(sweep.pick_subset(40, 10, s, seed=1)) for s in range(10)}
    assert len(subs) == 10
    assert tuple(sweep.pick_subset(40, 10, 0, seed=1)) != tuple(sweep.pick_subset(40, 10, 0, seed=2))


def test_asking_for_everything_gives_everything():
    assert np.array_equal(sweep.pick_subset(12, 12, 0), np.arange(12))


def test_sweep_shape_and_counts_too_big_are_skipped():
    _, views = make_views(n_views=8, noise_px=0.3, seed=1, k1=-0.1, k2=0.02)
    rows = sweep.run_sweep(views, SIZE, counts=range(3, 12), n_subsets=2)
    assert len(rows) == 2 * 6                                           # counts 3..8 only
    assert sorted({r["count"] for r in rows}) == [3, 4, 5, 6, 7, 8]


def test_failures_are_kept_with_a_reason():
    _, views = make_views(n_views=6, noise_px=0.3, seed=2)
    rows = sweep.run_sweep(views, SIZE, counts=[2, 4], n_subsets=2)
    bad = [r for r in rows if r["count"] == 2]
    assert len(bad) == 2 and all(not r["ok"] and "at least 3" in r["reason"] for r in bad)
    assert all(np.isnan(r["fx"]) for r in bad)
    assert all(r["ok"] for r in rows if r["count"] == 4)


def test_both_methods_use_the_same_subsets_and_agree():
    _, views = make_views(n_views=20, noise_px=0.3, seed=3, k1=-0.15, k2=0.03)
    ours = sweep.run_sweep(views, SIZE, counts=[8, 15], n_subsets=3)
    cv = sweep.run_sweep(views, SIZE, counts=[8, 15], n_subsets=3, method="opencv")
    assert [(r["count"], r["subset"]) for r in ours] == [(r["count"], r["subset"]) for r in cv]
    assert sweep.disagreements(ours, cv) == []                          # inside all three thresholds
    assert all(abs(a["fx"] - b["fx"]) / b["fx"] < 0.002 for a, b in zip(ours, cv))


def test_more_workers_give_the_same_numbers():
    _, views = make_views(n_views=10, noise_px=0.3, seed=4, k1=-0.1, k2=0.02)
    a = sweep.run_sweep(views, SIZE, counts=[4, 6], n_subsets=2, workers=1)
    b = sweep.run_sweep(views, SIZE, counts=[4, 6], n_subsets=2, workers=2)
    assert all(np.isclose(x["fx"], y["fx"], rtol=0, atol=1e-9) for x, y in zip(a, b))


def test_spread_shrinks_as_photos_are_added():
    _, views = make_views(n_views=30, noise_px=0.3, seed=5, k1=-0.18, k2=0.04)
    rows = sweep.run_sweep(views, SIZE, counts=[3, 25], n_subsets=8)
    table = {t["count"]: t for t in sweep.summarise(rows)}
    assert table[3]["fx_std"] > 2 * table[25]["fx_std"]
    assert table[25]["n_ok"] == 8


def test_summarise_numbers():
    rows = [fake_row(5, 0, fx=1000.0), fake_row(5, 1, fx=1002.0), fake_row(5, 2, fx=1004.0),
            fake_row(5, 3, ok=False), fake_row(7, 0, fx=1001.0)]
    rows[2]["converged"] = False
    t = {r["count"]: r for r in sweep.summarise(rows)}
    assert t[5]["n_subsets"] == 4 and t[5]["n_ok"] == 3 and t[5]["n_not_converged"] == 1
    assert np.isclose(t[5]["fx_mean"], 1002.0) and np.isclose(t[5]["fx_std"], 2.0)   # failure left out of the mean
    assert np.isnan(t[7]["fx_std"])                                                    # one value has no spread


def test_disagreements_flags_only_real_gaps():
    ours = [fake_row(5, 0), fake_row(5, 1), fake_row(5, 2), fake_row(5, 3), fake_row(5, 4, ok=False)]
    cv = [fake_row(5, 0), fake_row(5, 1, fx=1020.0), fake_row(5, 2, cx=643.0),
          fake_row(5, 3, mean_error=0.40), fake_row(5, 4)]
    flagged = {s: why for _, s, why in sweep.disagreements(ours, cv)}
    assert sorted(flagged) == [1, 2, 3, 4]
    assert "fx" in flagged[1] and "cx" in flagged[2] and "mean error" in flagged[3] and "only one" in flagged[4]
    assert sweep.disagreements(ours[:1], cv[:1]) == []


def test_csv_round_trip_keeps_failures():
    rows = [fake_row(3, 0, fx=999.5), fake_row(3, 1, ok=False)]
    path = os.path.join(tempfile.mkdtemp(), "s.csv")
    sweep.save_csv(rows, path)
    back = sweep.load_csv(path)
    assert back[0]["ok"] and back[0]["fx"] == 999.5 and back[0]["count"] == 3
    assert (not back[1]["ok"]) and back[1]["reason"] == "boom" and np.isnan(back[1]["fx"])


if __name__ == "__main__":
    import time
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    t0 = time.time()
    for f in tests:
        f()
        print("PASS", f.__name__)
    print(f"\nAll {len(tests)} checks passed in {time.time() - t0:.0f} s.")
