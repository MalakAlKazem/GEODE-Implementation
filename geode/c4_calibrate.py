"""
c4_calibrate.py -- two things the raw verification loop was missing.

    python3 c4_calibrate.py --results c4_results.csv

PART 1 -- threshold calibration
-------------------------------
The loop rejects when the PREDICTED d95 ratio exceeds 1.5, because 1.5 is
where the measured labels draw the line. That equivalence does not hold. A
regression model trained under MSE under-predicts extremes -- it is
rewarded for hedging toward the mean -- so predicted ratios are compressed
relative to measured ones. Applying the measured threshold to compressed
predictions systematically under-rejects, which is exactly the failure
mode observed: 49.8% of unsafe actions caught at a false-alarm rate of
only 18.9%. The gate was being too permissive, not uninformative.

This sweeps the predicted-ratio threshold and reports the operating curve.
Calibration is done on two topologies and reported on the third, so the
threshold is not chosen on the data it is evaluated on.

PART 2 -- dedicated predictor
-----------------------------
The progress report's section 7.5 preferred training a predictor directly
on the measured pairs over reusing the twin for extrapolation. The twin was
fitted to predict the state of a network it observes; asking it to score a
configuration that never existed is a different task, and the results
support the concern -- d95 ratio correlation 0.38, energy saving
correlation 0.28.

A dedicated model maps (baseline state, which cell is slept) directly to
(energy saving, d95 ratio), learning the counterfactual as its own
supervised problem. 640 labelled examples is small but the task is far
narrower than four-target state prediction.

Split by topology, as everywhere else, so the reported number is on a
network the model never saw.
"""

import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- part 1
def calibrate(df, mode="oracle", cal_seeds=(62, 63), test_seed=61):
    d = df[df.reassign == mode].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["pred_d95_ratio", "true_d95_ratio"])
    cal = d[d.seed.isin(cal_seeds)]
    tst = d[d.seed == test_seed]
    if cal.empty or tst.empty:
        print(f"  [{mode}] not enough data to calibrate")
        return None

    print(f"\n  calibration on seeds {list(cal_seeds)} ({len(cal)} actions), "
          f"reported on seed {test_seed} ({len(tst)} actions)")
    print(f"  predicted ratio: mean {d.pred_d95_ratio.mean():.2f} "
          f"median {d.pred_d95_ratio.median():.2f} "
          f"p95 {d.pred_d95_ratio.quantile(.95):.2f}")
    print(f"  measured  ratio: mean {d.true_d95_ratio.mean():.2f} "
          f"median {d.true_d95_ratio.median():.2f} "
          f"p95 {d.true_d95_ratio.quantile(.95):.2f}")
    comp = (d.true_d95_ratio.quantile(.95) - 1) / max(
        d.pred_d95_ratio.quantile(.95) - 1, 1e-6)
    print(f"  compression factor at p95: {comp:.1f}x  "
          f"(>1 means predictions are flattened, so the measured threshold "
          f"is too permissive when applied to them)")

    def stats(sub, thr):
        pred_unsafe = sub.pred_d95_ratio >= thr
        true_unsafe = ~sub.true_safe
        tp = int((pred_unsafe & true_unsafe).sum())
        fp = int((pred_unsafe & ~true_unsafe).sum())
        fn = int((~pred_unsafe & true_unsafe).sum())
        tn = int((~pred_unsafe & ~true_unsafe).sum())
        tpr = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        return tpr, fpr, (tpr + (1 - fpr)) / 2

    grid = np.round(np.arange(1.00, 2.05, 0.05), 2)
    print(f"\n  {'thresh':>7} {'cal TPR':>8} {'cal FPR':>8} {'cal bal':>8}")
    best, best_bal = None, -1
    for t in grid:
        tpr, fpr, bal = stats(cal, t)
        if np.isfinite(bal) and bal > best_bal:
            best, best_bal = t, bal
        if abs(t * 100 % 10) < 1e-6:      # print every 0.1 to stay readable
            print(f"  {t:>7.2f} {tpr:>8.3f} {fpr:>8.3f} {bal:>8.3f}")

    print(f"\n  chosen threshold {best:.2f} (best balanced accuracy on the "
          f"calibration topologies)")
    for label, thr in (("specified 1.50", 1.50), (f"calibrated {best:.2f}", best)):
        tpr, fpr, bal = stats(tst, thr)
        print(f"    held-out seed {test_seed}, {label:<16} "
              f"TPR {tpr:.3f}  FPR {fpr:.3f}  balanced {bal:.3f}")
    return best


# ---------------------------------------------------------------- part 2
def dedicated(df, mode="oracle", test_seed=61):
    """Predict (saving, d95 ratio) directly from action features.

    Deliberately a small model. With 640 examples a large one would fit the
    training topologies and say nothing about a new network. Features are
    what an operator knows before acting: which cell, how loaded it is, how
    loaded the network is, and how many UEs would be displaced.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\n  scikit-learn not installed -- skipping the dedicated "
              "predictor. pip install scikit-learn")
        return

    d = df[df.reassign == mode].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["true_d95_ratio", "true_saved_w"])

    # Features available before the action is taken. pred_* come from the
    # twin, so this is not twin-free -- it is the twin's output used as a
    # feature rather than as the decision, which is the point of comparison.
    feats = ["cell", "n_stranded", "predicted_saving_w", "pred_d95_base",
             "pred_d95_cf", "violation_rate_base"]
    feats = [f for f in feats if f in d.columns]

    tr = d[d.seed != test_seed]
    te = d[d.seed == test_seed]
    if len(tr) < 50 or len(te) < 20:
        print("\n  not enough data for the dedicated predictor")
        return

    print(f"\n  dedicated predictor: {len(tr)} train / {len(te)} test "
          f"(seed {test_seed} held out)")
    print(f"  features: {feats}")

    sc = StandardScaler().fit(tr[feats])
    Xtr, Xte = sc.transform(tr[feats]), sc.transform(te[feats])

    for target, name in (("true_d95_ratio", "d95 ratio"),
                         ("true_saved_w", "energy saving")):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=0)
        m.fit(Xtr, tr[target])
        p = m.predict(Xte)
        y = te[target].to_numpy()
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        corr = np.corrcoef(p, y)[0, 1]
        # what the twin alone achieved on the same held-out topology
        twin_col = ("pred_d95_ratio" if target == "true_d95_ratio"
                    else "predicted_saving_w")
        twin_corr = np.corrcoef(te[twin_col], y)[0, 1]
        print(f"    {name:<14} R2 {r2:>6.3f}  corr {corr:>6.3f}   "
              f"(twin alone: corr {twin_corr:>6.3f})")

    # safety decision from the dedicated model
    m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                  learning_rate=0.05, random_state=0)
    m.fit(Xtr, tr.true_d95_ratio)
    pr = m.predict(Xte)
    tu = ~te.true_safe.to_numpy()
    best = None
    for t in np.arange(1.0, 2.55, 0.05):
        pu = pr >= t
        tp, fp = int((pu & tu).sum()), int((pu & ~tu).sum())
        fn, tn = int((~pu & tu).sum()), int((~pu & ~tu).sum())
        tpr = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        bal = (tpr + (1 - fpr)) / 2
        if np.isfinite(bal) and (best is None or bal > best[3]):
            best = (t, tpr, fpr, bal)
    if best:
        print(f"    safety decision, best threshold {best[0]:.2f}: "
              f"TPR {best[1]:.3f}  FPR {best[2]:.3f}  balanced {best[3]:.3f}")
        print(f"    (threshold chosen on the held-out topology here, so this "
              f"is an upper bound, not a clean estimate)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="c4_results.csv")
    ap.add_argument("--test-seed", type=int, default=61)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    print(f"loaded {len(df)} evaluations from {args.results}")

    for mode in df.reassign.unique():
        print("\n" + "=" * 74)
        print(f"MODE: {mode}")
        print("=" * 74)
        cal_seeds = tuple(s for s in df.seed.unique() if s != args.test_seed)
        print("\nPART 1 -- threshold calibration")
        calibrate(df, mode, cal_seeds, args.test_seed)
        print("\nPART 2 -- dedicated predictor on the measured pairs")
        dedicated(df, mode, args.test_seed)


if __name__ == "__main__":
    main()