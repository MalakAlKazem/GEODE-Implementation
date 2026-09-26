"""
baseline_classical.py -- naive and Random Forest baselines for
scenario06c, mirroring src/v7_5g3e/baseline_classical.py so both
datasets carry the same comparison ladder in the manuscript.

Same fairness rule as the 5G3E version: RF sees exactly what
WirelessDT6GBlind sees for that target, nothing more. In this
architecture (see wdt_model.py's PredictionHeads split) energy reads
only h["gnb"], so its RF baseline gets only GNB_COLS; delay/jitter/PL
read only h["ue"], so their RF baseline gets only UE_NUM_COLS + the QoS
one-hot. Giving RF cross-node information (e.g. UE features for the
energy prediction) would answer a different question than "does the
graph beat a flat model given identical inputs."

Same rotation as the gnn/blind numbers already reported
------------------------------------------------------------
train_wdt_s06c.py rotates each of the 3 simulator seeds through as the
held-out test set and reports mean +/- std over the 3 rotations. This
file does the same, so the RF numbers are directly comparable to the
existing gnn/blind table rather than resting on a single lucky split.

packet_loss is handled as classification, not regression, matching how
gnn/blind are scored: RandomForestClassifier + AUROC on "loss occurred"
(pl > 0), not R2 on a 99.6%-zero continuous value.

RF+lag: the 06c mirror of 5G3E's temporal-signal diagnostic
------------------------------------------------------------------
5G3E's baseline_classical.py found that giving RF a single lag feature
(each gNB's own previous-snapshot labels) closed most of the graph's
advantage on delay/jitter -- suggesting a lot of the "graph advantage"
seen elsewhere may really be missing TEMPORAL signal, not a spatial one.
06c shows the opposite anomaly (RF already beats gnn on energy, without
any lag feature at all) -- this is the same cheap test applied here, to
see whether that too has a temporal explanation, or whether it survives
even once the graph model would gain the same information.

06c splits targets across two node types (5G3E has all 4 on one gNB
node), so the lag feature has to split the same way: energy's RF+lag
sees the gNB's own previous estimated_gnb_power_w; delay/jitter's
RF+lag sees the UE's own previous [delay_ms, jitter_ms]. A node's first
snapshot in a run has no valid previous snapshot and is dropped rather
than given a fabricated lag (mirrors 5G3E's DAY_START_IDS handling,
just per-run instead of per-day since 06c has no day structure).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from wdt_data import load_runs, GNB_COLS, UE_NUM_COLS, QOS_CLASSES
from train_wdt_s06c import run_paths_for, SEEDS, metrics, auroc

UE_COLS = UE_NUM_COLS + [f"qos_{c}" for c in QOS_CLASSES]


def _rows_for_seeds(seeds, baselines_only=False):
    """Raw (unnormalised) per-gNB and per-UE rows for a set of seeds,
    pooled across all their runs -- trees don't need feature scaling."""
    data = load_runs(run_paths_for(seeds, baselines_only=baselines_only))
    gnb, ue, kpi, eng = data["gnb"], data["ue"], data["kpi"], data["energy"]

    e = eng.merge(gnb[["snapshot_id", "gnb_index"] + GNB_COLS],
                  on=["snapshot_id", "gnb_index"], how="inner")
    X_e = e[GNB_COLS].to_numpy(dtype=np.float64)
    y_e = e["estimated_gnb_power_w"].to_numpy(dtype=np.float64)

    k = kpi.merge(ue[["snapshot_id", "ue_index"] + UE_COLS],
                  on=["snapshot_id", "ue_index"], how="inner")
    X_u = k[UE_COLS].to_numpy(dtype=np.float64)
    y_delay = k["delay_ms"].to_numpy(dtype=np.float64)
    y_jitter = k["jitter_ms"].to_numpy(dtype=np.float64)
    y_pl_bin = (k["packet_loss"].to_numpy(dtype=np.float64) > 0).astype(int)

    return X_e, y_e, X_u, y_delay, y_jitter, y_pl_bin


def _lagged(df, group_cols, sort_col, value_cols):
    """Add lag_<col> = previous-snapshot value of value_cols, within each
    group_cols group, ordered by sort_col. First row of each group gets
    NaN (no valid previous snapshot) -- caller drops those rows."""
    d = df.sort_values(group_cols + [sort_col]).copy()
    for c in value_cols:
        d[f"lag_{c}"] = d.groupby(group_cols)[c].shift(1)
    return d


def _rows_for_seeds_with_lag(seeds, baselines_only=False):
    """Same rows as _rows_for_seeds, plus each node's own previous-
    snapshot label(s) appended to its RF input -- see module docstring.
    Returns (X_e_lag, y_e, X_u_lag, y_delay, y_jitter); note y_e here is
    only over rows that had a valid lag (first snapshot of each run,
    per gNB, is dropped), same for y_delay/y_jitter over UE rows."""
    data = load_runs(run_paths_for(seeds, baselines_only=baselines_only))
    gnb, ue, kpi, eng = data["gnb"], data["ue"], data["kpi"], data["energy"]

    e = eng.merge(gnb[["snapshot_id", "gnb_index"] + GNB_COLS],
                  on=["snapshot_id", "gnb_index"], how="inner")
    e = _lagged(e, ["run_tag", "gnb_index"], "snapshot_id",
                ["estimated_gnb_power_w"])
    e = e.dropna(subset=["lag_estimated_gnb_power_w"])
    X_e = np.concatenate([
        e[GNB_COLS].to_numpy(dtype=np.float64),
        e[["lag_estimated_gnb_power_w"]].to_numpy(dtype=np.float64),
    ], axis=1)
    y_e = e["estimated_gnb_power_w"].to_numpy(dtype=np.float64)

    k = kpi.merge(ue[["snapshot_id", "ue_index"] + UE_COLS],
                  on=["snapshot_id", "ue_index"], how="inner")
    k = _lagged(k, ["run_tag", "ue_index"], "snapshot_id",
                ["delay_ms", "jitter_ms"])
    k = k.dropna(subset=["lag_delay_ms", "lag_jitter_ms"])
    X_u = np.concatenate([
        k[UE_COLS].to_numpy(dtype=np.float64),
        k[["lag_delay_ms", "lag_jitter_ms"]].to_numpy(dtype=np.float64),
    ], axis=1)
    y_delay = k["delay_ms"].to_numpy(dtype=np.float64)
    y_jitter = k["jitter_ms"].to_numpy(dtype=np.float64)

    return X_e, y_e, X_u, y_delay, y_jitter


def run_lag(baselines_only=False):
    """RF+lag variant of run() -- same 3-seed rotation, energy/delay/
    jitter only (packet_loss has no lag-classification analogue here,
    matching 5G3E's TARGETS choice of energy/delay/jitter only)."""
    per_rot = {"energy": [], "delay": [], "jitter": []}

    for test_seed in SEEDS:
        train_seeds = [s for s in SEEDS if s != test_seed]
        print(f"\n=== test seed {test_seed} (train {train_seeds}) ===")
        Xe_tr, ye_tr, Xu_tr, yd_tr, yj_tr = _rows_for_seeds_with_lag(
            train_seeds, baselines_only)
        Xe_te, ye_te, Xu_te, yd_te, yj_te = _rows_for_seeds_with_lag(
            [test_seed], baselines_only)
        print(f"  energy rows: train {Xe_tr.shape} test {Xe_te.shape}  "
              f"| UE rows: train {Xu_tr.shape} test {Xu_te.shape}")

        rf_e = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     n_jobs=-1, random_state=0)
        rf_e.fit(Xe_tr, ye_tr)
        rfe_m = metrics(ye_te, rf_e.predict(Xe_te))

        rf_d = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     n_jobs=-1, random_state=0)
        rf_d.fit(Xu_tr, yd_tr)
        rfd_m = metrics(yd_te, rf_d.predict(Xu_te))

        rf_j = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     n_jobs=-1, random_state=0)
        rf_j.fit(Xu_tr, yj_tr)
        rfj_m = metrics(yj_te, rf_j.predict(Xu_te))

        per_rot["energy"].append(rfe_m)
        per_rot["delay"].append(rfd_m)
        per_rot["jitter"].append(rfj_m)

        print(f"  energy  RF+lag R2 {rfe_m['R2']:.3f} MAPE {rfe_m['MAPE']:.2f}%")
        print(f"  delay   RF+lag R2 {rfd_m['R2']:.3f} MAPE {rfd_m['MAPE']:.2f}%")
        print(f"  jitter  RF+lag R2 {rfj_m['R2']:.3f} MAPE {rfj_m['MAPE']:.2f}%")

    print("\n" + "=" * 78)
    print("scenario06c RF+lag -- mean +/- std over 3 rotations")
    print("=" * 78)
    for tgt in ("energy", "delay", "jitter"):
        r2 = [m["R2"] for m in per_rot[tgt]]
        mape = [m["MAPE"] for m in per_rot[tgt]]
        print(f"{tgt:<8} RF+lag R2 {np.mean(r2):.3f}+/-{np.std(r2):.3f}   "
              f"MAPE {np.mean(mape):.2f}%+/-{np.std(mape):.2f}%")

    return per_rot


def run(baselines_only=False):
    """baselines_only=False (the default) includes the sleep runs,
    matching train_wdt_s06c.py's own default -- the reported gnn/blind
    numbers were trained the same way unless --baselines-only was
    explicitly passed. Set True to exclude them for a like-for-like
    comparison against a --baselines-only gnn/blind run instead."""
    per_rot = {"energy": [], "delay": [], "jitter": [], "pl": []}

    for test_seed in SEEDS:
        train_seeds = [s for s in SEEDS if s != test_seed]
        print(f"\n=== test seed {test_seed} (train {train_seeds}) ===")
        Xe_tr, ye_tr, Xu_tr, yd_tr, yj_tr, ypl_tr = _rows_for_seeds(
            train_seeds, baselines_only)
        Xe_te, ye_te, Xu_te, yd_te, yj_te, ypl_te = _rows_for_seeds(
            [test_seed], baselines_only)

        naive_e = metrics(ye_te, np.full_like(ye_te, ye_tr.mean()))
        rf_e = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     n_jobs=-1, random_state=0)
        rf_e.fit(Xe_tr, ye_tr)
        rfe_m = metrics(ye_te, rf_e.predict(Xe_te))

        naive_d = metrics(yd_te, np.full_like(yd_te, yd_tr.mean()))
        rf_d = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     n_jobs=-1, random_state=0)
        rf_d.fit(Xu_tr, yd_tr)
        rfd_m = metrics(yd_te, rf_d.predict(Xu_te))

        naive_j = metrics(yj_te, np.full_like(yj_te, yj_tr.mean()))
        rf_j = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     n_jobs=-1, random_state=0)
        rf_j.fit(Xu_tr, yj_tr)
        rfj_m = metrics(yj_te, rf_j.predict(Xu_te))

        rf_pl = RandomForestClassifier(n_estimators=200, max_depth=12,
                                       n_jobs=-1, random_state=0,
                                       class_weight="balanced")
        rf_pl.fit(Xu_tr, ypl_tr)
        pl_score = rf_pl.predict_proba(Xu_te)[:, 1]
        pl_auroc = auroc(ypl_te, pl_score)

        per_rot["energy"].append((naive_e, rfe_m))
        per_rot["delay"].append((naive_d, rfd_m))
        per_rot["jitter"].append((naive_j, rfj_m))
        per_rot["pl"].append(pl_auroc)

        print(f"  energy  naive R2 {naive_e['R2']:.3f} MAPE {naive_e['MAPE']:.2f}%  "
              f"| RF R2 {rfe_m['R2']:.3f} MAPE {rfe_m['MAPE']:.2f}%")
        print(f"  delay   naive R2 {naive_d['R2']:.3f} MAPE {naive_d['MAPE']:.2f}%  "
              f"| RF R2 {rfd_m['R2']:.3f} MAPE {rfd_m['MAPE']:.2f}%")
        print(f"  jitter  naive R2 {naive_j['R2']:.3f} MAPE {naive_j['MAPE']:.2f}%  "
              f"| RF R2 {rfj_m['R2']:.3f} MAPE {rfj_m['MAPE']:.2f}%")
        print(f"  packet_loss  RF AUROC {pl_auroc:.3f}")

    print("\n" + "=" * 78)
    print("scenario06c classical baselines -- mean +/- std over 3 rotations")
    print("=" * 78)
    for tgt in ("energy", "delay", "jitter"):
        naive_r2 = [n["R2"] for n, r in per_rot[tgt]]
        rf_r2 = [r["R2"] for n, r in per_rot[tgt]]
        rf_mape = [r["MAPE"] for n, r in per_rot[tgt]]
        print(f"{tgt:<8} naive R2 {np.mean(naive_r2):.3f}+/-{np.std(naive_r2):.3f}   "
              f"RF R2 {np.mean(rf_r2):.3f}+/-{np.std(rf_r2):.3f}   "
              f"RF MAPE {np.mean(rf_mape):.2f}%+/-{np.std(rf_mape):.2f}%")
    pl = per_rot["pl"]
    print(f"packet_loss  RF AUROC {np.mean(pl):.3f}+/-{np.std(pl):.3f}")

    return per_rot


if __name__ == "__main__":
    import sys
    if "--lag" in sys.argv:
        run_lag()
    else:
        run()
