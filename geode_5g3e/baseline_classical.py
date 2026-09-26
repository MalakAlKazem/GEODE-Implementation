"""
baseline_classical.py -- naive and Random Forest baselines for 5G3E,
to sit below WirelessDT6GBlind in the manuscript's comparison ladder.

Why these two, and why now
----------------------------
WirelessDT6GBlind answers "does message passing help, given the same
learned representation and training procedure?" -- that's an ablation
control, not a methodology baseline. A reviewer will ask how the whole
approach compares to (a) a trivial floor and (b) a standard non-graph
ML method, neither of which we had until this file. WirelessNet's own
evaluation (Perdomo et al. 2025) uses exactly this pattern -- FCDNN and
Random Forest baselines alongside their graph model.

Fairness: RF sees exactly what blind sees, nothing more
-----------------------------------------------------------
WirelessDT6GBlind's per-gNB encoder only ever reads that gNB's OWN 5
features (nof_ue, site_0, site_1, site_2, mean_rsrp) -- it never
aggregates UE or server data, because there is no message passing. RF
is given the identical input for a fair "no graph, no learned
representation" comparison point: same features in, same features out,
just a different (non-neural, non-representation-learning) model class
sitting between them. Giving RF anything more (e.g. aggregated UE
stats) would answer a different question than the one this ladder is
for.

Naive baseline: predict the training-set mean for every test row --
the floor that establishes the problem isn't trivial by construction.

packet_loss is excluded: it is a measured constant (0.0) across the
whole 5G3E dataset (see c3_heads.py's docstring) -- there is nothing
for a baseline to be measured against.

Lag-feature diagnostic: is this a temporal problem, not a spatial one?
--------------------------------------------------------------------------
On the Day3 generalisation test, RF beats both `blind` and `gnn` on
jitter and roughly ties `gnn` on delay -- using a model with no graph
and no learned representation at all. Every model tested so far
(naive, RF, blind, gnn) predicts from a single snapshot in isolation;
none of them has ever seen a gNB's own recent history. Delay/jitter are
downstream effects of congestion dynamics that unfold over several
snapshots, so this looks like a missing TEMPORAL signal, not a spatial
or architectural one. RF_lag adds each gNB's own previous-snapshot
labels (energy_W, delay_ms, jitter_ms) as three extra input features --
the cheapest possible test of that hypothesis, before committing to a
real temporal architecture (a GRU across snapshots, not just across
message-passing rounds within one).

Snapshot IDs are contiguous and day-ordered (Day1: 0..7121, Day2:
7122..13789, Day3: 13790..20763 -- see preprocess_v3.py's offsets), and
every day has the same 13 gNBs in the same gnb_index order, so
sid-1's gnb row i is truly "the same gNB, one snapshot earlier" EXCEPT
at a day boundary, where sid-1 belongs to a different day recorded
hours apart. Those boundary snapshots (DAY_START_IDS) are dropped
rather than given a fabricated lag.
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from wdt_data import load_norm_stats, load_split, PROCESSED_DIR
from train_wdt_5g3e import metrics

TARGETS = ["energy_W", "delay_ms", "jitter_ms"]
DAY_START_IDS = {0, 7122, 13790}


def load_raw_gnb_rows(snapshot_ids):
    """Each gNB's own 5 raw features (unnormalised -- trees don't need
    scaling) and its 4 raw labels, one row per gNB per snapshot."""
    xs, ys = [], []
    for sid in snapshot_ids:
        d = torch.load(f"{PROCESSED_DIR}/snapshot_{sid:05d}.pt", weights_only=False)
        xs.append(d["gnb"].x.numpy())
        ys.append(d["gnb"].y.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_raw_gnb_rows_with_lag(snapshot_ids):
    """Same as load_raw_gnb_rows, plus each gNB's own previous-snapshot
    [energy_W, delay_ms, jitter_ms] appended to the feature vector.
    Snapshots at a day boundary (no valid previous snapshot) are
    dropped rather than given a fabricated lag."""
    ids = sorted(s for s in snapshot_ids if s not in DAY_START_IDS)
    cache = {}

    def get(sid):
        if sid not in cache:
            cache[sid] = torch.load(f"{PROCESSED_DIR}/snapshot_{sid:05d}.pt",
                                    weights_only=False)
        return cache[sid]

    xs, ys = [], []
    for sid in ids:
        d, d_prev = get(sid), get(sid - 1)
        lag = d_prev["gnb"].y.numpy()[:, :3]  # energy, delay, jitter
        xs.append(np.concatenate([d["gnb"].x.numpy(), lag], axis=1))
        ys.append(d["gnb"].y.numpy())
        cache.pop(sid - 1, None)  # each sid only needed as "prev" once
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _fit_rf(X_tr, y_tr, X_te, y_te):
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               n_jobs=-1, random_state=0)
    rf.fit(X_tr, y_tr)
    return metrics(y_te, rf.predict(X_te))


def run():
    split = load_split()
    print(f"Loading train ({len(split['train'])} snapshots) and "
          f"test ({len(split['test'])} snapshots) gNB rows...")
    X_tr, y_tr = load_raw_gnb_rows(split["train"])
    X_te, y_te = load_raw_gnb_rows(split["test"])
    print(f"  train rows: {X_tr.shape}  test rows: {X_te.shape}")

    print("Loading lag-feature variant (own previous-snapshot labels)...")
    Xl_tr, yl_tr = load_raw_gnb_rows_with_lag(split["train"])
    Xl_te, yl_te = load_raw_gnb_rows_with_lag(split["test"])
    print(f"  train rows: {Xl_tr.shape}  test rows: {Xl_te.shape}")

    results = {}
    for i, tgt in enumerate(TARGETS):
        yt_tr, yt_te = y_tr[:, i], y_te[:, i]

        naive_pred = np.full_like(yt_te, yt_tr.mean())
        naive_m = metrics(yt_te, naive_pred)
        rf_m = _fit_rf(X_tr, yt_tr, X_te, yt_te)
        rf_lag_m = _fit_rf(Xl_tr, yl_tr[:, i], Xl_te, yl_te[:, i])

        results[tgt] = {"naive": naive_m, "rf": rf_m, "rf_lag": rf_lag_m}
        print(f"\n{tgt}")
        print(f"  naive   R2 {naive_m['R2']:.3f}  MAE {naive_m['MAE']:.3f}  "
              f"MAPE {naive_m['MAPE']:.2f}%")
        print(f"  RF      R2 {rf_m['R2']:.3f}  MAE {rf_m['MAE']:.3f}  "
              f"MAPE {rf_m['MAPE']:.2f}%")
        print(f"  RF+lag  R2 {rf_lag_m['R2']:.3f}  MAE {rf_lag_m['MAE']:.3f}  "
              f"MAPE {rf_lag_m['MAPE']:.2f}%")

    return results


if __name__ == "__main__":
    run()
