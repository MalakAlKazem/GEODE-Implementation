"""
wdt_data.py -- loading, normalisation and dataset construction for
scenario06c. Self-contained: reads the extracted CSVs directly rather
than going through multi_run_loader, so the feature set and the
non-circularity guarantee are visible in one place.

Feature selection and why active_state is absent
------------------------------------------------
gNB inputs deliberately exclude anything derived from the energy label.
Energy is computed by the extractor as
    P = P_idle + dP * (RB_used / RB_max) * N_TRX
so any feature carrying RB occupancy or sleep state would let the model
read the answer off its own input. num_connected_ues and offered_ul_bps
describe demand, not the resource allocation that demand produced, which
is the line the non-circularity principle draws. The sleep flag stays
diagnostic-only and is never an input.

Normalisation is fitted on training snapshots only, then applied to
validation and test. Fitting on everything would leak test statistics
into the model.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from c1_graph import (build_graph, build_interference_edges,
                      precompute_handover_edges)

# num_bands is retained for consistency with the architecture's stated
# gNB feature set, but audit_s06c.py confirms it is constant at 25 in all
# 25 runs -- zero variance, so it contributes nothing and the normaliser
# divides it by a floored std of 1.0. The gNB node therefore carries five
# informative features, not six, and three under --starve-gnb. Kept rather
# than dropped so the feature list matches the written architecture; the
# inertness is documented instead of hidden.
#
# tx_power_dbm is also constant WITHIN a run, but varies across p37/p43/p46,
# so it is informative in the pooled dataset. Not the same case.
GNB_COLS = ["pos_x", "pos_y", "tx_power_dbm", "num_connected_ues",
            "offered_ul_bps", "num_bands"]
UE_NUM_COLS = ["pos_x", "pos_y", "speed_mps", "offered_ul_bps", "sinr_db"]
QOS_CLASSES = ["eMBB", "URLLC", "mMTC"]          # one-hot, fixed order
KPI_COLS = ["delay_ms", "jitter_ms", "packet_loss"]

# Which KPI label resolution to load.
#   kpi_targets.csv     3 s traffic phases  -> 2,400 labels/run, stride 30
#   kpi_targets_1s.csv  1 s windows         -> ~6,700 labels/run, stride 10
# The 1 s files carry two extra columns: kpi_window_id, and n_packets, the
# number of packets the estimate rests on. n_packets == 0 marks a row with
# no measurement behind it, forward-filled by repair_kpi_grid_1s_s06c.py.
# Packet loss is NOT rebinned in either file -- its counters are per-app
# scalars, so it stays at 3 s resolution and repeats across the three 1 s
# windows of its phase.
KPI_FILE = os.environ.get("WDT_KPI_FILE", "kpi_targets.csv")


class Normalizer:
    """z-score, fitted on training rows only."""

    def __init__(self, cols):
        self.cols = cols
        self.mean_ = None
        self.std_ = None

    def fit(self, df):
        a = df[self.cols].to_numpy(dtype=np.float32)
        self.mean_ = a.mean(0)
        s = a.std(0)
        s[s < 1e-8] = 1.0
        self.std_ = s
        return self

    def transform(self, df):
        a = df[self.cols].to_numpy(dtype=np.float32)
        return (a - self.mean_) / self.std_


class LogLabelNormalizer:
    """log1p then z-score, for heavy-tailed targets.

    Delay at 1 s resolution spans roughly 9 ms to 1900 ms -- over two
    orders of magnitude, with a long right tail of congested windows. Under
    MSE on the raw value a handful of those windows dominate the gradient:
    the first 1 s run showed RMSE 385 ms against MAE 145 ms, which is the
    signature of a loss chasing outliers. Compressing with log1p makes the
    target roughly symmetric so that a 20 ms error at 20 ms and a 200 ms
    error at 200 ms count similarly, which is also what MAPE measures.

    log1p rather than log because jitter reaches 0.22 ms and log1p is well
    behaved near zero. Both targets are non-negative by construction.

    inverse() returns milliseconds, so metrics and the C4 threshold checks
    still work in physical units.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.log = True

    def fit(self, y):
        z = np.log1p(np.clip(np.asarray(y, dtype=np.float64), 0, None))
        self.mean_ = float(z.mean())
        s = float(z.std())
        self.std_ = s if s > 1e-8 else 1.0
        return self

    def transform(self, y):
        z = np.log1p(np.clip(np.asarray(y, dtype=np.float64), 0, None))
        return ((z - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, y):
        if torch.is_tensor(y):
            return torch.expm1(y * self.std_ + self.mean_)
        return np.expm1(np.asarray(y, dtype=np.float64) * self.std_ + self.mean_)


class LabelNormalizer:
    """Plain z-score. `log` is False so callers can tell the two apart."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.log = False

    def fit(self, y):
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = float(y.mean())
        s = float(y.std())
        self.std_ = s if s > 1e-8 else 1.0
        return self

    def transform(self, y):
        return (y - self.mean_) / self.std_

    def inverse(self, y):
        return y * self.std_ + self.mean_


def load_run(folder, run_tag):
    """The seven CSVs of one run, tagged so runs can be concatenated."""
    def rd(name):
        return pd.read_csv(os.path.join(folder, name))
    gnb = rd("gnb_inputs.csv")
    ue = rd("ue_inputs.csv")
    kpi = rd(KPI_FILE)
    eng = rd("energy_targets.csv")
    srv = rd("serving_edges.csv")
    for d in (gnb, ue, kpi, eng, srv):
        d["run_tag"] = run_tag
    # UE qos one-hot
    for c in QOS_CLASSES:
        ue[f"qos_{c}"] = (ue.qos_class == c).astype(np.float32)
    return {"gnb": gnb, "ue": ue, "kpi": kpi, "energy": eng, "serving": srv}


def load_runs(run_paths):
    """run_paths: {run_tag: folder}. Snapshot ids are offset per run so
    they stay unique after concatenation."""
    parts = {k: [] for k in ("gnb", "ue", "kpi", "energy", "serving")}
    handover = {}
    interference = None
    offset = 0
    for i, (tag, folder) in enumerate(sorted(run_paths.items())):
        r = load_run(folder, tag)
        off = i * 100000
        for k in parts:
            r[k]["snapshot_id"] = r[k]["snapshot_id"] + off
            parts[k].append(r[k])
        handover.update(precompute_handover_edges(r["serving"]))
        if interference is None:
            snap0 = r["gnb"][r["gnb"].snapshot_id == r["gnb"].snapshot_id.min()]
            interference = build_interference_edges(snap0)
    data = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    data["handover"] = handover
    data["interference"] = interference
    data["ue_cols"] = UE_NUM_COLS + [f"qos_{c}" for c in QOS_CLASSES]
    return data


LAG_GNB_COLS = ["lag_energy_w"]
LAG_UE_COLS = ["lag_delay_ms", "lag_jitter_ms"]


def add_lag_features(data):
    """Mutates data["gnb"]/data["ue"] in place, adding each node's own
    previous-snapshot label(s) as extra raw columns (LAG_GNB_COLS /
    LAG_UE_COLS) -- the training-time counterpart of
    baseline_classical.py's RF+lag diagnostic, which found this signal
    alone (no graph) explains most of RF's apparent edge over gnn on
    this dataset.

    06c splits targets across node types (energy on gnb, delay/jitter on
    ue), so the lag splits the same way: gnb gets its own previous
    estimated_gnb_power_w, ue gets its own previous [delay_ms,
    jitter_ms]. Lag is computed within (run_tag, node_index), sorted by
    snapshot_id -- a run's first snapshot has no valid previous value.

    Returns the set of snapshot_ids that are a run's first snapshot for
    ANY node (i.e. would have a NaN/fabricated lag) -- the caller must
    drop these from every id list (train/val/test) rather than train or
    evaluate on a fabricated lag of 0.
    """
    eng = data["energy"][["snapshot_id", "gnb_index", "run_tag",
                           "estimated_gnb_power_w"]].sort_values(
        ["run_tag", "gnb_index", "snapshot_id"])
    eng["lag_energy_w"] = eng.groupby(
        ["run_tag", "gnb_index"])["estimated_gnb_power_w"].shift(1)
    data["gnb"] = data["gnb"].merge(
        eng[["snapshot_id", "gnb_index", "run_tag", "lag_energy_w"]],
        on=["snapshot_id", "gnb_index", "run_tag"], how="left")

    kpi = data["kpi"][["snapshot_id", "ue_index", "run_tag",
                        "delay_ms", "jitter_ms"]].sort_values(
        ["run_tag", "ue_index", "snapshot_id"])
    kpi["lag_delay_ms"] = kpi.groupby(
        ["run_tag", "ue_index"])["delay_ms"].shift(1)
    kpi["lag_jitter_ms"] = kpi.groupby(
        ["run_tag", "ue_index"])["jitter_ms"].shift(1)
    data["ue"] = data["ue"].merge(
        kpi[["snapshot_id", "ue_index", "run_tag",
             "lag_delay_ms", "lag_jitter_ms"]],
        on=["snapshot_id", "ue_index", "run_tag"], how="left")

    run_starts = set(
        data["gnb"].groupby("run_tag").snapshot_id.min().tolist())
    drop_ids = set(
        data["gnb"].loc[data["gnb"].lag_energy_w.isna(),
                        "snapshot_id"]).union(
        data["ue"].loc[data["ue"].lag_delay_ms.isna(), "snapshot_id"])
    assert drop_ids == run_starts, \
        "lag NaNs should exactly match each run's first snapshot_id"
    return drop_ids


class SnapshotDataset(Dataset):
    """One HeteroData per snapshot, built once and cached.

    Per-snapshot pandas filtering inside the training loop was previously
    the dominant cost in this pipeline, so the slices are taken once here
    via groupby rather than re-filtered every epoch.
    """

    def __init__(self, data, snapshot_ids, gnb_norm, ue_norm,
                 energy_norm=None, kpi_norms=None, pl_bin_edges=None,
                 use_e1=True, use_e2=True, use_e3=True, use_e4=True):
        ids = set(snapshot_ids)
        gnb_g = {k: v for k, v in data["gnb"].groupby("snapshot_id") if k in ids}
        ue_g = {k: v for k, v in data["ue"].groupby("snapshot_id") if k in ids}
        kpi_g = {k: v for k, v in data["kpi"].groupby("snapshot_id") if k in ids}
        eng_g = {k: v for k, v in data["energy"].groupby("snapshot_id") if k in ids}
        srv_g = {k: v for k, v in data["serving"].groupby("snapshot_id") if k in ids}

        self.graphs = []
        for s in sorted(ids):
            if s not in gnb_g or s not in ue_g:
                continue
            gr = gnb_g[s].sort_values("gnb_index")
            ur = ue_g[s].sort_values("ue_index")
            er = eng_g[s].sort_values("gnb_index")
            kr = kpi_g[s].sort_values("ue_index")

            gx = torch.tensor(gnb_norm.transform(gr), dtype=torch.float32)
            ux = torch.tensor(ue_norm.transform(ur), dtype=torch.float32)

            ey = torch.tensor(
                energy_norm.transform(er.estimated_gnb_power_w.to_numpy()),
                dtype=torch.float32).unsqueeze(-1)

            ky = np.stack([
                kpi_norms["delay"].transform(kr.delay_ms.to_numpy()),
                kpi_norms["jitter"].transform(kr.jitter_ms.to_numpy()),
            ], axis=1)
            ky = torch.tensor(ky, dtype=torch.float32)

            # n_packets == 0 means the label was forward-filled rather
            # than measured. Kept on the graph so a training run can mask
            # those rows out of the KPI loss instead of fitting to them.
            npk = (kr.n_packets.to_numpy() if "n_packets" in kr.columns
                   else np.full(len(kr), -1))

            pl_raw = kr.packet_loss.to_numpy()
            pl_bin = torch.tensor((pl_raw > 0).astype(np.float32))
            pl_cls = torch.tensor(
                np.digitize(pl_raw, pl_bin_edges[1:-1]) if pl_bin_edges is not None
                else np.zeros_like(pl_raw), dtype=torch.long)

            g = build_graph(gr, ur, srv_g.get(s, ur.iloc[0:0]),
                            gx, ux, energy_y=ey, kpi_y=ky,
                            interference=data["interference"],
                            handover=data["handover"].get(s),
                            use_e1=use_e1, use_e2=use_e2,
                            use_e3=use_e3, use_e4=use_e4)
            g["ue"].pl_binary = pl_bin
            g["ue"].pl_class = pl_cls
            g["ue"].n_packets = torch.tensor(npk, dtype=torch.long)
            self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]


def compute_bin_edges(pl_nonzero, num_bins=5):
    """Quantile bins over the non-zero packet-loss values only."""
    if len(pl_nonzero) < num_bins:
        return np.linspace(0, 1, num_bins + 1)
    qs = np.linspace(0, 1, num_bins + 1)
    e = np.quantile(pl_nonzero, qs)
    e[0], e[-1] = 0.0, 1.0
    return np.unique(e)