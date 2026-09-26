"""
wdt_data.py -- loading, normalisation and edge-toggling for the 5G3E
real testbed, mirroring src/v6/wdt_data.py's contract so the same
training harness pattern (gnn vs blind, ablation sweep) applies here.

Does NOT re-parse raw JSON/CSV. src/v3/preprocess_v3.py already produced
leak-free per-snapshot HeteroData graphs at data/5g3e_v3/processed/ --
raw (un-normalised) node features, gNB-level labels
[energy_W, delay_ms, jitter_ms, packet_loss], and a fixed edge topology
per day (E1 serving is round-robin-assigned, not measured; E2/E3 are
static site-membership partitions -- see the C1 docstring in v3 for why).
This module only adds: train-only normalisation, edge-type toggling for
ablation, and packaging into a torch Dataset the v6-style training loop
can consume.

Split
-----
split.pt gives {'train': [...], 'val': [...], 'test': [...]}: Day 1
(minus the last 15%) trains, the last 15% of Day 1 validates, Day 2
tests. This is the ONLY honest test split -- there is no second
independent day to rotate against, unlike scenario06c's 3-seed rotation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

PROCESSED_DIR = os.environ.get("GEODE_5G3E", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "5g3e_processed"))

# All six static relations present in every stored snapshot.
ALL_RELATIONS = [
    ("ue", "connects", "gnb"),
    ("gnb", "rev_connects", "ue"),
    ("gnb", "interferes", "gnb"),
    ("gnb", "handover", "gnb"),
    ("gnb", "backhaul", "srv"),
    ("srv", "rev_backhaul", "gnb"),
]

# (use_e1..use_e4) -> which relations survive. E1 serving and its reverse
# travel together; E4 backhaul and its reverse travel together.
def relations_for(use_e1=True, use_e2=True, use_e3=True, use_e4=True):
    rels = []
    if use_e1:
        rels += [("ue", "connects", "gnb"), ("gnb", "rev_connects", "ue")]
    if use_e2:
        rels += [("gnb", "interferes", "gnb")]
    if use_e3:
        rels += [("gnb", "handover", "gnb")]
    if use_e4:
        rels += [("gnb", "backhaul", "srv"), ("srv", "rev_backhaul", "gnb")]
    return rels


Y_COLS = ["energy_W", "delay_ms", "jitter_ms", "packet_loss"]

# Day boundaries in the snapshot-id sequence (see preprocess_v3.py's
# printed offsets: Day1 0..7121, Day2 7122..13789, Day3 13790..20763).
# A day-start id has no valid "previous snapshot" -- sid-1 would belong
# to a different day recorded hours apart, not one lag-step earlier.
DAY_START_IDS = {0, 7122, 13790}


class Normalizer:
    """z-score over a [N, F] feature block, fitted on train snapshots only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, mean, std):
        # norm_stats.pt already has train-only mean/std computed by
        # preprocess_v3.py -- reuse it rather than recomputing, so both
        # pipelines agree on the exact same numbers.
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.std_ = np.asarray(std, dtype=np.float32)
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_


class LabelNormalizer:
    """Plain z-score for one label column, inverse() returns physical units."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, mean, std):
        self.mean_ = float(mean)
        self.std_ = float(std) if std > 1e-8 else 1.0
        return self

    def transform(self, y):
        return (y - self.mean_) / self.std_

    def inverse(self, y):
        return y * self.std_ + self.mean_


def load_norm_stats():
    """norm_stats.pt: train-only mean/std for ue/gnb/srv features and the
    4 gNB labels. See preprocess_v3.py:compute_norm_stats."""
    stats = torch.load(os.path.join(PROCESSED_DIR, "norm_stats.pt"),
                        weights_only=False)
    ue_norm = Normalizer().fit(stats["ue_mean"], stats["ue_std"])
    gnb_norm = Normalizer().fit(stats["gnb_mean"], stats["gnb_std"])
    srv_norm = Normalizer().fit(stats["srv_mean"], stats["srv_std"])
    y_norms = {}
    for i, col in enumerate(Y_COLS):
        y_norms[col] = LabelNormalizer().fit(stats["y_mean"][i], stats["y_std"][i])
    return {"ue": ue_norm, "gnb": gnb_norm, "srv": srv_norm, "y": y_norms,
            "ue_cols": stats["ue_cols"], "gnb_cols": stats["gnb_cols"],
            "srv_cols": stats["srv_cols"]}


def load_split():
    return torch.load(os.path.join(PROCESSED_DIR, "split.pt"), weights_only=False)


class SnapshotDataset(Dataset):
    """Wraps stored HeteroData snapshots: applies train-fit normalisation,
    drops edge types per the ablation flags, and normalises the 4 gNB
    labels in place while keeping the raw values for physical-unit metrics.

    Snapshots are read from disk once at construction time (13,792 total,
    small individually) rather than re-loaded every __getitem__ call.

    use_lag: appends each gNB's own [energy, delay, jitter] from the
    PREVIOUS snapshot to its input features (normalised the same way as
    the current label). This mirrors baseline_classical.py's RF+lag
    exactly, for a controlled test of the same hypothesis on the GNN --
    see that module's docstring for why this turned out to matter far
    more than any single architecture ablation. Day-start snapshots have
    no valid previous snapshot and are dropped, same as RF+lag.
    """

    def __init__(self, snapshot_ids, norms,
                 use_e1=True, use_e2=True, use_e3=True, use_e4=True,
                 use_lag=False):
        self.norms = norms
        self.keep_rels = set(relations_for(use_e1, use_e2, use_e3, use_e4))
        self.use_lag = use_lag
        ids = sorted(snapshot_ids)
        if use_lag:
            ids = [s for s in ids if s not in DAY_START_IDS]
        self.snapshot_ids = ids
        self._prev_y_cache = {}
        self.graphs = [self._load_one(sid) for sid in self.snapshot_ids]
        self._prev_y_cache = None  # no longer needed once loading is done

    def _prev_raw_y(self, sid):
        """Raw [energy,delay,jitter,pl] for `sid`, used as a lag source.
        Cached because a snapshot is `current` for one graph and `prev`
        for the next -- caching means each file is read once, not twice,
        in the common case where snapshot_ids are contiguous."""
        if sid not in self._prev_y_cache:
            d = torch.load(os.path.join(PROCESSED_DIR, f"snapshot_{sid:05d}.pt"),
                           weights_only=False)
            self._prev_y_cache[sid] = d["gnb"].y.numpy()
        return self._prev_y_cache[sid]

    def _load_one(self, sid):
        d = torch.load(os.path.join(PROCESSED_DIR, f"snapshot_{sid:05d}.pt"),
                        weights_only=False)

        if self.use_lag:
            # Make this snapshot's own raw y available for sid+1's lag
            # lookup, so the contiguous-neighbour case never re-reads
            # the same file from disk.
            self._prev_y_cache[sid] = d["gnb"].y.numpy()

        d["ue"].x = torch.tensor(self.norms["ue"].transform(d["ue"].x.numpy()),
                                  dtype=torch.float32)
        gnb_x = self.norms["gnb"].transform(d["gnb"].x.numpy())

        if self.use_lag:
            prev_y_raw = self._prev_raw_y(sid - 1)[:, :3]  # energy, delay, jitter
            prev_y_norm = np.stack([
                self.norms["y"][col].transform(prev_y_raw[:, i])
                for i, col in enumerate(Y_COLS[:3])
            ], axis=1)
            gnb_x = np.concatenate([gnb_x, prev_y_norm], axis=1)
            # Also kept as its own tensor (not just baked into gnb.x) so
            # predict_delta can compute delta = current - lag directly,
            # without reverse-engineering which input columns are the lag.
            d["gnb"].lag_y_norm = torch.tensor(prev_y_norm, dtype=torch.float32)
            self._prev_y_cache.pop(sid - 1, None)  # sid-1 never needed as prev again

        d["gnb"].x = torch.tensor(gnb_x, dtype=torch.float32)
        d["srv"].x = torch.tensor(self.norms["srv"].transform(d["srv"].x.numpy()),
                                   dtype=torch.float32)

        y_raw = d["gnb"].y.numpy()  # [N_gnb, 4] energy, delay, jitter, pl
        y_norm = np.stack([
            self.norms["y"][col].transform(y_raw[:, i])
            for i, col in enumerate(Y_COLS)
        ], axis=1)
        d["gnb"].y = torch.tensor(y_norm, dtype=torch.float32)
        # kept for physical-unit metrics without re-inverting per column
        d["gnb"].y_raw = torch.tensor(y_raw, dtype=torch.float32)

        for rel in ALL_RELATIONS:
            if rel not in self.keep_rels and rel in d.edge_types:
                del d[rel]

        return d

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]


def in_dims_for(norms, use_lag=False):
    return {"ue": len(norms["ue_cols"]),
            "gnb": len(norms["gnb_cols"]) + (3 if use_lag else 0),
            "srv": len(norms["srv_cols"])}
