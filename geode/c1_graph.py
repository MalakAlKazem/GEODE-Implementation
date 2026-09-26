"""
c1_graph.py -- C1 Graph Construction for WirelessDT-6G.

Builds G = (V, E, X, F) per 100 ms snapshot with the four edge types from
the architecture spec, each independently toggleable so C1 can be ablated
without touching the encoder:

    E1  ('ue','serves','gnb') + reverse   -- serving assignment
    E2  ('gnb','interferes','gnb')        -- inter-cell coupling
    E3  ('gnb','handover','gnb')          -- observed UE transfers
    E4  ('gnb','backhaul','core')         -- aggregation point

Why each edge type is optional
------------------------------
E2 was rejected twice on scenario07 (static-distance and SINR-derived
variants both failed to add value). The stated suspicion was weak
inter-cell coupling at 1000 m inter-site distance. scenario06c halves
that to 500 m and raises handovers by 34%, so the condition under which
E2 was rejected no longer holds and the rejection needs re-testing
rather than assuming. Flags exist so that test is one argument, not a
code edit.

E3 is time-varying, unlike E2
-----------------------------
E2 is a fixed geometric relation: the cells do not move, so the edge set
is identical in every snapshot and only the encoder's attention over it
varies. E3 is different -- it records that a UE actually transferred
between two cells, which is an event, not a property of the layout. An
edge is added at the snapshot where the serving cell changes and held
for HANDOVER_WINDOW snapshots afterwards, because the disturbance a
handover causes (re-buffering, scheduler churn) outlasts the instant of
the switch. With the window at 0 the relation would be empty in ~99% of
snapshots and could not carry signal.

E4 and the core node
--------------------
The spec lists E4 as BS->Core. A single core node gives every gNB a
shared neighbour, which is a path for network-wide state to reach each
cell in one hop rather than through a chain of gNB-gNB steps. Its
features are the aggregate of the gNB features it collects, so it acts
as a learned global summary rather than a constant.

Honest note: E4 is the component with the weakest independent
justification. It is included because the spec lists it and because a
global aggregation node is cheap, not because there is prior evidence it
helps here. The ablation should be read with that in mind.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

HANDOVER_WINDOW = 30          # snapshots an E3 edge persists (30 = one 3 s phase)


# --------------------------------------------------------------------------
# E2 -- interference, static per run
# --------------------------------------------------------------------------
def build_interference_edges(gnb_snapshot_df, max_distance_m=None):
    """Fully-connected directed gNB-gNB graph, distance as edge attribute.

    No distance threshold by default. At 500 m spacing in a 1500x1000 m
    area the furthest pair is ~1118 m apart, which is well inside the
    range where one cell's transmissions still raise another's noise
    floor, so excluding any pair would need a justification we do not
    have. max_distance_m exists for testing that assumption, not because
    a particular value is recommended.
    """
    g = gnb_snapshot_df.sort_values("gnb_index")
    idx = g.gnb_index.to_numpy()
    xy = g[["pos_x", "pos_y"]].to_numpy()
    src, dst, attr = [], [], []
    for i in range(len(idx)):
        for j in range(len(idx)):
            if i == j:
                continue
            d = float(np.hypot(*(xy[i] - xy[j])))
            if max_distance_m is not None and d > max_distance_m:
                continue
            src.append(i); dst.append(j); attr.append([d])
    if not src:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(attr, dtype=torch.float32))


# --------------------------------------------------------------------------
# E3 -- handover, time-varying, precomputed once per run
# --------------------------------------------------------------------------
def precompute_handover_edges(serving_df, window=HANDOVER_WINDOW):
    """{snapshot_id: (edge_index[2,E], edge_attr[E,1])} for E3.

    Detects each serving-cell change per UE, then holds an edge between
    the two cells for `window` snapshots. edge_attr carries how many UEs
    transferred over that pair in the window, so a pair exchanging six
    UEs is distinguishable from one exchanging a single UE.
    """
    df = serving_df.sort_values(["ue_index", "snapshot_id"])
    df = df.assign(prev=df.groupby("ue_index").serving_gnb_index.shift())
    ho = df[(df.prev.notna()) & (df.prev != df.serving_gnb_index)]

    counts = {}   # snapshot -> {(a,b): n}
    all_snaps = np.sort(serving_df.snapshot_id.unique())
    snap_pos = {s: i for i, s in enumerate(all_snaps)}
    for _, r in ho.iterrows():
        a, b = int(r.prev), int(r.serving_gnb_index)
        start = snap_pos[r.snapshot_id]
        for k in range(start, min(start + window, len(all_snaps))):
            s = all_snaps[k]
            d = counts.setdefault(s, {})
            d[(a, b)] = d.get((a, b), 0) + 1
            d[(b, a)] = d.get((b, a), 0) + 1   # undirected, stored both ways

    out = {}
    for s in all_snaps:
        pairs = counts.get(s, {})
        if not pairs:
            out[s] = (torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1)))
        else:
            src = [p[0] for p in pairs]
            dst = [p[1] for p in pairs]
            att = [[float(v)] for v in pairs.values()]
            out[s] = (torch.tensor([src, dst], dtype=torch.long),
                      torch.tensor(att, dtype=torch.float32))
    return out


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------
def build_graph(gnb_rows, ue_rows, serving_rows,
                gnb_feats, ue_feats,
                energy_y=None, kpi_y=None,
                interference=None, handover=None,
                use_e1=True, use_e2=True, use_e3=True, use_e4=True):
    """One HeteroData snapshot.

    gnb_rows / ue_rows : per-snapshot slices, already sorted by index
    gnb_feats/ue_feats : normalised float32 tensors [N, F]
    interference       : (edge_index, edge_attr) from build_interference_edges
    handover           : (edge_index, edge_attr) for THIS snapshot
    """
    g = HeteroData()
    g["gnb"].x = gnb_feats
    g["ue"].x = ue_feats
    n_gnb = gnb_feats.shape[0]

    if energy_y is not None:
        g["gnb"].y = energy_y
    if kpi_y is not None:
        g["ue"].y = kpi_y

    # --- E1 serving -------------------------------------------------------
    if use_e1:
        ue_pos = {u: i for i, u in enumerate(ue_rows.ue_index.to_numpy())}
        gnb_pos = {b: i for i, b in enumerate(gnb_rows.gnb_index.to_numpy())}
        src, dst = [], []
        for u, b in zip(serving_rows.ue_index.to_numpy(),
                        serving_rows.serving_gnb_index.to_numpy()):
            if u in ue_pos and b in gnb_pos:
                src.append(ue_pos[u]); dst.append(gnb_pos[int(b)])
        ei = (torch.tensor([src, dst], dtype=torch.long) if src
              else torch.zeros((2, 0), dtype=torch.long))
        g["ue", "serves", "gnb"].edge_index = ei
        g["gnb", "served_by", "ue"].edge_index = ei.flip(0)

    # --- E2 interference --------------------------------------------------
    if use_e2 and interference is not None:
        ei, ea = interference
        g["gnb", "interferes", "gnb"].edge_index = ei
        g["gnb", "interferes", "gnb"].edge_attr = ea

    # --- E3 handover ------------------------------------------------------
    if use_e3:
        ei, ea = handover if handover is not None else (
            torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1)))
        g["gnb", "handover", "gnb"].edge_index = ei
        g["gnb", "handover", "gnb"].edge_attr = ea

    # --- E4 backhaul ------------------------------------------------------
    if use_e4:
        # Core features are the mean of gNB features: a learned global
        # summary rather than a constant placeholder.
        g["core"].x = gnb_feats.mean(dim=0, keepdim=True)
        ar = torch.arange(n_gnb, dtype=torch.long)
        zeros = torch.zeros(n_gnb, dtype=torch.long)
        g["gnb", "backhaul", "core"].edge_index = torch.stack([ar, zeros])
        g["core", "serves_bh", "gnb"].edge_index = torch.stack([zeros, ar])

    return g


def edge_types_present(use_e1=True, use_e2=True, use_e3=True, use_e4=True):
    """The relation list the encoder must be built for. Kept here so C1 and
    C2 cannot disagree about which relations exist."""
    rels = []
    if use_e1:
        rels += [("ue", "serves", "gnb"), ("gnb", "served_by", "ue")]
    if use_e2:
        rels += [("gnb", "interferes", "gnb")]
    if use_e3:
        rels += [("gnb", "handover", "gnb")]
    if use_e4:
        rels += [("gnb", "backhaul", "core"), ("core", "serves_bh", "gnb")]
    return rels
