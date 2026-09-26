"""
c2_hmpgnn.py -- C2 Heterogeneous Multi-Path GNN encoder, 5G3E variant.

Ported from src/v6/c2_hmpgnn.py. T message-passing rounds, a type-specific
message function per edge type, DGAT attention on the interference
relation, GRU state update, LayerNorm, output H^T in R^(|V| x 64).

One real difference from the v6 (scenario06c) encoder
-------------------------------------------------------
v6's GATv2Conv on the interference edge takes edge_dim=1, fed by the
distance between the two gNBs. 5G3E has no gNB positions in its feature
set (preprocess_v3.py extracts [nof_ue, site_0, site_1, site_2,
mean_rsrp] -- no pos_x/pos_y), and the stored (gnb, interferes, gnb)
edges carry no edge_attr. Attention here therefore runs on GATv2's own
node-feature attention alone, with no edge_dim. This is weaker than
v6's distance-informed version -- it can still learn to weight a
congested site-mate over an idle one from their hidden states, but it
cannot use physical proximity as a prior the way v6 can.

Which relation gets attention, and why only that one
----------------------------------------------------
DGAT (GATv2 here) is applied to the interference relation alone. All
gNBs at the same site interfere with each other, so a plain mean would
average their wildly differing loads into one number and lose exactly
the distinction that matters -- a congested site-mate and an idle one
would contribute equally. The serving relation does not need this: a UE
has exactly one serving gNB, so there is nothing to attend over.

Over-smoothing
--------------
T=10 with a GRU is not the same risk as 10 stacked convolutions, because
the update gate can learn to keep most of the previous state. But on a
13-node gNB graph, 10 rounds is far more than needed to reach every
node -- more so here than on 06c's 6-node case. T is a constructor
argument and should be ablated, not assumed.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv


class HMPGNNEncoder(nn.Module):
    def __init__(self, in_dims: dict, hidden: int = 64, rounds: int = 10,
                 relations=None, attention_on=("gnb", "interferes", "gnb"),
                 heads: int = 2, dropout: float = 0.1, use_gru: bool = True):
        """
        in_dims      : {node_type: input_feature_dim}, e.g.
                       {"ue": 7, "gnb": 5, "srv": 1} for 5G3E
        relations    : list of (src, rel, dst) triples the graph will contain
        hidden       : spec says 64
        rounds       : T
        attention_on : relation to give GATv2 attention, or None for no
                       attention anywhere (every relation uses SAGEConv).
                       Setting this to None while keeping the interference
                       edges separates two things that --no-e2 changes
                       together: the edges themselves and the attention
                       over them.
        use_gru      : False replaces the recurrent update with a plain
                       overwrite, h <- norm(message). That isolates what
                       the GRU contributes, which is the most distinctive
                       part of this encoder -- without it the architecture
                       is a weight-tied stack of convolutions, and T=10
                       should over-smooth badly.
        """
        super().__init__()
        self.rounds = rounds
        self.hidden = hidden
        self.use_gru = use_gru
        self.node_types = list(in_dims.keys())

        # Type-specific input projection (spec: MLP init per node type)
        self.encoders = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                              nn.Linear(hidden, hidden))
            for nt, d in in_dims.items()
        })

        # One message function per edge type, reused across all T rounds.
        # No edge_dim: 5G3E's interference edges carry no distance
        # attribute (see module docstring).
        attn = tuple(attention_on) if attention_on is not None else None
        convs = {}
        for rel in relations:
            if attn is not None and tuple(rel) == attn:
                convs[tuple(rel)] = GATv2Conv(
                    (hidden, hidden), hidden // heads, heads=heads,
                    add_self_loops=False)
            else:
                convs[tuple(rel)] = SAGEConv((hidden, hidden), hidden)
        self.conv = HeteroConv(convs, aggr="sum")
        self.attention_on = attn

        # GRU state update per node type, shared across rounds.
        self.grus = (nn.ModuleDict({nt: nn.GRUCell(hidden, hidden)
                                    for nt in in_dims})
                     if use_gru else None)
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden)
                                    for nt in in_dims})
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        h = {nt: self.encoders[nt](x) for nt, x in x_dict.items()}

        for _ in range(self.rounds):
            m = self.conv(h, edge_index_dict)
            new_h = {}
            for nt in h:
                if nt in m and m[nt] is not None:
                    if self.use_gru:
                        # GRU decides how much of the message to absorb.
                        upd = self.grus[nt](m[nt], h[nt])
                    else:
                        # Plain overwrite: the previous state is discarded,
                        # which is what a stack of convolutions does.
                        upd = m[nt]
                    upd = self.norms[nt](upd)
                    new_h[nt] = self.dropout(upd)
                else:
                    # A node type with no incoming edges this round keeps
                    # its state rather than being zeroed.
                    new_h[nt] = h[nt]
            h = new_h
        return h
