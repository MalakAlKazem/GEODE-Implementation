"""
c2_hmpgnn.py -- C2 Heterogeneous Multi-Path GNN encoder.

Spec: T message-passing rounds, a type-specific message function per edge
type, DGAT attention on the interference relation, GRU state update,
LayerNorm, output H^T in R^(|V| x 64).

Difference from what was used before
------------------------------------
The models used up to now stack independent HeteroConv layers and pass
the output of one straight into the next -- each layer has its own
weights and no memory of the earlier state. The spec's encoder is a
recurrent one: a single set of message weights applied T times, with a
GRU deciding at each round how much of the incoming message to fold into
the node's running state. That is closer to RouteNet's design and it is
what makes T=10 rounds viable; ten stacked independent SAGEConv layers
would over-smooth badly.

Which relation gets attention, and why only that one
----------------------------------------------------
DGAT (GATv2 here) is applied to the interference relation alone. Every
gNB is connected to every other, so a plain mean would average six cells
of wildly differing load into one number and lose exactly the
distinction that matters -- a congested neighbour and an idle one would
contribute equally. Attention lets the encoder weight the neighbour that
is actually interfering. The serving relation does not need this: a UE
has exactly one serving cell, so there is nothing to attend over.

Over-smoothing
--------------
T=10 with a GRU is not the same risk as 10 stacked convolutions, because
the update gate can learn to keep most of the previous state. But it is
still a risk on a 6-node gNB graph, where 10 rounds is far more than
needed to reach every node. T is a constructor argument and should be
ablated; the spec's 10 is a starting point inherited from RouteNet's
much larger topologies, not a value validated here.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv


class HMPGNNEncoder(nn.Module):
    def __init__(self, in_dims: dict, hidden: int = 64, rounds: int = 10,
                 relations=None, attention_on=("gnb", "interferes", "gnb"),
                 heads: int = 2, dropout: float = 0.1, use_gru: bool = True):
        """
        in_dims      : {node_type: input_feature_dim}
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
        attn = tuple(attention_on) if attention_on is not None else None
        convs = {}
        for rel in relations:
            if attn is not None and tuple(rel) == attn:
                convs[tuple(rel)] = GATv2Conv(
                    (hidden, hidden), hidden // heads, heads=heads,
                    edge_dim=1, add_self_loops=False)
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = {nt: self.encoders[nt](x) for nt, x in x_dict.items()}

        # GATv2Conv is the only conv taking edge_attr; pass just its entry.
        ea = {}
        if (edge_attr_dict and self.attention_on is not None
                and self.attention_on in edge_attr_dict):
            ea[self.attention_on] = edge_attr_dict[self.attention_on]

        for _ in range(self.rounds):
            m = self.conv(h, edge_index_dict, edge_attr_dict=ea) if ea \
                else self.conv(h, edge_index_dict)
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