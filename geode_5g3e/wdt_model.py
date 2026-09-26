"""
wdt_model.py -- WirelessDT-6G model for 5G3E: C2 encoder + C3 heads.

Also defines the graph-blind control -- the piece missing from the
original src/v3 pipeline (train_v3.py never trained one). The control
must differ from the full model in exactly one respect -- the presence
of message passing -- or the comparison measures architecture size
rather than the value of the graph. Same encoders, same heads, same
hidden width; the only thing removed is the C2 conv/GRU loop.

Why the blind control should fail harder here than on 06c
------------------------------------------------------------
energy_W = 130 + 470 * cpu_util, and cpu_util lives only on the srv
node. WirelessDT6GBlind never runs a single edge -- a gNB's forward
pass sees only its own 5 features (nof_ue, site one-hot, mean_rsrp),
none of which is cpu_util or a private proxy for it (one server powers
4-5 gNBs, so even if site correlated with load it's a shared, noisy
signal, not an owned one). On 06c, by contrast, the blind gNB still
sees its own offered_ul_bps/num_connected_ues, which are a real if
imperfect proxy for its own energy. So the predicted outcome here is a
LARGER gnn-vs-blind energy gap than 06c's, not just a repeat of it --
that's a measurement to make, not an assumption to state.
"""

import torch
import torch.nn as nn

from c2_hmpgnn import HMPGNNEncoder
from c3_heads import PredictionHeads


class WirelessDT6G(nn.Module):
    def __init__(self, in_dims: dict, relations, hidden: int = 64,
                 rounds: int = 10, num_pl_bins: int = 5, use_moe: bool = True,
                 num_experts: int = 4, top_k: int = 2, dropout: float = 0.1,
                 attention_on=("gnb", "interferes", "gnb"),
                 use_gru: bool = True, predict_delta: bool = False):
        super().__init__()
        self.encoder = HMPGNNEncoder(
            in_dims, hidden=hidden, rounds=rounds, relations=relations,
            attention_on=attention_on, dropout=dropout, use_gru=use_gru)
        self.heads = PredictionHeads(
            hidden, num_pl_bins=num_pl_bins, use_moe=use_moe,
            num_experts=num_experts, k=top_k, dropout=dropout,
            predict_delta=predict_delta)

    def forward(self, x_dict, edge_index_dict):
        h = self.encoder(x_dict, edge_index_dict)
        return self.heads(h)


class WirelessDT6GBlind(nn.Module):
    """Identical minus message passing: each node predicts from its own
    features alone. This is the number the graph has to beat -- and on
    5G3E, energy specifically should be close to unbeatable for this
    model, since cpu_util never reaches a gNB without an edge to cross."""

    def __init__(self, in_dims: dict, relations=None, hidden: int = 64,
                 rounds: int = 10, num_pl_bins: int = 5, use_moe: bool = True,
                 num_experts: int = 4, top_k: int = 2, dropout: float = 0.1,
                 attention_on=None, use_gru: bool = True, predict_delta: bool = False):
        super().__init__()
        self.encoders = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                              nn.Linear(hidden, hidden))
            for nt, d in in_dims.items()
        })
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in in_dims})
        self.dropout = nn.Dropout(dropout)
        self.heads = PredictionHeads(
            hidden, num_pl_bins=num_pl_bins, use_moe=use_moe,
            num_experts=num_experts, k=top_k, dropout=dropout,
            predict_delta=predict_delta)

    def forward(self, x_dict, edge_index_dict=None):
        h = {nt: self.dropout(self.norms[nt](self.encoders[nt](x)))
             for nt, x in x_dict.items()}
        return self.heads(h)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
