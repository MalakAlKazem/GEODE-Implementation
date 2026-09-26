"""
wdt_model.py -- WirelessDT-6G model: C2 encoder + C3 heads.

Also defines the graph-blind control. The control must differ from the
full model in exactly one respect -- the presence of message passing --
or the comparison measures architecture size rather than the value of
the graph. Same encoders, same heads, same hidden width, same head
structure; the only thing removed is the C2 conv/GRU loop.
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
                 use_gru: bool = True, deep_energy: bool = False):
        super().__init__()
        self.encoder = HMPGNNEncoder(
            in_dims, hidden=hidden, rounds=rounds, relations=relations,
            attention_on=attention_on, dropout=dropout, use_gru=use_gru)
        self.heads = PredictionHeads(
            hidden, num_pl_bins=num_pl_bins, use_moe=use_moe,
            num_experts=num_experts, k=top_k, dropout=dropout,
            deep_energy=deep_energy)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return self.heads(h)


class WirelessDT6GBlind(nn.Module):
    """Identical minus message passing: each node predicts from its own
    features alone. This is the number the graph has to beat."""

    def __init__(self, in_dims: dict, relations=None, hidden: int = 64,
                 rounds: int = 10, num_pl_bins: int = 5, use_moe: bool = True,
                 num_experts: int = 4, top_k: int = 2, dropout: float = 0.1,
                 attention_on=None, use_gru: bool = True, deep_energy: bool = False):
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
            deep_energy=deep_energy)

    def forward(self, x_dict, edge_index_dict=None, edge_attr_dict=None):
        h = {nt: self.dropout(self.norms[nt](self.encoders[nt](x)))
             for nt, x in x_dict.items()}
        return self.heads(h)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)