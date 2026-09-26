"""
c3_heads.py -- C3 Mixture-of-Experts prediction heads.

Spec: four heads (energy, delay, jitter, packet loss), each a MoE of four
experts with top-2 sparse gating; hierarchical two-stage classifier for
packet loss.

Standing caveat
---------------
The knowledge base records "single-task beats joint MoE" as a locked
decision from the scenario07 work. That finding is not overturned here
and this module does not claim otherwise. It exists so the spec
architecture can be measured on scenario06c, which differs from
scenario07 in the respects that plausibly mattered -- balanced load,
bounded congestion, 34% more handovers. If MoE loses again on this data
the earlier decision is confirmed across two datasets, which is a
stronger statement than it currently rests on.

use_moe=False collapses each head to a plain MLP, which is the
single-task-equivalent head. That is the control the MoE must beat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEHead(nn.Module):
    """n experts, top-k sparse gating, weighted sum of the chosen experts."""

    def __init__(self, hidden, out_dim, num_experts=4, k=2, dropout=0.1):
        super().__init__()
        self.k = min(k, num_experts)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(hidden, out_dim))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden, num_experts)

    def forward(self, h):
        logits = self.gate(h)                                  # [N, E]
        topv, topi = torch.topk(logits, self.k, dim=-1)        # [N, k]
        w = F.softmax(topv, dim=-1)                            # renormalised

        out = None
        for slot in range(self.k):
            idx = topi[:, slot]                                # [N]
            weight = w[:, slot].unsqueeze(-1)                  # [N,1]
            # Evaluate each expert once over the nodes routed to it,
            # rather than all experts over all nodes.
            part = torch.zeros(h.shape[0], self.experts[0][-1].out_features,
                               device=h.device, dtype=h.dtype)
            for e in range(self.num_experts):
                mask = idx == e
                if mask.any():
                    part[mask] = self.experts[e](h[mask])
            out = part * weight if out is None else out + part * weight
        return out

    def gate_entropy(self, h):
        """Diagnostic: high entropy means the gate is not specialising and
        the MoE is behaving like an ensemble average."""
        p = F.softmax(self.gate(h), dim=-1)
        return -(p * torch.log(p + 1e-9)).sum(-1).mean()


def _head(hidden, out_dim, use_moe, num_experts, k, dropout):
    if use_moe:
        return MoEHead(hidden, out_dim, num_experts, k, dropout)
    return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden, out_dim))


def _deep_head(hidden, out_dim, dropout, width_mult=2, depth=3):
    """Higher-capacity plain MLP: `width_mult`x hidden width, `depth`
    hidden layers, single path (no expert routing). Diagnostic for
    whether a target's head is capacity-starved rather than needing
    MoE -- see the RF-beats-gnn-on-06c-energy finding in
    src/v7_5g3e/SWEEP_RESULTS.md. Keeps the shared encoder untouched,
    so this stays inside the joint multi-task architecture rather than
    proposing a separate single-task model."""
    w = hidden * width_mult
    layers = [nn.Linear(hidden, w), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(depth - 1):
        layers += [nn.Linear(w, w), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(w, out_dim))
    return nn.Sequential(*layers)


class PacketLossHead(nn.Module):
    """Two stages: is loss non-zero, and if so which severity bin.

    Packet loss is zero in roughly 99.6% of scenario06c samples. Regressing
    on that directly makes the trivial all-zero prediction near-optimal by
    MSE, which is why it is split: a binary head for occurrence, and a bin
    classifier trained only on the non-zero cases.
    """

    def __init__(self, hidden, num_bins=5, use_moe=True,
                 num_experts=4, k=2, dropout=0.1):
        super().__init__()
        self.binary = _head(hidden, 1, use_moe, num_experts, k, dropout)
        self.bins = _head(hidden, num_bins, use_moe, num_experts, k, dropout)

    def forward(self, h):
        return self.binary(h).squeeze(-1), self.bins(h)


class PredictionHeads(nn.Module):
    """Energy on gNB nodes; delay, jitter and packet loss on UE nodes."""

    def __init__(self, hidden, num_pl_bins=5, use_moe=True,
                 num_experts=4, k=2, dropout=0.1, deep_energy=False):
        super().__init__()
        self.use_moe = use_moe
        self.energy = (_deep_head(hidden, 1, dropout) if deep_energy
                       else _head(hidden, 1, use_moe, num_experts, k, dropout))
        self.delay = _head(hidden, 1, use_moe, num_experts, k, dropout)
        self.jitter = _head(hidden, 1, use_moe, num_experts, k, dropout)
        self.packet_loss = PacketLossHead(hidden, num_pl_bins, use_moe,
                                          num_experts, k, dropout)

    def forward(self, h_dict):
        g, u = h_dict["gnb"], h_dict["ue"]
        pl_bin, pl_bins = self.packet_loss(u)
        return {
            "energy": self.energy(g),
            "delay": self.delay(u),
            "jitter": self.jitter(u),
            "pl_binary_logit": pl_bin,
            "pl_bin_logits": pl_bins,
        }
