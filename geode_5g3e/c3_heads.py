"""
c3_heads.py -- C3 Mixture-of-Experts prediction heads, 5G3E variant.

Ported from src/v6/c3_heads.py: MoEHead (n experts, top-k gating) and a
two-stage PacketLossHead (binary occurrence + severity bin) are reused
unchanged -- both are node-type-agnostic, they just consume whatever
hidden vector they're given.

The one real change from v6
----------------------------
06c has per-UE KPI ground truth, so v6's PredictionHeads reads energy
from gNB hidden state and delay/jitter/packet_loss from UE hidden state.
5G3E has no such per-UE measurement -- the only ground truth is each
gNB's own average_latency (see preprocess_v3.py). All four targets are
therefore gNB-level here: energy, delay, jitter AND packet_loss all read
h["gnb"], none read h["ue"]. UE nodes still exist and still pass
messages (their RSRP/SNR/buffer state informs a gNB's aggregate delay
via the serving edge) -- they just carry no head of their own.

Packet loss is a constant on this testbed -- kept, reported as N/A
--------------------------------------------------------------------
Measured directly: packet_loss is 0.0 in every sampled snapshot across
the whole training split (0/1586 nonzero), not just mostly zero like
06c's 99.6%. The PacketLossHead is kept anyway so the architecture is
identical to 06c's for a clean cross-dataset comparison, but training
loop / manuscript are responsible for reporting its metric as N/A --
undefined, since a constant label has no AUROC to speak of -- rather
than as a real result. Do not present a PL score computed on this head
as evidence of anything.
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


class PacketLossHead(nn.Module):
    """Two stages: is loss non-zero, and if so which severity bin.

    On 5G3E, "is loss non-zero" has exactly one true answer across the
    whole dataset (no) -- see module docstring.
    """

    def __init__(self, hidden, num_bins=5, use_moe=True,
                 num_experts=4, k=2, dropout=0.1):
        super().__init__()
        self.binary = _head(hidden, 1, use_moe, num_experts, k, dropout)
        self.bins = _head(hidden, num_bins, use_moe, num_experts, k, dropout)

    def forward(self, h):
        return self.binary(h).squeeze(-1), self.bins(h)


class BinnedHead(nn.Module):
    """Two-stage jitter head, ported from src/v3/model_v3.py's BinnedHead.

    Stage 1 -- binary BCE gate: is jitter meaningfully non-zero (y_norm > 0)?
    Stage 2 -- MoE bin classifier (CE loss) over n_bins spanning
               [bin_lo, bin_hi] in NORMALISED label space.
    Point prediction = p(Stage1) * E[bin value under Stage2]  (soft gate:
    collapses toward 0 when Stage 1 says "trivial", follows Stage 2
    otherwise). That soft value is what gets scored for R2/MAPE; the two
    stages' own logits are what the loss and the bin-accuracy metric use.

    Defaults (60 bins, [-2.0, 4.0], bin width 0.1) are v3's own choice for
    jitter specifically, carried over unchanged: v7 reuses v3's exact
    norm_stats.pt, so jitter's normalised distribution here is the same
    one those bins were sized for. 10 bins = 1.0 normalised unit = 1 std,
    which is what "within +/-1 std" bin-accuracy means below.
    """

    def __init__(self, hidden, n_bins=60, bin_lo=-2.0, bin_hi=4.0,
                 num_experts=4, k=2, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.bin_lo = bin_lo
        self.bin_hi = bin_hi
        self.binary = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.bin_moe = MoEHead(hidden, n_bins, num_experts, k, dropout)
        edges = torch.linspace(bin_lo, bin_hi, n_bins + 1)
        self.register_buffer("bin_centres", (edges[:-1] + edges[1:]) / 2.0)

    def forward(self, h):
        binary_logit = self.binary(h).squeeze(-1)
        p = torch.sigmoid(binary_logit)
        bin_logits = self.bin_moe(h)
        bin_probs = F.softmax(bin_logits, dim=-1)
        stage2_value = (bin_probs * self.bin_centres).sum(-1)
        return p * stage2_value, binary_logit, bin_logits

    def discretise(self, y_norm):
        """Normalised target -> bin index, for the Stage 2 CE loss and for
        the bin-accuracy metric."""
        width = (self.bin_hi - self.bin_lo) / self.n_bins
        idx = ((y_norm - self.bin_lo) / width).long()
        return idx.clamp(0, self.n_bins - 1)


class PredictionHeads(nn.Module):
    """Energy, delay, jitter and packet loss -- all on gNB nodes.

    Unlike v6's split (energy on gNB, delay/jitter/PL on UE), every
    target here reads the same gNB hidden state, because 5G3E's only
    ground truth is per-gNB. Jitter is the one head that isn't plain
    regression -- see BinnedHead above -- UNLESS predict_delta is set.

    predict_delta: heads predict the CHANGE from the previous snapshot
    (requires wdt_data.py's use_lag=True, so that previous value is
    available as an input feature and as a reconstruction target in
    train_wdt_5g3e.py) rather than the absolute value. Motivated
    directly by the lag-feature finding in SWEEP_RESULTS.md: once a
    node's own recent history is available, most of the answer is
    "close to what it was a moment ago", so the head only has to learn
    the smaller residual on top rather than the full absolute value.
    BinnedHead's fixed bin range [-2,4] was sized for the ABSOLUTE
    jitter distribution and has no natural meaning for a delta target,
    so predict_delta swaps jitter to a plain regression head too --
    this is also the target where gnn+lag still trails RF+lag on R2,
    so it's the one delta-framing is actually meant to help.
    """

    def __init__(self, hidden, num_pl_bins=5, num_jitter_bins=60,
                 use_moe=True, num_experts=4, k=2, dropout=0.1,
                 predict_delta=False):
        super().__init__()
        self.use_moe = use_moe
        self.predict_delta = predict_delta
        self.energy = _head(hidden, 1, use_moe, num_experts, k, dropout)
        self.delay = _head(hidden, 1, use_moe, num_experts, k, dropout)
        self.jitter = (_head(hidden, 1, use_moe, num_experts, k, dropout)
                       if predict_delta else
                       BinnedHead(hidden, n_bins=num_jitter_bins,
                                 num_experts=num_experts, k=k, dropout=dropout))
        self.packet_loss = PacketLossHead(hidden, num_pl_bins, use_moe,
                                          num_experts, k, dropout)

    def forward(self, h_dict):
        g = h_dict["gnb"]
        pl_bin, pl_bins = self.packet_loss(g)
        out = {
            "energy": self.energy(g),
            "delay": self.delay(g),
            "pl_binary_logit": pl_bin,
            "pl_bin_logits": pl_bins,
        }
        if self.predict_delta:
            out["jitter"] = self.jitter(g).squeeze(-1)  # match BinnedHead's 1-D shape
        else:
            j_val, j_bin, j_bins = self.jitter(g)
            out["jitter"] = j_val
            out["jitter_binary_logit"] = j_bin
            out["jitter_bin_logits"] = j_bins
        return out
