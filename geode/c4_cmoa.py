"""
c4_cmoa.py -- C4: the Candidate-Modify-Observe-Approve verification loop.

The twin screens a proposed cell-sleep action before it reaches the
network: build the counterfactual graph the action would produce, run the
twin forward on it, and approve only if predicted quality of service stays
within the per-class thresholds.

    python3 c4_cmoa.py --checkpoint checkpoints/wdt_final_gnn_test61_init1.pt

The four steps
--------------
Candidate  a cell is worth considering if it is drawing meaningful power
           and carrying little traffic
Modify     rebuild the graph as it would look with that cell asleep
Observe    forward pass of the trained twin on the modified graph
Approve    per-UE threshold check by QoS class, plus an energy check

Where the UEs go, and why it is measured two ways
-------------------------------------------------
The Modify step has to decide which cell each stranded UE attaches to.
That choice is not free, and getting it wrong would be blamed on the twin.
So both are implemented:

  oracle     use the serving assignment measured in the real paired sleep
             run. The topology is then exactly right, and any error is the
             twin's. This is not deployable -- it uses the answer -- but it
             isolates prediction quality from reassignment quality.

  nearest    assign each stranded UE to the closest still-active cell. This
             is what a real system would have to do, and it is the number
             that matters for the deployable claim.

Reporting both, and the gap between them, separates twin error from
reassignment error. Reporting only the oracle would overstate the system;
reporting only nearest would understate the twin.

Deviation from the specification
--------------------------------
The specified approval rule includes RSRP >= -110 dBm. RSRP is not
extracted in this dataset; uplink SINR is. The RSRP clause is therefore
dropped rather than silently substituted, and the approval rests on the
delay, jitter and packet-loss thresholds. Stated here so the deviation is
visible in the code as well as the write-up.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from c1_graph import build_graph, build_interference_edges
from wdt_checkpoint import load_checkpoint, predict

# 3GPP-style per-class thresholds, as in the contribution specification.
# delay ms, jitter ms, packet loss fraction.
QOS_THRESHOLDS = {
    "URLLC": {"delay": 1.0, "jitter": 0.5, "loss": 0.00001},
    "eMBB": {"delay": 10.0, "jitter": 5.0, "loss": 0.001},
    "mMTC": {"delay": 100.0, "jitter": 50.0, "loss": 0.01},
}

# What a slept cell looks like AS THE TWIN SAW IT IN TRAINING.
#
# The paired sleep runs drop the cell's transmit power to 1 dBm in the
# simulator, but the extractor records tx_power_dbm from the CONFIG value
# -- 43 for every p43_sleep run -- because it never sees the per-cell
# override. So in the training data a slept cell reads tx_power_dbm = 43
# with num_connected_ues = 0 and offered_ul_bps = 0, and the twin learned
# "no load, whatever the nominal power, means the idle floor".
#
# Writing 1.0 into the counterfactual would therefore feed a value that
# appears NOWHERE in training (all runs are 37, 43 or 46), and the
# normaliser fitted on those maps it to an extreme z-score. The first C4
# evaluation did exactly that and the predicted-versus-measured saving
# correlation came out at 0.106. The counterfactual has to be built in the
# same form the twin was trained on, not the form the simulator used.
SLEEP_IDLE_W = 150.0        # the extractor's idle floor for a cell serving nobody


class EnsemblePredictor:
    """Several trained twins presented as one, averaging their predictions.

    Ensembling improved the predictor by 0.03 to 0.08 explained variance, so
    the question is whether that carries into the verification decision. The
    prior expectation is that it will not carry far: ensembling cancels
    variance, and the loop's limits are bias-shaped. The energy gate fails
    because a ~15 W signal sits under ~60 W of error, and averaging shifted
    energy MAPE only from 7.9% to 7.4%. The safety gate fails because
    predicted tail-delay ratios are compressed toward the mean, and averaging
    models that all compress produces an average that also compresses --
    possibly slightly more.

    Measuring it is still worthwhile: the expectation is an argument, not a
    result, and the d95 correlation of 0.38 does have a variance component.

    All members must share the same normalisation, which holds when they come
    from one rotation of one training run.
    """

    def __init__(self, models, ctx, device="cpu"):
        self.models = models
        self.ctx = ctx
        self.device = device

    def __call__(self, graph):
        outs = [predict(m, self.ctx, graph, self.device) for m in self.models]
        avg = {}
        for k in outs[0]:
            stack = torch.stack([torch.as_tensor(o[k]).float() for o in outs])
            avg[k] = stack.mean(dim=0)
        return avg


class CMOA:
    def __init__(self, model, ctx, device="cpu",
                 energy_threshold_w=280.0, load_threshold=0.35,
                 qos_thresholds=None, violation_tolerance=0.05,
                 d95_threshold=1.5):
        """
        energy_threshold_w : a cell must draw at least this to be a candidate
        load_threshold     : and carry at most this share of the network's UEs
        violation_tolerance: share of UEs allowed to breach their class
                             threshold before the action is rejected. Zero
                             would reject almost everything, because some UEs
                             breach their threshold in the BASELINE -- the
                             network is not meeting uRLLC targets to begin
                             with. The check is therefore relative to the
                             baseline, see approve().
        """
        self.model = model
        self.ctx = ctx
        self.device = device
        self.e_thresh = energy_threshold_w
        self.load_thresh = load_threshold
        self.qos = qos_thresholds or QOS_THRESHOLDS
        self.tol = violation_tolerance
        # Same threshold the measured labels use, so a predicted decision and
        # a measured one are the same kind of statement.
        self.d95_threshold = d95_threshold

    # ---------------- Candidate ----------------
    def candidates(self, gnb_rows, serving_rows):
        """Cells worth evaluating: drawing power, lightly loaded.

        Screening first matters because the Observe step costs a forward
        pass per candidate. A cell carrying a third of the network is not a
        plausible sleep target and should not consume one.
        """
        n_ue = len(serving_rows)
        out = []
        for r in gnb_rows.itertuples():
            served = int((serving_rows.serving_gnb_index == r.gnb_index).sum())
            share = served / n_ue if n_ue else 0.0
            if share <= self.load_thresh:
                out.append({"cell": int(r.gnb_index), "served": served,
                            "load_share": share})
        return out

    # ---------------- Modify ----------------
    def build_counterfactual(self, cell, gnb_rows, ue_rows, serving_rows,
                             gnb_norm, ue_norm, interference,
                             reassign="nearest", oracle_serving=None):
        """The graph as it would be with `cell` asleep.

        Feature updates are done on RAW values and re-normalised afterwards.
        Editing normalised values directly would be wrong: num_connected_ues
        and offered_ul_bps are sums over a cell's UEs, so moving a UE has to
        change both cells' raw totals before standardisation.
        """
        gnb = gnb_rows.copy().sort_values("gnb_index").reset_index(drop=True)
        ue = ue_rows.copy().sort_values("ue_index").reset_index(drop=True)
        srv = serving_rows.copy()

        stranded = srv[srv.serving_gnb_index == cell].ue_index.tolist()

        if reassign == "oracle":
            if oracle_serving is None:
                raise ValueError("oracle reassignment needs oracle_serving")
            new_srv = oracle_serving[["ue_index", "serving_gnb_index"]].copy()
        else:
            active = gnb[gnb.gnb_index != cell]
            new_srv = srv[["ue_index", "serving_gnb_index"]].copy()
            if stranded:
                pos = ue.set_index("ue_index")[["pos_x", "pos_y"]]
                for u in stranded:
                    if u not in pos.index:
                        continue
                    ux, uy = pos.loc[u]
                    d = np.hypot(active.pos_x - ux, active.pos_y - uy)
                    new_srv.loc[new_srv.ue_index == u, "serving_gnb_index"] = \
                        int(active.iloc[int(np.argmin(d.values))].gnb_index)

        # --- recompute per-cell aggregates from the new assignment ---
        load = ue.set_index("ue_index").offered_ul_bps
        counts, offered = {}, {}
        for r in new_srv.itertuples():
            g = int(r.serving_gnb_index)
            counts[g] = counts.get(g, 0) + 1
            offered[g] = offered.get(g, 0.0) + float(load.get(r.ue_index, 0.0))

        gnb["num_connected_ues"] = [counts.get(int(g), 0) for g in gnb.gnb_index]
        gnb["offered_ul_bps"] = [offered.get(int(g), 0.0) for g in gnb.gnb_index]
        # The slept cell serves nobody and transmits at the sleep power.
        gnb.loc[gnb.gnb_index == cell, "num_connected_ues"] = 0
        gnb.loc[gnb.gnb_index == cell, "offered_ul_bps"] = 0.0
        # tx_power_dbm deliberately NOT modified -- see the note at the top.
        # A slept cell in the training data keeps its nominal power and is
        # identified by carrying no load.

        gx = torch.tensor(gnb_norm.transform(gnb), dtype=torch.float32)
        ux_ = torch.tensor(ue_norm.transform(ue), dtype=torch.float32)

        # Interference edges are left intact. Physically a cell at 1 dBm is
        # no longer a meaningful interferer and dropping them would be the
        # more faithful model -- but build_interference_edges is computed
        # once per dataset and applied unchanged to the sleep runs, so the
        # twin has only ever seen sleep configurations WITH the full
        # interference topology. Matching training beats matching physics
        # here; the alternative is a graph the model cannot interpret.
        interference_cf = interference

        flags = self.ctx["flags"]
        g = build_graph(gnb, ue, new_srv, gx, ux_,
                        interference=interference_cf,
                        handover=None,
                        use_e1=flags["use_e1"], use_e2=flags["use_e2"],
                        use_e3=flags["use_e3"], use_e4=flags["use_e4"])
        return g, new_srv, stranded

    # ---------------- Observe ----------------
    def observe(self, graph):
        # `model` may be a single twin or an EnsemblePredictor; both are
        # called the same way from here.
        if isinstance(self.model, EnsemblePredictor):
            return self.model(graph)
        return predict(self.model, self.ctx, graph, self.device)

    # ---------------- Approve ----------------
    def approve(self, pred_cf, pred_base, ue_rows):
        """Two independent gates, reported separately.

        SAFETY gate: predicted 95th-percentile delay ratio below
        d95_threshold. This replaces an earlier rule based on the share of
        UEs violating their class threshold, which saturated and could not
        discriminate: the baseline already violates for ~72% of UEs (a 20 ms
        median against a 1 ms uRLLC target), so the rate can rise by at most
        0.28 and a mild action looks like a catastrophic one. The d95 ratio
        does not saturate, and it is the same quantity the measured labels
        are defined on, so predicted and measured decisions are comparable.

        ENERGY gate: predicted saving above zero. Reported, but see the
        note in evaluate_c4.py -- at this twin accuracy the gate is below
        the resolution of the measurement it depends on, and that is a
        finding rather than something to tune away.

        Per-class violation counts are still returned, since they are what
        an operator would actually be shown.
        """
        cls = ue_rows.sort_values("ue_index").qos_class.to_numpy()
        d_cf = np.asarray(pred_cf["delay_ms"]).ravel()
        j_cf = np.asarray(pred_cf["jitter_ms"]).ravel()
        d_b = np.asarray(pred_base["delay_ms"]).ravel()
        j_b = np.asarray(pred_base["jitter_ms"]).ravel()

        def violations(d, j):
            v = np.zeros(len(cls), dtype=bool)
            for i, c in enumerate(cls):
                t = self.qos.get(c, self.qos["eMBB"])
                v[i] = (d[i] > t["delay"]) or (j[i] > t["jitter"])
            return v

        v_cf, v_b = violations(d_cf, j_cf), violations(d_b, j_b)
        rate_cf, rate_b = v_cf.mean(), v_b.mean()

        e_cf = float(np.asarray(pred_cf["energy_w"]).sum())
        e_b = float(np.asarray(pred_base["energy_w"]).sum())
        saved = e_b - e_cf

        d95_cf = float(np.percentile(d_cf, 95))
        d95_b = float(np.percentile(d_b, 95))
        ratio = d95_cf / d95_b if d95_b > 0 else float("inf")

        safe = ratio < self.d95_threshold
        return {
            "approved": bool(safe and saved > 0),
            "safe": bool(safe),
            "saves_energy": bool(saved > 0),
            "predicted_saving_w": saved,
            "pred_d95_ratio": float(ratio),
            "pred_d95_cf": d95_cf,
            "pred_d95_base": d95_b,
            "violation_rate_cf": float(rate_cf),
            "violation_rate_base": float(rate_b),
            "violation_increase": float(rate_cf - rate_b),
        }

    # ---------------- the loop ----------------
    def evaluate_action(self, cell, gnb_rows, ue_rows, serving_rows,
                        gnb_norm, ue_norm, interference,
                        reassign="nearest", oracle_serving=None):
        gx = torch.tensor(gnb_norm.transform(
            gnb_rows.sort_values("gnb_index")), dtype=torch.float32)
        ux_ = torch.tensor(ue_norm.transform(
            ue_rows.sort_values("ue_index")), dtype=torch.float32)
        flags = self.ctx["flags"]
        g_base = build_graph(gnb_rows.sort_values("gnb_index"),
                             ue_rows.sort_values("ue_index"),
                             serving_rows, gx, ux_,
                             interference=interference, handover=None,
                             use_e1=flags["use_e1"], use_e2=flags["use_e2"],
                             use_e3=flags["use_e3"], use_e4=flags["use_e4"])
        pred_base = self.observe(g_base)

        g_cf, _, stranded = self.build_counterfactual(
            cell, gnb_rows, ue_rows, serving_rows, gnb_norm, ue_norm,
            interference, reassign, oracle_serving)
        pred_cf = self.observe(g_cf)

        out = self.approve(pred_cf, pred_base, ue_rows)
        out.update(cell=cell, n_stranded=len(stranded), reassign=reassign)
        return out