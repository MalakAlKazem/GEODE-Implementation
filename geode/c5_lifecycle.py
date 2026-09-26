"""
c5_lifecycle.py -- C5: the three-phase digital twin lifecycle.

    python3 c5_lifecycle.py --checkpoint checkpoints/wdt_final_gnn_test61_init1.pt

Training -> Operational -> Fine-tuning, where fine-tuning returns to
Operational rather than to Training. The direction matters: the twin is not
rebuilt when it drifts, it is corrected while remaining in service.

  Phase 1  Training      fit on historical data, hold out a topology
  Phase 2  Operational   predict each arriving snapshot, compare against the
                         outcome once it is observed, track rolling error
  Phase 3  Fine-tuning   when rolling error crosses a threshold, freeze the
                         encoder, update the heads on recent observations,
                         return to Operational

What this can and cannot demonstrate
------------------------------------
There is no live network. Phase 2 is simulated by replaying a held-out
topology snapshot by snapshot, which is a faithful test of the *mechanism*
-- does drift get detected, does the correction help, does performance hold
afterwards -- but not of deployment. The honest claim is that the lifecycle
is implemented and its drift-detection and recovery behaviour measured on
replayed data.

Why the encoder is frozen during fine-tuning
--------------------------------------------
Two reasons, one principled and one practical. The encoder learned the
graph structure from 1.8 million UE-snapshots; the fine-tuning buffer holds
a few hundred. Updating it on that would discard far more than it adds. And
freezing it bounds what can go wrong: the relational representation is
fixed, so a bad fine-tune degrades the output mapping rather than the
model's understanding of the network. Recovery is then a matter of
reloading the head weights.
"""

import argparse
import os
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from c1_graph import build_graph, build_interference_edges
from wdt_checkpoint import load_checkpoint, predict
from wdt_data import GNB_COLS, QOS_CLASSES

BASE = os.environ.get("GEODE_S06C", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "scenario06c"))
PHASE = 30


def read_run(folder):
    g = pd.read_csv(os.path.join(folder, "gnb_inputs.csv"))
    u = pd.read_csv(os.path.join(folder, "ue_inputs.csv"))
    k = pd.read_csv(os.path.join(folder, "kpi_targets.csv"))
    e = pd.read_csv(os.path.join(folder, "energy_targets.csv"))
    s = pd.read_csv(os.path.join(folder, "serving_edges.csv"))
    for c in QOS_CLASSES:
        u[f"qos_{c}"] = (u.qos_class == c).astype(np.float32)
    return {"gnb": g, "ue": u, "kpi": k, "energy": e, "serving": s}


class Lifecycle:
    def __init__(self, model, ctx, window=20, mape_trigger=25.0,
                 ft_lr=1e-4, ft_epochs=20, buffer_size=200, device="cpu",
                 holdout=0.3, min_buffer=30):
        """
        window        snapshots in the rolling error estimate
        mape_trigger  rolling delay MAPE above which fine-tuning fires
        ft_lr         fine-tuning learning rate; an order below training,
                      because the aim is to correct the output mapping, not
                      to relearn it
        buffer_size   observations retained for fine-tuning, most recent first
        holdout       fraction of the buffer, most recent, reserved for
                      judging the correction rather than training on it
        min_buffer    fine-tuning is not attempted below this many
                      observations. Correcting on 20 snapshots overfits, and
                      the first run showed exactly that.
        """
        self.model = model
        self.ctx = ctx
        self.device = device
        self.window = window
        self.trigger = mape_trigger
        self.ft_lr = ft_lr
        self.ft_epochs = ft_epochs
        self.buffer = deque(maxlen=buffer_size)
        self.holdout = holdout
        self.min_buffer = min_buffer
        self.rolling = deque(maxlen=window)
        self.events = []
        self.history = []

    # ---------------- Phase 2 ----------------
    def step(self, t, graph, true_delay_ms, true_energy_w):
        """One operational cycle: predict, act, then observe the outcome."""
        p = predict(self.model, self.ctx, graph, self.device)
        pd_ms = np.asarray(p["delay_ms"]).ravel()
        pe_w = np.asarray(p["energy_w"]).ravel()

        nz = np.abs(true_delay_ms) > 1e-6
        d_mape = (np.abs((true_delay_ms[nz] - pd_ms[nz]) / true_delay_ms[nz])
                  .mean() * 100) if nz.any() else np.nan
        e_mape = (np.abs((true_energy_w - pe_w) / true_energy_w).mean() * 100)

        self.rolling.append(d_mape)
        self.buffer.append((graph, true_delay_ms, true_energy_w))
        rolling_mape = float(np.nanmean(self.rolling))

        rec = {"t": t, "delay_mape": float(d_mape), "energy_mape": float(e_mape),
               "rolling_delay_mape": rolling_mape, "finetuned": False}

        # Only consider a trigger once the window is full; before that the
        # rolling estimate is not comparable to the threshold.
        if (len(self.rolling) == self.window and rolling_mape > self.trigger
                and len(self.buffer) >= self.min_buffer):
            gain = self.finetune()
            rec["finetuned"] = True
            rec["ft_gain"] = gain
            self.rolling.clear()      # start a fresh window after correcting
            self.events.append({"t": t, "rolling_before": rolling_mape,
                                "gain": gain})
        self.history.append(rec)
        return rec

    # ---------------- Phase 3 ----------------
    def finetune(self):
        """Freeze the encoder, update the heads on the older part of the
        buffer, and measure the effect on the newest part.

        The first version trained on the whole buffer and reported the change
        in error over that same buffer. That number cannot distinguish a
        genuine correction from overfitting: with 20 snapshots and 20 epochs,
        buffer error falls whether or not the model has learned anything
        transferable. One of the two observed events reported +11.4 pp, which
        is only interpretable once the measurement is separated from the
        training data.

        The buffer is therefore split chronologically: the older
        `1 - holdout` fraction is trained on, the newest `holdout` fraction
        is used to judge. Chronological rather than random, because the
        question is whether correcting on the past helps on what comes next.

        Returns the change in held-out delay MAPE, negative meaning
        improvement.
        """
        if len(self.buffer) < self.min_buffer:
            return float("nan")

        items = list(self.buffer)
        cut = max(int(len(items) * (1 - self.holdout)), 1)
        train_items, eval_items = items[:cut], items[cut:]
        if not eval_items:
            return float("nan")

        for p in self.model.encoder.parameters():
            p.requires_grad = False
        head_params = [p for p in self.model.heads.parameters()]
        opt = torch.optim.AdamW(head_params, lr=self.ft_lr, weight_decay=1e-4)

        dn = self.ctx["kpi_norm"]["delay"]
        en = self.ctx["energy_norm"]

        before = self._mape_over(eval_items)
        self.model.train()
        for _ in range(self.ft_epochs):
            for graph, td, te in train_items:
                g = graph.to(self.device)
                ea = {r: g[r].edge_attr for r in g.edge_types
                      if hasattr(g[r], "edge_attr") and g[r].edge_attr is not None}
                out = self.model(g.x_dict, g.edge_index_dict, ea)
                tgt_d = torch.tensor(dn.transform(td), dtype=torch.float32,
                                     device=self.device)
                tgt_e = torch.tensor(en.transform(te), dtype=torch.float32,
                                     device=self.device).unsqueeze(-1)
                loss = (F.mse_loss(out["delay"].squeeze(-1), tgt_d)
                        + F.huber_loss(out["energy"], tgt_e))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, 5.0)
                opt.step()
        self.model.eval()
        after = self._mape_over(eval_items)

        # Phase 3 returns to Operational, not to Training: the encoder is
        # unfrozen for bookkeeping only, never retrained here.
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        return float(after - before)

    def _mape_over(self, items):
        errs = []
        for graph, td, _ in items:
            p = predict(self.model, self.ctx, graph, self.device)
            pd_ms = np.asarray(p["delay_ms"]).ravel()
            nz = np.abs(td) > 1e-6
            if nz.any():
                errs.append(np.abs((td[nz] - pd_ms[nz]) / td[nz]).mean() * 100)
        return float(np.mean(errs)) if errs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="topology to replay; defaults to the held-out one")
    ap.add_argument("--powers", nargs="+", default=["p37", "p43", "p46"],
                    help="runs replayed back to back as one stream")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="newest fraction of the buffer reserved for judging "
                         "the correction rather than training on it")
    ap.add_argument("--min-buffer", type=int, default=30,
                    help="no fine-tuning below this many observations")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--trigger", type=float, default=25.0)
    ap.add_argument("--out", default="c5_lifecycle.csv")
    args = ap.parse_args()

    model, ctx = load_checkpoint(args.checkpoint)
    seed = args.seed or ctx["test_seed"]
    print(f"checkpoint trained on {ctx['train_seeds']}, held out {ctx['test_seed']}")
    print(f"replaying seed {seed} {args.powers} as the operational stream")
    if seed in (ctx["train_seeds"] or []):
        print("  NOTE: this topology was in training, so drift will be")
        print("  understated. Replay the held-out topology for the real test.")

    flags = ctx["flags"]
    lc = Lifecycle(model, ctx, window=args.window, mape_trigger=args.trigger,
                   holdout=args.holdout, min_buffer=args.min_buffer)

    # Several runs replayed back to back. One run at stride 30 gives only 40
    # cycles, which was too few to say whether fine-tuning helps -- the first
    # attempt produced two events, one helping and one harming. Chaining the
    # power levels multiplies the stream without inventing data, and a real
    # deployment is a continuous stream rather than a single 120 s window.
    streams = args.powers
    print(f"\nPhase 2 -- operational replay over {streams}, "
          f"stride {args.stride}, window {args.window}, "
          f"trigger {args.trigger:.0f}% rolling delay MAPE, "
          f"buffer holdout {args.holdout:.0%}, min buffer {args.min_buffer}")

    t = 0
    for power in streams:
        folder = os.path.join(BASE, f"extracted_s06c{seed}_{power}")
        if not os.path.isdir(folder):
            print(f"  (no {power} for seed {seed}, skipping)")
            continue
        run = read_run(folder)
        snaps = np.sort(run["gnb"].snapshot_id.unique())[args.stride // 2::args.stride]
        snap0 = run["gnb"][run["gnb"].snapshot_id == snaps[0]]
        interference = build_interference_edges(snap0)
        print(f"  -- {power}: {len(snaps)} cycles")

        for sid in snaps:
            gr = run["gnb"][run["gnb"].snapshot_id == sid].sort_values("gnb_index")
            ur = run["ue"][run["ue"].snapshot_id == sid].sort_values("ue_index")
            sr = run["serving"][run["serving"].snapshot_id == sid]
            kr = run["kpi"][run["kpi"].snapshot_id == sid].sort_values("ue_index")
            er = run["energy"][run["energy"].snapshot_id == sid].sort_values("gnb_index")
            if gr.empty or ur.empty or kr.empty:
                continue

            gx = torch.tensor(ctx["gnb_norm"].transform(gr), dtype=torch.float32)
            ux = torch.tensor(ctx["ue_norm"].transform(ur), dtype=torch.float32)
            g = build_graph(gr, ur, sr, gx, ux, interference=interference,
                            handover=None, use_e1=flags["use_e1"],
                            use_e2=flags["use_e2"], use_e3=flags["use_e3"],
                            use_e4=flags["use_e4"])
            rec = lc.step(t, g, kr.delay_ms.to_numpy(),
                          er.estimated_gnb_power_w.to_numpy())
            rec["power"] = power
            if rec["finetuned"]:
                verdict = ("helped" if rec["ft_gain"] < 0
                           else "HARMED" if rec["ft_gain"] > 0 else "no change")
                print(f"    t={t:>4}  rolling {rec['rolling_delay_mape']:>7.1f}%"
                      f" -> FINE-TUNE, held-out MAPE {rec['ft_gain']:+.1f}pp "
                      f"({verdict})")
            t += 1

    h = pd.DataFrame(lc.history)
    h.to_csv(args.out, index=False)

    print(f"\n{'='*70}")
    print("LIFECYCLE SUMMARY")
    print(f"{'='*70}")
    print(f"  cycles: {len(h)}   fine-tuning events: {len(lc.events)}")
    print("\n  NOTE on interpreting rising error: the twin is static and the")
    print("  stream is one topology replayed in time order, so an increase is")
    print("  NOT model staleness. Later traffic phases are more congested and")
    print("  harder to predict. The trigger is responding to scenario")
    print("  non-stationarity, which exercises the same mechanism a real drift")
    print("  would, but is a different phenomenon and should be described as")
    print("  such.")
    print(f"  delay MAPE   first quarter {h.delay_mape[:len(h)//4].mean():>7.1f}%"
          f"   last quarter {h.delay_mape[-len(h)//4:].mean():>7.1f}%")
    print(f"  energy MAPE  first quarter {h.energy_mape[:len(h)//4].mean():>7.1f}%"
          f"   last quarter {h.energy_mape[-len(h)//4:].mean():>7.1f}%")

    if lc.events:
        gains = [e["gain"] for e in lc.events]
        helped = sum(1 for g_ in gains if g_ < 0)
        gains = [g_ for g_ in gains if np.isfinite(g_)]
        helped = sum(1 for g_ in gains if g_ < 0)
        print(f"\n  fine-tuning: {helped}/{len(gains)} events reduced HELD-OUT "
              f"error, mean change {np.mean(gains):+.1f} pp")
        print("  Held-out, not buffer: the newest 30% of the buffer is kept")
        print("  out of the update and used to judge it, so the number cannot")
        print("  be satisfied by overfitting the correction set.")
        if np.mean(gains) > 0:
            print("  A positive mean says the correction is doing net harm --")
            print("  the trigger fires too readily, or the learning rate and")
            print("  epoch count are too aggressive for the buffer size.")
    else:
        print(f"\n  no fine-tuning triggered: rolling delay MAPE stayed below "
              f"{args.trigger:.0f}%")
        print("  Either the twin does not drift on this topology, or the")
        print("  trigger is set above the error the twin actually exhibits.")
        print(f"  Observed rolling MAPE: min {h.rolling_delay_mape.min():.1f}% "
              f"mean {h.rolling_delay_mape.mean():.1f}% "
              f"max {h.rolling_delay_mape.max():.1f}%")
        print("  Set --trigger near the observed mean to exercise Phase 3.")

    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()