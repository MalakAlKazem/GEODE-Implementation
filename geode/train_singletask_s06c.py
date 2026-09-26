"""
train_singletask_s06c.py -- one model per target, to settle whether the
shared encoder costs accuracy on scenario06c.

    python3 train_singletask_s06c.py --targets delay
    python3 train_singletask_s06c.py --targets energy delay jitter packet_loss

Why this exists
---------------
The four-base-station work compared these two arrangements directly and
found separate models decisively better: the delay advantage over the
graph-blind control went from indistinguishable-from-noise under joint
mixture-of-experts heads to +0.48 under separate single-task models, with
jitter going from +0.281 to +0.54. The proposed mechanism was gradient
competition -- energy is the easiest target and dominates a shared
representation, at the expense of the harder quality targets.

Every scenario06c result reported so far uses the shared encoder. That
leaves an open contradiction: the only direct comparison available favours
an arrangement the headline results did not use. This script runs the same
comparison on scenario06c so the question is settled on the dataset the
results come from.

Two independent scenario06c findings point the same way without testing it.
Energy and the quality targets were shown to want different data sampling
rates, and packet loss was shown not to benefit from the graph while delay
does. A single shared encoder cannot accommodate either.

What is held constant
---------------------
Everything except the head arrangement: same rotation over three held-out
topologies, same initialisations, same stride, same normalisation fitted on
training snapshots only, same graph-blind control trained identically. The
control is retrained per target here too, since a single-task control is
the correct comparison for a single-task model.

Interpreting the result
-----------------------
The number to compare is the GAP against the control, not the absolute
score. A single-task model may be more accurate simply because it is
solving an easier problem; what matters is whether the graph's advantage
over an identically-configured control is larger or smaller than it is
under joint training.

One caveat carried from the earlier work: where the largest gaps were
measured there, the graph's absolute accuracy was near zero and the gap
partly reflected the control falling below the test mean. Mean absolute
error is the fairer statistic in that regime, so it is reported alongside.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from c1_graph import edge_types_present
from c2_hmpgnn import HMPGNNEncoder
from c3_heads import PacketLossHead, _head
from wdt_data import (load_runs, SnapshotDataset, Normalizer,
                      LabelNormalizer, compute_bin_edges, GNB_COLS)

BASE = os.environ.get("GEODE_S06C", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "scenario06c"))
SEEDS = (61, 62, 63)
POWERS = ("p37", "p43", "p46")
SLEEP = {61: [0, 1, 2, 3, 4, 5], 62: [0, 1, 2, 5], 63: [0, 1, 2, 3, 4, 5]}
PHASE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Which node type each target is read from, and how many outputs it needs.
TARGETS = {
    "energy": {"node": "gnb", "dim": 1},
    "delay": {"node": "ue", "dim": 1},
    "jitter": {"node": "ue", "dim": 1},
    "packet_loss": {"node": "ue", "dim": None},   # two-stage head
}


class SingleTaskGNN(nn.Module):
    """The same encoder as the joint model, with exactly one head.

    Deliberately not a reduced encoder: if the encoder were also shrunk,
    a difference in result could be attributed to capacity rather than to
    the head arrangement, which is the thing under test.
    """

    def __init__(self, target, in_dims, relations, hidden=64, rounds=4,
                 num_pl_bins=5, use_moe=True, dropout=0.1,
                 attention_on=("gnb", "interferes", "gnb")):
        super().__init__()
        self.target = target
        self.node = TARGETS[target]["node"]
        self.encoder = HMPGNNEncoder(
            in_dims, hidden=hidden, rounds=rounds, relations=relations,
            attention_on=attention_on, dropout=dropout)
        if target == "packet_loss":
            self.head = PacketLossHead(hidden, num_bins=num_pl_bins,
                                       use_moe=use_moe, dropout=dropout)
        else:
            self.head = _head(hidden, 1, use_moe, 4, 2, dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return self.head(h[self.node])


class SingleTaskBlind(nn.Module):
    """Identical minus message passing, per target."""

    def __init__(self, target, in_dims, relations=None, hidden=64, rounds=4,
                 num_pl_bins=5, use_moe=True, dropout=0.1, attention_on=None):
        super().__init__()
        self.target = target
        self.node = TARGETS[target]["node"]
        self.encoders = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                              nn.Linear(hidden, hidden))
            for nt, d in in_dims.items()})
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        if target == "packet_loss":
            self.head = PacketLossHead(hidden, num_bins=num_pl_bins,
                                       use_moe=use_moe, dropout=dropout)
        else:
            self.head = _head(hidden, 1, use_moe, 4, 2, dropout)

    def forward(self, x_dict, edge_index_dict=None, edge_attr_dict=None):
        h = self.drop(self.norm(self.encoders[self.node](x_dict[self.node])))
        return self.head(h)


def run_paths_for(seeds):
    p = {}
    for s in seeds:
        for pw in POWERS:
            p[f"s{s}_{pw}"] = os.path.join(BASE, f"extracted_s06c{s}_{pw}")
        for k in SLEEP[s]:
            p[f"s{s}_p43_sleep{k}"] = os.path.join(
                BASE, f"extracted_s06c{s}_p43_sleep{k}")
    return p


def subsample(snaps, stride, phase=PHASE):
    snaps = np.asarray(snaps)
    if stride <= 1:
        return snaps
    return snaps[min(stride // 2, phase // 2)::stride]


def metrics(y_true, y_pred, denorm=None):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if denorm is not None:
        y_true = np.asarray(denorm.inverse(y_true), dtype=np.float64)
        y_pred = np.asarray(denorm.inverse(y_pred), dtype=np.float64)
    mae = np.abs(y_true - y_pred).mean()
    nz = np.abs(y_true) > 1e-6
    mape = (np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]).mean() * 100
            if nz.any() else float("nan"))
    return {"R2": float(r2), "MAE": float(mae), "MAPE": float(mape)}


def auroc(y, s):
    y = np.asarray(y).ravel(); s = np.asarray(s).ravel()
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def run_epoch(model, loader, target, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    yt, yp = [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            ea = {r: batch[r].edge_attr for r in batch.edge_types
                  if hasattr(batch[r], "edge_attr")
                  and batch[r].edge_attr is not None}
            out = model(batch.x_dict, batch.edge_index_dict, ea)

            if target == "energy":
                true = batch["gnb"].y
                loss = F.huber_loss(out, true)
                pred = out
            elif target == "delay":
                true = batch["ue"].y[:, 0]
                loss = F.mse_loss(out.squeeze(-1), true)
                pred = out.squeeze(-1)
            elif target == "jitter":
                true = batch["ue"].y[:, 1]
                loss = F.mse_loss(out.squeeze(-1), true)
                pred = out.squeeze(-1)
            else:                                   # packet_loss
                pl_bin, pl_bins = out
                true = batch["ue"].pl_binary
                loss = F.binary_cross_entropy_with_logits(pl_bin, true)
                nzm = true > 0
                if nzm.any():
                    loss = loss + F.cross_entropy(
                        pl_bins[nzm], batch["ue"].pl_class[nzm],
                        label_smoothing=0.1)
                pred = torch.sigmoid(pl_bin)

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        bs = true.shape[0]
        tot += float(loss) * bs
        n += bs
        yt.append(true.detach().cpu().numpy())
        yp.append(pred.detach().cpu().numpy())

    return tot / max(n, 1), np.concatenate(yt), np.concatenate(yp)


def one_rotation(test_seed, target, args):
    train_seeds = [s for s in SEEDS if s != test_seed]
    data = load_runs(run_paths_for(SEEDS))
    ue_cols = data["ue_cols"]

    tag_snaps = (data["gnb"][["run_tag", "snapshot_id"]].drop_duplicates()
                 .groupby("run_tag").snapshot_id
                 .apply(lambda s: np.sort(s.unique())).to_dict())

    def tags_of(seed):
        return [t for t in tag_snaps if t.startswith(f"s{seed}_")]

    tr_ids, va_ids = [], []
    for s in train_seeds:
        for t in tags_of(s):
            snaps = tag_snaps[t]
            n_phase = len(snaps) // PHASE
            n_val = max(int(round(n_phase * args.val_frac)), 1)
            cut = (n_phase - n_val) * PHASE
            tr_ids.extend(subsample(snaps[:cut], args.stride).tolist())
            va_ids.extend(subsample(snaps[cut:], args.stride).tolist())
    te_ids = []
    for t in tags_of(test_seed):
        te_ids.extend(subsample(tag_snaps[t], args.stride).tolist())
    tr_ids, va_ids, te_ids = sorted(tr_ids), sorted(va_ids), sorted(te_ids)

    gtr = data["gnb"][data["gnb"].snapshot_id.isin(tr_ids)]
    utr = data["ue"][data["ue"].snapshot_id.isin(tr_ids)]
    gnb_norm = Normalizer(GNB_COLS).fit(gtr)
    ue_norm = Normalizer(ue_cols).fit(utr)

    etr = data["energy"][data["energy"].snapshot_id.isin(tr_ids)]
    ktr = data["kpi"][data["kpi"].snapshot_id.isin(tr_ids)]
    energy_norm = LabelNormalizer().fit(etr.estimated_gnb_power_w.to_numpy())
    kpi_norms = {"delay": LabelNormalizer().fit(ktr.delay_ms.to_numpy()),
                 "jitter": LabelNormalizer().fit(ktr.jitter_ms.to_numpy())}
    pl = ktr.packet_loss.to_numpy()
    edges = compute_bin_edges(pl[pl > 0], num_bins=5)
    n_bins = max(len(edges) - 1, 2)

    flags = dict(use_e1=True, use_e2=True, use_e3=False, use_e4=True)
    mk = lambda ids: SnapshotDataset(data, ids, gnb_norm, ue_norm,
                                     energy_norm, kpi_norms, edges, **flags)
    tr_ds, va_ds, te_ds = mk(tr_ids), mk(va_ids), mk(te_ids)

    rels = edge_types_present(**flags)
    in_dims = {"gnb": len(GNB_COLS), "ue": len(ue_cols)}
    if flags["use_e4"]:
        in_dims["core"] = len(GNB_COLS)

    denorm = {"energy": energy_norm, "delay": kpi_norms["delay"],
              "jitter": kpi_norms["jitter"]}.get(target)

    print(f"\n  === test seed {test_seed} | target {target} ===")
    print(f"      train {len(tr_ds)}  val {len(va_ds)}  test {len(te_ds)}")

    out = {"gnn": [], "blind": []}
    for init in args.inits:
        for kind, Cls in (("gnn", SingleTaskGNN), ("blind", SingleTaskBlind)):
            torch.manual_seed(init)
            model = Cls(target, in_dims, rels, hidden=args.hidden,
                        rounds=args.rounds, num_pl_bins=n_bins,
                        use_moe=not args.no_moe,
                        dropout=args.dropout).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
            trl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
            val = DataLoader(va_ds, batch_size=args.batch)
            tel = DataLoader(te_ds, batch_size=args.batch)

            best, best_state, best_ep, bad = float("inf"), None, 0, 0
            for ep in range(1, args.epochs + 1):
                run_epoch(model, trl, target, opt)
                vl, _, _ = run_epoch(model, val, target)
                if vl < best - 1e-5:
                    best, best_ep, bad = vl, ep, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            model.load_state_dict(best_state)
            _, yt, yp = run_epoch(model, tel, target)

            if target == "packet_loss":
                m = {"AUROC": auroc(yt, yp)}
                score = m["AUROC"]
            else:
                m = metrics(yt, yp, denorm)
                score = m["R2"]
            m["_best_epoch"] = best_ep
            out[kind].append(m)
            extra = (f"  MAE {m['MAE']:.2f}" if "MAE" in m else "")
            print(f"      [{kind:5}] init {init:>2}  ep {best_ep:>3}  "
                  f"score {score:.4f}{extra}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["delay"],
                    choices=list(TARGETS))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--inits", type=int, nargs="+", default=[1, 7, 13])
    ap.add_argument("--test-seed", type=int, default=None)
    ap.add_argument("--no-moe", action="store_true")
    ap.add_argument("--out", default="singletask_results.json")
    args = ap.parse_args()

    print("Single-task comparison on scenario06c")
    print("Head arrangement is the only thing that differs from the joint")
    print("configuration; encoder, data, rotation and control are identical.")

    results = {}
    for target in args.targets:
        print(f"\n{'='*70}\nTARGET: {target}\n{'='*70}")
        rot = [args.test_seed] if args.test_seed else list(SEEDS)
        per_seed = {s: one_rotation(s, target, args) for s in rot}
        results[target] = per_seed

        key = "AUROC" if target == "packet_loss" else "R2"
        g = [m[key] for r in per_seed.values() for m in r["gnn"]]
        b = [m[key] for r in per_seed.values() for m in r["blind"]]
        gaps = np.array(g) - np.array(b)
        print(f"\n  {target}: single-task gnn {np.mean(g):.4f} \u00b1 {np.std(g):.4f}"
              f"   blind {np.mean(b):.4f} \u00b1 {np.std(b):.4f}")
        print(f"  {'':<{len(target)}}  gap {gaps.mean():+.4f} "
              f"({int((gaps > 0).sum())}/{len(gaps)} positive)")
        if target != "packet_loss":
            gm = [m["MAE"] for r in per_seed.values() for m in r["gnn"]]
            bm = [m["MAE"] for r in per_seed.values() for m in r["blind"]]
            print(f"  {'':<{len(target)}}  MAE  gnn {np.mean(gm):.2f}  "
                  f"blind {np.mean(bm):.2f}  "
                  f"reduction {(1-np.mean(gm)/np.mean(bm))*100:.1f}%")

    with open(args.out, "w") as f:
        json.dump({t: {str(s): v for s, v in r.items()}
                   for t, r in results.items()}, f, indent=2)
    print(f"\nwritten {args.out}")

    print("\n" + "=" * 70)
    print("COMPARE AGAINST THE JOINT CONFIGURATION")
    print("=" * 70)
    print("  Joint mixture-of-experts, same dataset, same rotation:")
    print("    energy   gnn 0.514  blind 0.225  gap +0.289  (9/9)")
    print("    delay    gnn 0.518  blind 0.067  gap +0.451  (9/9)")
    print("    jitter   gnn 0.491  blind 0.050  gap +0.441  (9/9)")
    print("    loss     gnn 0.817  blind 0.828  gap \u22120.010  (4/9)")
    print("\n  The four-base-station work found separate models decisively")
    print("  better: delay from indistinguishable-from-noise to +0.48, jitter")
    print("  from +0.281 to +0.54. If that replicates here, the shared encoder")
    print("  costs accuracy and its use needs the counterfactual justification.")
    print("  If it does not, gradient competition is dataset-dependent.")
    print("\n  Compare the GAP, not the absolute score: a single-task model may")
    print("  score higher simply because its problem is easier.")


if __name__ == "__main__":
    main()
