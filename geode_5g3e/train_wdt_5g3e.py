"""
train_wdt_5g3e.py -- train and evaluate WirelessDT-6G on the real 5G3E
testbed (Day 1 train+val, Day 2 held-out test).

    python3 train_wdt_5g3e.py                    # full model
    python3 train_wdt_5g3e.py --no-e4            # drop the backhaul edge
    python3 train_wdt_5g3e.py --no-baseline       # skip the blind control
    python3 train_wdt_5g3e.py --sweep             # run every ablation config

Why there is no seed rotation here
-----------------------------------
src/v6's methodology holds out each of 3 simulator seeds in turn and
reports the spread, because a single split is one sample and a lucky
split cannot be told apart from a real result. 5G3E has exactly two
days, not interchangeable seeds -- Day 2 is the only honest held-out
set, so there is nothing to rotate. What *does* carry over is the
multi-init loop: report mean +/- std over --inits on the one split that
exists, instead of the single deterministic run src/v3/train_v3.py did.

What counts as the result
--------------------------
Same principle as v6: the gap between the graph model and the
graph-blind control, not the absolute R^2. For energy specifically, see
wdt_model.py's docstring for why that gap is expected to be LARGER here
than on scenario06c, not a repeat of it -- cpu_util never reaches a
blind gNB at all.

Packet loss is reported as N/A, not scored
--------------------------------------------
packet_loss is a measured constant (0.0) across the entire 5G3E
dataset -- see c3_heads.py's docstring. auroc() already returns NaN
when there are zero positive examples, which is the correct behaviour
here: there is no AUROC for a label with one class. Do not read a
number out of this field.
"""

import argparse
import copy
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from wdt_data import (load_norm_stats, load_split, SnapshotDataset,
                      relations_for, in_dims_for, Y_COLS)
from wdt_model import WirelessDT6G, WirelessDT6GBlind, count_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SWEEP = [
    ("full",     {}),
    ("no_e1",    {"no_e1": True}),      # serving edges (round-robin, not measured)
    ("no_e2",    {"no_e2": True}),      # interference edges (intra-site)
    ("no_e3",    {"no_e3": True}),      # handover edges (cross-site)
    ("no_e4",    {"no_e4": True}),      # backhaul edge -- energy's only route
    ("no_moe",   {"no_moe": True}),
    ("no_gru",   {"no_gru": True}),
    ("no_attn",  {"no_attn": True}),
    ("rounds4",  {"rounds": 4}),
]


def metrics(y_true, y_pred, denorm=None):
    """R2, MAE, RMSE, MAPE -- identical to src/v6/train_wdt_s06c.py.

    Training runs on z-scored labels; MAE/MAPE are only meaningful after
    converting back to physical units (Watts, ms), which is what the
    evaluation targets (energy MAPE < 10%, delay MAPE < 17.39%) are
    stated in.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    def _r2(a, b):
        ss_res = ((a - b) ** 2).sum()
        ss_tot = ((a - a.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2 = _r2(y_true, y_pred)
    if denorm is not None:
        y_true = np.asarray(denorm.inverse(y_true), dtype=np.float64)
        y_pred = np.asarray(denorm.inverse(y_pred), dtype=np.float64)
    r2_phys = _r2(y_true, y_pred) if denorm is not None else r2

    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    nz = np.abs(y_true) > 1e-6
    mape = (np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]).mean() * 100
            if nz.any() else float("nan"))
    return {"R2": float(r2), "R2_phys": float(r2_phys),
            "MAE": float(mae), "RMSE": float(rmse),
            "MAPE": float(mape), "units": denorm is not None}


def auroc(y_true, score):
    """Returns NaN when the label has only one class present -- which is
    always true for packet_loss on this dataset. That NaN IS the correct
    "N/A" signal; do not special-case it away."""
    y_true = np.asarray(y_true).ravel()
    score = np.asarray(score).ravel()
    pos, neg = y_true == 1, y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def _discretise_jitter(y_norm, bin_lo=-2.0, bin_hi=4.0, n_bins=60):
    """Same formula as BinnedHead.discretise, standalone so the w1std
    metric still works when jitter isn't a BinnedHead (predict_delta
    mode uses a plain regression head instead -- see c3_heads.py)."""
    width = (bin_hi - bin_lo) / n_bins
    idx = ((y_norm - bin_lo) / width).long()
    return idx.clamp(0, n_bins - 1)


def run_epoch(model, loader, opt=None, w=(0.35, 0.30, 0.20, 0.15)):
    train = opt is not None
    model.train() if train else model.eval()
    tot_loss, tot_n = 0.0, 0
    acc = {k: ([], []) for k in ("energy", "delay", "jitter", "pl", "jitter_bins")}
    predict_delta = model.heads.predict_delta

    for batch in loader:
        batch = batch.to(DEVICE)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            is_blind = isinstance(model, WirelessDT6GBlind)
            out = (model(batch.x_dict) if is_blind
                   else model(batch.x_dict, batch.edge_index_dict))

            y = batch["gnb"].y                 # normalised [N, 4]
            y_raw = batch["gnb"].y_raw          # physical units [N, 4]
            e_true, d_true, j_true = y[:, 0], y[:, 1], y[:, 2]
            pl_bin = (y_raw[:, 3] > 0).float()  # always 0 on this dataset

            if predict_delta:
                # Heads predict the CHANGE from the previous snapshot;
                # loss is on that delta, but everything downstream
                # (metrics, checkpoints) needs the reconstructed
                # absolute value -- lag + predicted delta.
                lag = batch["gnb"].lag_y_norm  # [N, 3] energy, delay, jitter
                l_e = F.huber_loss(out["energy"].squeeze(-1), e_true - lag[:, 0])
                l_d = F.mse_loss(out["delay"].squeeze(-1), d_true - lag[:, 1])
                l_j = F.mse_loss(out["jitter"], j_true - lag[:, 2])

                e_pred = lag[:, 0] + out["energy"].squeeze(-1)
                d_pred = lag[:, 1] + out["delay"].squeeze(-1)
                j_pred = lag[:, 2] + out["jitter"]
                j_cls_true = _discretise_jitter(j_true)
                j_cls_pred = _discretise_jitter(j_pred.detach())
            else:
                l_e = F.huber_loss(out["energy"].squeeze(-1), e_true)
                l_d = F.mse_loss(out["delay"].squeeze(-1), d_true)

                # Jitter: two-stage loss matching src/v3/train_v3.py --
                # BCE(is jitter above its own mean?) + CE(which of 60
                # bins?), the CE term over ALL rows (unlike packet_loss
                # below, jitter is a continuous quantity that's
                # essentially never exactly zero, so every row has a
                # meaningful bin to classify into).
                j_bin_head = model.heads.jitter
                j_binary_true = (j_true > 0.0).float()
                j_cls_true = j_bin_head.discretise(j_true)
                l_j = (F.binary_cross_entropy_with_logits(
                           out["jitter_binary_logit"], j_binary_true)
                       + F.cross_entropy(out["jitter_bin_logits"], j_cls_true))

                e_pred = out["energy"].squeeze(-1)
                d_pred = out["delay"].squeeze(-1)
                j_pred = out["jitter"]
                j_cls_pred = out["jitter_bin_logits"].argmax(-1)

            l_p = F.binary_cross_entropy_with_logits(out["pl_binary_logit"], pl_bin)
            nzm = pl_bin > 0
            if nzm.any():
                # Never true on 5G3E (packet_loss is a constant), kept so
                # the loss composition matches src/v6 exactly if that
                # ever changes.
                pl_cls = torch.zeros_like(pl_bin, dtype=torch.long)
                l_p = l_p + F.cross_entropy(out["pl_bin_logits"][nzm],
                                            pl_cls[nzm], label_smoothing=0.1)
            loss = w[0]*l_e + w[1]*l_d + w[2]*l_j + w[3]*l_p

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        n = e_true.shape[0]
        tot_loss += float(loss) * n
        tot_n += n
        acc["energy"][0].append(e_true.detach().cpu().numpy())
        acc["energy"][1].append(e_pred.detach().cpu().numpy())
        acc["delay"][0].append(d_true.detach().cpu().numpy())
        acc["delay"][1].append(d_pred.detach().cpu().numpy())
        acc["jitter"][0].append(j_true.detach().cpu().numpy())
        acc["jitter"][1].append(j_pred.detach().cpu().numpy())
        acc["jitter_bins"][0].append(j_cls_true.detach().cpu().numpy())
        acc["jitter_bins"][1].append(j_cls_pred.detach().cpu().numpy())
        acc["pl"][0].append(pl_bin.detach().cpu().numpy())
        acc["pl"][1].append(torch.sigmoid(out["pl_binary_logit"]).detach().cpu().numpy())

    res = {}
    for k in ("energy", "delay", "jitter"):
        res[k] = metrics(np.concatenate(acc[k][0]), np.concatenate(acc[k][1]))
    # "within +/-1 std" bin accuracy -- directly comparable to
    # src/v3/train_v3.py's jitter_w10_acc (77.2%), unlike R2/MAPE above.
    jb_true = np.concatenate(acc["jitter_bins"][0])
    jb_pred = np.concatenate(acc["jitter_bins"][1])
    res["jitter"]["w1std_acc"] = float((np.abs(jb_true - jb_pred) <= 10).mean() * 100)
    res["jitter"]["exact_bin_acc"] = float((jb_true == jb_pred).mean() * 100)
    pl_true = np.concatenate(acc["pl"][0])
    pl_score = np.concatenate(acc["pl"][1])
    res["packet_loss"] = {"AUROC": auroc(pl_true, pl_score),
                          "n_positive": int((pl_true == 1).sum()),
                          "note": "N/A -- packet_loss is a measured constant "
                                  "(0.0) on this testbed, not a real target"}
    return tot_loss / max(tot_n, 1), res


def denorm_metrics(model, loader, y_norms):
    """Re-run in eval mode and report energy/delay/jitter in physical
    units (W, ms) rather than z-scores, using the LabelNormalizers already
    fit by preprocess_v3.py."""
    model.eval()
    acc = {k: ([], []) for k in ("energy", "delay", "jitter")}
    predict_delta = model.heads.predict_delta
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            is_blind = isinstance(model, WirelessDT6GBlind)
            out = (model(batch.x_dict) if is_blind
                   else model(batch.x_dict, batch.edge_index_dict))
            y_raw = batch["gnb"].y_raw
            if predict_delta:
                lag = batch["gnb"].lag_y_norm
                e_pred = lag[:, 0] + out["energy"].squeeze(-1)
                d_pred = lag[:, 1] + out["delay"].squeeze(-1)
                j_pred = lag[:, 2] + out["jitter"]
            else:
                e_pred = out["energy"].squeeze(-1)
                d_pred = out["delay"].squeeze(-1)
                j_pred = out["jitter"]
            acc["energy"][0].append(y_raw[:, 0].cpu().numpy())
            acc["energy"][1].append(y_norms["energy_W"].inverse(
                e_pred.detach().cpu().numpy()))
            acc["delay"][0].append(y_raw[:, 1].cpu().numpy())
            acc["delay"][1].append(y_norms["delay_ms"].inverse(
                d_pred.detach().cpu().numpy()))
            acc["jitter"][0].append(y_raw[:, 2].cpu().numpy())
            acc["jitter"][1].append(y_norms["jitter_ms"].inverse(
                j_pred.detach().cpu().numpy()))
    return {k: metrics(np.concatenate(acc[k][0]), np.concatenate(acc[k][1]))
           for k in acc}


def one_run(args, norms, split):
    # predict_delta needs lag_y_norm on the graph to compute
    # lag + delta; force it on rather than silently producing a graph
    # with no lag_y_norm attribute if someone passes --predict-delta alone.
    if args.predict_delta:
        args.use_lag = True

    edge_flags = dict(use_e1=not args.no_e1, use_e2=not args.no_e2,
                      use_e3=not args.no_e3, use_e4=not args.no_e4)
    rels = relations_for(**edge_flags)
    in_dims = in_dims_for(norms, use_lag=args.use_lag)

    tr_ds = SnapshotDataset(split["train"], norms, use_lag=args.use_lag, **edge_flags)
    va_ds = SnapshotDataset(split["val"], norms, use_lag=args.use_lag, **edge_flags)
    te_ds = SnapshotDataset(split["test"], norms, use_lag=args.use_lag, **edge_flags)

    print(f"  snapshots  train {len(tr_ds)}  val {len(va_ds)}  test {len(te_ds)}")
    print(f"  relations  {[f'{a}-{b}->{c}' for a,b,c in rels]}")
    print(f"  MoE {not args.no_moe} | GRU {not args.no_gru} | "
          f"attn {not args.no_attn} | rounds {args.rounds} | hidden {args.hidden}"
          f" | lag {args.use_lag} | delta {args.predict_delta}")

    per_init = {"gnn": [], "blind": []}
    for init in args.inits:
        for kind, Cls in (("gnn", WirelessDT6G), ("blind", WirelessDT6GBlind)):
            if kind == "blind" and args.no_baseline:
                continue
            torch.manual_seed(init)
            model = Cls(in_dims, rels, hidden=args.hidden, rounds=args.rounds,
                        use_moe=not args.no_moe, dropout=args.dropout,
                        use_gru=not args.no_gru,
                        attention_on=(None if args.no_attn
                                      else ("gnb", "interferes", "gnb")),
                        predict_delta=args.predict_delta,
                        ).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
            # Same values as src/v3/train_v3.py. LR-patience (5) sits well
            # under early-stop patience (--patience, default 20) so the
            # scheduler gets several decay steps before training gives up
            # -- without that headroom it would never actually fire.
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
            trl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
            val = DataLoader(va_ds, batch_size=args.batch)
            tel = DataLoader(te_ds, batch_size=args.batch)

            best, best_state, best_ep, bad = float("inf"), None, 0, 0
            for ep in range(1, args.epochs + 1):
                run_epoch(model, trl, opt)
                vl, _ = run_epoch(model, val)
                sched.step(vl)
                if vl < best - 1e-5:
                    best, best_ep, bad = vl, ep, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            model.load_state_dict(best_state)
            _, m = run_epoch(model, tel)
            m["_phys"] = denorm_metrics(model, tel, norms["y"])
            m["_best_epoch"] = best_ep
            m["_params"] = count_params(model)
            per_init[kind].append(m)
            p = m["_phys"]
            print(f"  [{kind:5}] init {init}  ep {best_ep:>3}  "
                  f"E R2 {m['energy']['R2']:.3f} MAPE {p['energy']['MAPE']:.2f}%  "
                  f"D R2 {m['delay']['R2']:.3f} MAPE {p['delay']['MAPE']:.2f}%  "
                  f"J R2 {m['jitter']['R2']:.3f} w1std {m['jitter']['w1std_acc']:.1f}%")
            if args.save and (args.save_all or init == args.inits[0]):
                os.makedirs(args.save, exist_ok=True)
                ck = {
                    "state_dict": best_state, "kind": kind, "in_dims": in_dims,
                    "relations": rels, "edge_flags": edge_flags,
                    "hidden": args.hidden, "rounds": args.rounds,
                    "use_moe": not args.no_moe, "use_gru": not args.no_gru,
                    "use_attn": not args.no_attn, "use_lag": args.use_lag,
                    "predict_delta": args.predict_delta,
                    "dropout": args.dropout,
                    "init": init, "best_epoch": best_ep, "test_metrics": m,
                }
                fname = f"wdt5g3e_{args.tag}_{kind}_init{init}.pt"
                torch.save(ck, os.path.join(args.save, fname))
    return per_init


def summarise(per_init, args):
    print("\n" + "=" * 78)
    print("WirelessDT-6G on 5G3E -- mean over inits (Day 2 test, physical units)")
    print("=" * 78)
    for tgt in ("energy", "delay", "jitter"):
        g = [m["_phys"][tgt]["MAPE"] for m in per_init["gnn"]]
        gr2 = [m[tgt]["R2"] for m in per_init["gnn"]]
        line = f"{tgt:<8} MAPE  GNN {np.mean(g):.2f}% +/- {np.std(g):.2f}%   R2 {np.mean(gr2):.3f}"
        if not args.no_baseline and per_init["blind"]:
            b = [m["_phys"][tgt]["MAPE"] for m in per_init["blind"]]
            br2 = [m[tgt]["R2"] for m in per_init["blind"]]
            line += (f"   |   blind {np.mean(b):.2f}% +/- {np.std(b):.2f}%   "
                     f"R2 {np.mean(br2):.3f}   R2 gap {np.mean(gr2)-np.mean(br2):+.3f}")
        print(line)
    gw = [m["jitter"]["w1std_acc"] for m in per_init["gnn"]]
    w1std_line = f"jitter w1std_acc  GNN {np.mean(gw):.1f}% +/- {np.std(gw):.1f}%"
    if not args.no_baseline and per_init["blind"]:
        bw = [m["jitter"]["w1std_acc"] for m in per_init["blind"]]
        w1std_line += f"   |   blind {np.mean(bw):.1f}% +/- {np.std(bw):.1f}%"
    w1std_line += "   (v3 single-run reference: 77.2%)"
    print(w1std_line)
    print("packet_loss  N/A -- constant 0.0 across this testbed, not a real target")
    print("\nThe gnn-vs-blind gap is the result, energy especially: cpu_util can")
    print("only reach a gNB through the backhaul edge (see wdt_model.py).")


def sweep(args, norms, split):
    results = {}
    for name, overrides in SWEEP:
        if args.only and name not in args.only:
            continue
        a = copy.deepcopy(args)
        for k, v in overrides.items():
            setattr(a, k, v)
        a.tag = name
        print("\n" + "#" * 78)
        print(f"# CONFIG: {name}" + (f"   ({overrides})" if overrides else "   (reference)"))
        print("#" * 78)
        per_init = one_run(a, norms, split)
        summarise(per_init, a)
        results[name] = per_init
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inits", type=int, nargs="+", default=[1, 7, 13])
    ap.add_argument("--no-e1", action="store_true")
    ap.add_argument("--no-e2", action="store_true")
    ap.add_argument("--no-e3", action="store_true")
    ap.add_argument("--no-e4", action="store_true")
    ap.add_argument("--no-moe", action="store_true")
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--use-lag", action="store_true",
                    help="append each gNB's own previous-snapshot "
                         "[energy,delay,jitter] to its input features")
    ap.add_argument("--predict-delta", action="store_true",
                    help="heads predict the CHANGE from the previous "
                         "snapshot rather than the absolute value "
                         "(implies --use-lag; jitter switches from "
                         "BinnedHead to a plain regression head)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--save", type=str, default="checkpoints_5g3e")
    ap.add_argument("--save-all", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--only", type=str, nargs="+", default=None)
    ap.add_argument("--tag", type=str, default="full")
    args = ap.parse_args()

    norms = load_norm_stats()
    split = load_split()

    if args.sweep:
        results = sweep(args, norms, split)
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            summary = {name: {
                "gnn_energy_mape": float(np.mean([m["_phys"]["energy"]["MAPE"]
                                                  for m in pi["gnn"]])),
                "gnn_delay_mape": float(np.mean([m["_phys"]["delay"]["MAPE"]
                                                 for m in pi["gnn"]])),
            } for name, pi in results.items()}
            with open(os.path.join(args.save, "sweep_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
    else:
        per_init = one_run(args, norms, split)
        summarise(per_init, args)


if __name__ == "__main__":
    main()
