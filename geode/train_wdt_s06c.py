"""
train_wdt_s06c.py -- train and evaluate WirelessDT-6G on scenario06c.

    python3 train_wdt_s06c.py                      # full model, rotate all 3
    python3 train_wdt_s06c.py --no-e2              # drop interference edges
    python3 train_wdt_s06c.py --no-moe             # plain MLP heads
    python3 train_wdt_s06c.py --rounds 4           # shallower encoder
    python3 train_wdt_s06c.py --test-seed 62       # one rotation only
    python3 train_wdt_s06c.py --baselines-only     # exclude sleep runs

Rotation, not a single split
----------------------------
With three topologies a single held-out test set is one sample and gives
no way to separate a real result from a lucky split. Every topology is
held out in turn and the spread is reported. That triples runtime and is
worth it.

What counts as the result
-------------------------
The gap between the graph model and the graph-blind control, not the
absolute R^2. Both see identical features; the only difference is
message passing. A high R^2 that the control also achieves says the
features were sufficient and the graph added nothing.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from wdt_data import (load_runs, SnapshotDataset, Normalizer,
                      LabelNormalizer, LogLabelNormalizer, compute_bin_edges,
                      GNB_COLS, KPI_COLS, add_lag_features,
                      LAG_GNB_COLS, LAG_UE_COLS)

# Starved variant: the gNB keeps only configuration and geometry. The two
# columns removed -- offered_ul_bps and num_connected_ues -- are already
# the sum over the UEs attached to that cell, and energy is essentially a
# function of how much traffic a cell carried. Handing the model that sum
# means a graph-blind network can read the answer off directly, which is
# why energy showed no graph gap. Starved, load can only be recovered by
# following the serving edges and aggregating the UE nodes, which is also
# exactly the situation C4 faces: a counterfactual sleep configuration has
# no measured aggregate, because those UEs never attached to those cells.
STARVED_GNB_COLS = ["pos_x", "pos_y", "tx_power_dbm", "num_bands"]
from wdt_model import WirelessDT6G, WirelessDT6GBlind, count_params
from c1_graph import edge_types_present

BASE = os.environ.get("GEODE_S06C", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "scenario06c"))
SEEDS = (61, 62, 63)
POWERS = ("p37", "p43", "p46")
SLEEP = {61: [0, 1, 2, 3, 4, 5], 62: [0, 1, 2, 5], 63: [0, 1, 2, 3, 4, 5]}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_paths_for(seeds, baselines_only=False):
    paths = {}
    for s in seeds:
        for p in POWERS:
            paths[f"s{s}_{p}"] = os.path.join(BASE, f"extracted_s06c{s}_{p}")
        if not baselines_only:
            for k in SLEEP[s]:
                paths[f"s{s}_p43_sleep{k}"] = os.path.join(
                    BASE, f"extracted_s06c{s}_p43_sleep{k}")
    return paths


def metrics(y_true, y_pred, denorm=None):
    """R2, MAE, RMSE, MAPE.

    denorm : the LabelNormalizer this target was scaled with, or None.

    Why this matters: training runs on z-scored labels, so y_true here is
    in standard deviations, not watts or milliseconds. R2 is scale-free and
    unaffected, but MAE in standard deviations is not interpretable and
    MAPE is actively wrong -- standardised values cross zero, so dividing
    by them produces arbitrarily large numbers. Both are converted back to
    physical units before being computed. The evaluation targets are stated
    in physical units (energy MAPE under 10%, delay MAPE against M3Net's
    17.39%), so they cannot be checked any other way.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    def _r2(a, b):
        ss_res = ((a - b) ** 2).sum()
        ss_tot = ((a - a.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # R2 in the space the model was trained in. Under plain z-scoring this
    # equals the physical-space R2, because R2 is invariant to an affine
    # transform. Under a log transform it does NOT, so both are reported:
    # R2 here, R2_phys below. Only R2_phys is comparable across the two
    # label treatments.
    r2 = _r2(y_true, y_pred)

    if denorm is not None:
        y_true = np.asarray(denorm.inverse(y_true), dtype=np.float64)
        y_pred = np.asarray(denorm.inverse(y_pred), dtype=np.float64)
    r2_phys = _r2(y_true, y_pred) if denorm is not None else r2

    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    # MAPE only over samples with a meaningful denominator.
    nz = np.abs(y_true) > 1e-6
    mape = (np.abs((y_true[nz] - y_pred[nz]) / y_true[nz]).mean() * 100
            if nz.any() else float("nan"))
    return {"R2": float(r2), "R2_phys": float(r2_phys),
            "MAE": float(mae), "RMSE": float(rmse),
            "MAPE": float(mape), "units": denorm is not None}


def auroc(y_true, score):
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


def run_epoch(model, loader, opt=None, w=(0.35, 0.30, 0.20, 0.15),
              norms=None, mask_filled=False):
    train = opt is not None
    model.train() if train else model.eval()
    tot = 0.0
    acc = {k: ([], []) for k in ("energy", "delay", "jitter", "pl")}
    # Severity stage: collected only over rows where loss actually occurred,
    # since the bin classifier is trained on those rows alone. Kept separate
    # from the binary stage because the two answer different questions --
    # "does loss occur" and "given that it did, how bad".
    bins_true, bins_pred = [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            ea = {}
            for rel in batch.edge_types:
                store = batch[rel]
                if hasattr(store, "edge_attr") and store.edge_attr is not None:
                    ea[rel] = store.edge_attr
            out = model(batch.x_dict, batch.edge_index_dict, ea)

            e_true = batch["gnb"].y
            k_true = batch["ue"].y
            pl_bin = batch["ue"].pl_binary
            pl_cls = batch["ue"].pl_class

            l_e = F.huber_loss(out["energy"], e_true)
            # Forward-filled labels carry n_packets == 0. Fitting to them
            # teaches the model to reproduce a neighbouring window's value
            # rather than anything measured, so they can be masked out. The
            # rows stay in the graph either way -- only their contribution
            # to the delay/jitter loss is dropped.
            if mask_filled and hasattr(batch["ue"], "n_packets"):
                keep = batch["ue"].n_packets != 0
                if keep.any():
                    l_d = F.mse_loss(out["delay"].squeeze(-1)[keep],
                                     k_true[keep, 0])
                    l_j = F.mse_loss(out["jitter"].squeeze(-1)[keep],
                                     k_true[keep, 1])
                else:
                    l_d = l_j = torch.zeros((), device=DEVICE)
            else:
                l_d = F.mse_loss(out["delay"].squeeze(-1), k_true[:, 0])
                l_j = F.mse_loss(out["jitter"].squeeze(-1), k_true[:, 1])
            l_p_binary = F.binary_cross_entropy_with_logits(
                out["pl_binary_logit"], pl_bin)
            l_p = l_p_binary
            nzm = pl_bin > 0
            if nzm.any():
                l_p = l_p + F.cross_entropy(
                    out["pl_bin_logits"][nzm], pl_cls[nzm],
                    label_smoothing=0.1)
            loss = w[0]*l_e + w[1]*l_d + w[2]*l_j + w[3]*l_p

            # Selection loss EXCLUDES the packet-loss severity term.
            # That term is a cross-entropy over the non-zero rows only --
            # about 0.3% of samples, so a handful per batch. When the model
            # is confidently wrong on two of them it spikes tenfold or more
            # and dominates the combined number despite carrying the
            # smallest weight, which in scenario07 pinned checkpoint
            # selection near epoch 1 while every individual metric was
            # still improving. The bins head still trains normally through
            # `loss`; it simply does not get to veto which checkpoint is
            # kept.
            sel = w[0]*l_e + w[1]*l_d + w[2]*l_j + w[3]*l_p_binary

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        tot += float(sel) * e_true.shape[0]      # selection uses `sel`
        acc["energy"][0].append(e_true.detach().cpu().numpy())
        acc["energy"][1].append(out["energy"].detach().cpu().numpy())
        acc["delay"][0].append(k_true[:, 0].detach().cpu().numpy())
        acc["delay"][1].append(out["delay"].squeeze(-1).detach().cpu().numpy())
        acc["jitter"][0].append(k_true[:, 1].detach().cpu().numpy())
        acc["jitter"][1].append(out["jitter"].squeeze(-1).detach().cpu().numpy())
        acc["pl"][0].append(pl_bin.detach().cpu().numpy())
        acc["pl"][1].append(torch.sigmoid(
            out["pl_binary_logit"]).detach().cpu().numpy())
        if nzm.any():
            bins_true.append(pl_cls[nzm].detach().cpu().numpy())
            bins_pred.append(out["pl_bin_logits"][nzm]
                             .argmax(-1).detach().cpu().numpy())

    res = {}
    dn = norms or {}
    for k in ("energy", "delay", "jitter"):
        res[k] = metrics(np.concatenate(acc[k][0]), np.concatenate(acc[k][1]),
                         denorm=dn.get(k))
    pl_true = np.concatenate(acc["pl"][0])
    pl_score = np.concatenate(acc["pl"][1])
    plm = {"AUROC": auroc(pl_true, pl_score),
           "n_positive": int((pl_true == 1).sum()),
           "positive_rate": float((pl_true == 1).mean() * 100)}

    if bins_true:
        bt = np.concatenate(bins_true)
        bp = np.concatenate(bins_pred)
        plm["bin_acc"] = float((bt == bp).mean() * 100)
        plm["bin_n"] = int(len(bt))
        # Macro F1 over the classes present. Plain accuracy can look
        # respectable by always predicting the largest bin, a real risk at
        # this support level, so the two are reported together.
        f1s = []
        for cls in np.unique(bt):
            tp = int(((bp == cls) & (bt == cls)).sum())
            fp = int(((bp == cls) & (bt != cls)).sum())
            fn = int(((bp != cls) & (bt == cls)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        plm["bin_f1"] = float(np.mean(f1s) * 100)
        # The score a model gets for always guessing the commonest bin.
        # bin_acc must beat this to mean anything at all.
        _, counts = np.unique(bt, return_counts=True)
        plm["bin_majority"] = float(counts.max() / len(bt) * 100)
    else:
        plm.update(bin_acc=float("nan"), bin_f1=float("nan"),
                   bin_n=0, bin_majority=float("nan"))

    res["packet_loss"] = plm
    return tot / max(len(loader.dataset), 1), res


PHASE = 30          # snapshots per traffic phase; KPI labels are constant
                    # within a phase, so any temporal cut must land on a
                    # phase boundary or the same label appears on both
                    # sides of the split.


def subsample(snaps, stride, phase=PHASE):
    """Thin a run's snapshots to break KPI label duplication.

    Every (UE, phase) pair has ONE measured delay/jitter/loss value,
    broadcast across all 30 snapshots of that phase. At stride 1 the model
    therefore sees each unique label 30 times per epoch, so one epoch is
    effectively 30 passes over the unique data -- which is why every model
    in the first full run early-stopped at epoch 1-4. It had already
    converged and begun overfitting before the first epoch ended.

    At stride 30 one snapshot per phase is kept, and epochs mean what they
    normally mean.

    Which snapshot: the middle of the phase. The label is the phase's mean
    delay, so the mid-phase network state is the input that corresponds to
    it most closely; the first and last snapshots sit next to traffic
    transitions.

    What is lost: inputs (positions, SINR, offered load) do vary within a
    phase even though the label does not, so thinning discards real input
    diversity. That is a genuine trade, not a free win -- the duplicated
    rows were acting as augmentation with a fixed target. stride 1
    reproduces the original behaviour if the comparison is wanted.
    """
    snaps = np.asarray(snaps)
    if stride <= 1:
        return snaps
    offset = min(stride // 2, phase // 2)
    return snaps[offset::stride]


def one_rotation(test_seed, args):
    """Train on both non-test topologies; validate on a phase-aligned
    temporal tail carved from within them; test on the held-out topology.

    Why not one topology for training and one for validation
    --------------------------------------------------------
    Three topologies split 1/1/1 leaves a single network to learn from,
    and the scenario07 results this is compared against used five. Using
    both non-test topologies doubles the training signal at the cost of a
    validation set drawn from seen networks rather than an unseen one.
    Validation is only used for early stopping and checkpoint selection,
    never reported, so the weaker guarantee is acceptable there. The test
    topology stays completely unseen either way, which is the claim that
    matters.

    Why the cut is phase-aligned
    ----------------------------
    Each (UE, phase) has one measured KPI value broadcast across 30
    snapshots. A cut inside a phase block puts identical labels in train
    and validation, which is the leak found and fixed in scenario07.
    Cutting on multiples of PHASE avoids it by construction.
    """
    train_seeds = [s for s in SEEDS if s != test_seed]

    data = load_runs(run_paths_for(SEEDS, args.baselines_only))
    ue_cols = data["ue_cols"]
    gnb_cols = STARVED_GNB_COLS if args.starve_gnb else GNB_COLS

    lag_drop_ids = set()
    if args.use_lag or args.use_lag_energy_only:
        lag_drop_ids = add_lag_features(data)
        gnb_cols = gnb_cols + LAG_GNB_COLS
        if args.use_lag:
            ue_cols = ue_cols + LAG_UE_COLS
        # --use-lag-energy-only: ue_cols deliberately left untouched, even
        # though add_lag_features() also computed lag_delay_ms/lag_jitter_ms
        # on data["ue"] -- those columns simply aren't in ue_cols, so
        # Normalizer/SnapshotDataset never select them. gnb still needs its
        # run-start snapshots dropped the same way (lag_drop_ids covers
        # both node types' run starts, which coincide -- see
        # add_lag_features' docstring/assert).

    tag_snaps = (data["gnb"][["run_tag", "snapshot_id"]]
                 .drop_duplicates().groupby("run_tag").snapshot_id
                 .apply(lambda s: np.sort(s.unique())).to_dict())

    def tags_of(seed):
        return [t for t in tag_snaps if t.startswith(f"s{seed}_")]

    tr_ids, va_ids = [], []
    for s in train_seeds:
        for t in tags_of(s):
            snaps = tag_snaps[t]
            n_phase = len(snaps) // PHASE
            n_val = max(int(round(n_phase * args.val_frac)), 1)
            cut = (n_phase - n_val) * PHASE          # phase-aligned
            tr_ids.extend(subsample(snaps[:cut], args.stride).tolist())
            va_ids.extend(subsample(snaps[cut:], args.stride).tolist())

    te_ids = []
    for t in tags_of(test_seed):
        te_ids.extend(subsample(tag_snaps[t], args.stride).tolist())

    if lag_drop_ids:
        tr_ids = [i for i in tr_ids if i not in lag_drop_ids]
        va_ids = [i for i in va_ids if i not in lag_drop_ids]
        te_ids = [i for i in te_ids if i not in lag_drop_ids]

    tr_ids, va_ids, te_ids = sorted(tr_ids), sorted(va_ids), sorted(te_ids)

    gtr = data["gnb"][data["gnb"].snapshot_id.isin(tr_ids)]
    utr = data["ue"][data["ue"].snapshot_id.isin(tr_ids)]
    gnb_norm = Normalizer(gnb_cols).fit(gtr)
    ue_norm = Normalizer(ue_cols).fit(utr)

    etr = data["energy"][data["energy"].snapshot_id.isin(tr_ids)]
    ktr = data["kpi"][data["kpi"].snapshot_id.isin(tr_ids)]
    energy_norm = LabelNormalizer().fit(etr.estimated_gnb_power_w.to_numpy())
    KpiNorm = LogLabelNormalizer if args.log_kpi else LabelNormalizer
    kpi_norms = {"delay": KpiNorm().fit(ktr.delay_ms.to_numpy()),
                 "jitter": KpiNorm().fit(ktr.jitter_ms.to_numpy())}
    pl = ktr.packet_loss.to_numpy()
    edges = compute_bin_edges(pl[pl > 0], num_bins=5)
    n_bins = max(len(edges) - 1, 2)

    flags = dict(use_e1=not args.no_e1, use_e2=not args.no_e2,
                 use_e3=not args.no_e3, use_e4=not args.no_e4)
    mk = lambda ids: SnapshotDataset(data, ids, gnb_norm, ue_norm,
                                     energy_norm, kpi_norms, edges, **flags)
    tr_ds, va_ds, te_ds = mk(tr_ids), mk(va_ids), mk(te_ids)

    rels = edge_types_present(**flags)
    in_dims = {"gnb": len(gnb_cols), "ue": len(ue_cols)}
    if flags["use_e4"]:
        in_dims["core"] = len(gnb_cols)

    print(f"\n=== test seed {test_seed} (unseen) | train+val seeds {train_seeds} ===")
    print(f"  snapshots  train {len(tr_ds)}  val {len(va_ds)}  test {len(te_ds)}"
          f"   (val = last {args.val_frac:.0%} of each training run, phase-aligned)")
    print(f"  relations  {[f'{a}-{b}->{c}' for a,b,c in rels]}")
    print(f"  MoE {not args.no_moe} | GRU {not args.no_gru} | "
          f"attn {not args.no_attn} | rounds {args.rounds} | hidden {args.hidden}"
          f" | stride {args.stride} | gnb feats {len(gnb_cols)}"
          f"{' (STARVED)' if args.starve_gnb else ''}"
          f"{f' | use_lag (dropped {len(lag_drop_ids)} run-start snapshots)' if args.use_lag else ''}"
          f"{f' | use_lag_energy_only (dropped {len(lag_drop_ids)} run-start snapshots)' if args.use_lag_energy_only else ''}")

    per_init = {"gnn": [], "blind": []}
    for init in args.inits:
        for kind, Cls in (("gnn", WirelessDT6G), ("blind", WirelessDT6GBlind)):
            if kind == "blind" and args.no_baseline:
                continue
            torch.manual_seed(init)
            model = Cls(in_dims, rels, hidden=args.hidden, rounds=args.rounds,
                        num_pl_bins=n_bins, use_moe=not args.no_moe,
                        dropout=args.dropout,
                        use_gru=not args.no_gru,
                        attention_on=(None if args.no_attn
                                      else ("gnb", "interferes", "gnb")),
                        deep_energy=args.deep_energy,
                        ).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
            trl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
            val = DataLoader(va_ds, batch_size=args.batch)
            tel = DataLoader(te_ds, batch_size=args.batch)

            # Normalisers passed so reported MAE/RMSE/MAPE are in watts and
            # milliseconds rather than standard deviations.
            metric_norms = {"energy": energy_norm,
                            "delay": kpi_norms["delay"],
                            "jitter": kpi_norms["jitter"]}
            best, best_state, best_ep, bad = float("inf"), None, 0, 0
            for ep in range(1, args.epochs + 1):
                run_epoch(model, trl, opt, norms=metric_norms,
                          mask_filled=args.mask_filled)
                vl, _ = run_epoch(model, val, norms=metric_norms)
                if vl < best - 1e-5:
                    best, best_ep, bad = vl, ep, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            model.load_state_dict(best_state)
            _, m = run_epoch(model, tel, norms=metric_norms)
            m["_best_epoch"] = best_ep
            m["_params"] = count_params(model)
            per_init[kind].append(m)
            print(f"  [{kind:5}] init {init}  ep {best_ep:>3}  "
                  f"E {m['energy']['R2']:.3f}  D {m['delay']['R2']:.3f}  "
                  f"J {m['jitter']['R2']:.3f}  PL {m['packet_loss']['AUROC']:.3f}"
                  f" (bin {m['packet_loss']['bin_acc']:.0f}%/maj "
                  f"{m['packet_loss']['bin_majority']:.0f}%)"
                  f"  |  Dphys {m['delay']['R2_phys']:.3f}"
                  f"  E {m['energy']['MAPE']:.1f}% {m['energy']['MAE']:.1f}W"
                  f"  D {m['delay']['MAPE']:.1f}% {m['delay']['MAE']:.1f}ms")
            if args.save and (args.save_all or init == args.inits[0]):
                os.makedirs(args.save, exist_ok=True)
                # Normalisers are part of the checkpoint, not an
                # afterthought: without them the weights cannot be applied
                # to new data at all, which is exactly what C4 needs to do
                # when it evaluates a counterfactual graph.
                ck = {
                    "state_dict": best_state,
                    "kind": kind,
                    "in_dims": in_dims,
                    "relations": rels,
                    "flags": flags,
                    "hidden": args.hidden,
                    "rounds": args.rounds,
                    "use_moe": not args.no_moe,
                    "deep_energy": args.deep_energy,
                    "dropout": args.dropout,
                    "n_bins": n_bins,
                    "test_seed": test_seed,
                    "train_seeds": train_seeds,
                    "init": init,
                    "best_epoch": best_ep,
                    "test_metrics": m,
                    # everything needed to reproduce the input pipeline
                    "gnb_cols": gnb_cols,
                    "ue_cols": ue_cols,
                    "gnb_norm": {"mean": gnb_norm.mean_, "std": gnb_norm.std_},
                    "ue_norm": {"mean": ue_norm.mean_, "std": ue_norm.std_},
                    "energy_norm": {"mean": energy_norm.mean_,
                                    "std": energy_norm.std_},
                    "kpi_norm": {k: {"mean": v.mean_, "std": v.std_}
                                 for k, v in kpi_norms.items()},
                    "pl_bin_edges": edges,
                }
                fname = f"wdt_{args.tag}_{kind}_test{test_seed}_init{init}.pt"
                torch.save(ck, os.path.join(args.save, fname))
    return per_init


def summarise(all_rot, args):
    print("\n" + "=" * 78)
    print("WirelessDT-6G on scenario06c -- mean over rotations and inits")
    print("=" * 78)
    keys = [("energy", "R2"), ("delay", "R2"), ("jitter", "R2"),
            ("packet_loss", "AUROC")]
    for tgt, met in keys:
        g = [m[tgt][met] for rot in all_rot.values() for m in rot["gnn"]]
        line = f"{tgt:<12} {met}  GNN {np.mean(g):.4f} +/- {np.std(g):.4f}"
        if not args.no_baseline:
            b = [m[tgt][met] for rot in all_rot.values() for m in rot["blind"]]
            gaps = np.array(g) - np.array(b)
            line += (f"   blind {np.mean(b):.4f} +/- {np.std(b):.4f}"
                     f"   gap {gaps.mean():+.4f} "
                     f"({int((gaps > 0).sum())}/{len(gaps)} positive)")
        print(line)
    if args.log_kpi:
        print("\nR2 in physical (ms) space -- the figure comparable with "
              "non-log runs:")
        for tgt in ("delay", "jitter"):
            v = [m[tgt]["R2_phys"] for rot in all_rot.values() for m in rot["gnn"]]
            b = [m[tgt]["R2_phys"] for rot in all_rot.values() for m in rot["blind"]] \
                if not args.no_baseline else None
            line = f"  {tgt:<8} GNN {np.mean(v):.4f}"
            if b:
                gaps = np.array(v) - np.array(b)
                line += (f"   blind {np.mean(b):.4f}   gap {gaps.mean():+.4f} "
                         f"({int((gaps>0).sum())}/{len(gaps)} positive)")
            print(line)

    print("\nPhysical units (graph model), against the evaluation plan targets:")
    for tgt, unit, target in (("energy", "W", "MAPE < 10%"),
                              ("delay", "ms", "MAPE < 17.39% (M3Net)"),
                              ("jitter", "ms", "\u2014")):
        mape = [m[tgt]["MAPE"] for rot in all_rot.values() for m in rot["gnn"]]
        mae = [m[tgt]["MAE"] for rot in all_rot.values() for m in rot["gnn"]]
        rmse = [m[tgt]["RMSE"] for rot in all_rot.values() for m in rot["gnn"]]
        print(f"  {tgt:<8} MAPE {np.mean(mape):>8.2f}%   "
              f"MAE {np.mean(mae):>8.2f} {unit:<3} "
              f"RMSE {np.mean(rmse):>8.2f} {unit:<3}  target: {target}")

    print("\nPacket-loss severity stage (conditional on loss occurring):")
    for kind in (("gnn", "blind") if not args.no_baseline else ("gnn",)):
        a = [m["packet_loss"]["bin_acc"] for rot in all_rot.values() for m in rot[kind]]
        f = [m["packet_loss"]["bin_f1"] for rot in all_rot.values() for m in rot[kind]]
        mj = [m["packet_loss"]["bin_majority"] for rot in all_rot.values() for m in rot[kind]]
        n = [m["packet_loss"]["bin_n"] for rot in all_rot.values() for m in rot[kind]]
        print(f"  {kind:<6} accuracy {np.nanmean(a):>5.1f}%   macro-F1 "
              f"{np.nanmean(f):>5.1f}%   majority-class {np.nanmean(mj):>5.1f}%"
              f"   n={int(np.mean(n))} lossy rows")
    print("  Accuracy at or below the majority-class rate means the severity")
    print("  head is not discriminating -- it would score as well by always")
    print("  predicting the commonest bin.")

    print("\nThe gap is the result. Both models see identical features;")
    print("only message passing differs.")


# ---------------------------------------------------------------------------
# Ablation sweep
# ---------------------------------------------------------------------------
# Each entry changes exactly one thing from the full configuration, so a
# difference in the result is attributable to that one component. "starved"
# is the exception and is not a component ablation at all: it changes the
# INPUTS rather than the architecture, and answers a different question --
# whether the graph can reconstruct per-cell load when it is not handed the
# aggregate. Read it separately from the others.
# Two of these entries are NOT architecture ablations and must not be read
# in the same column as the rest:
#   stride*   -- changes how the 30x KPI label duplication is handled. The
#                simulator measures delay once per 3 s traffic phase while
#                network state is sampled every 100 ms, so one label covers
#                30 snapshots. This is a data-resolution question.
#   starved   -- changes the gNB input features, not the model.
SWEEP = [
    ("full",     {}),                    # stride 30, one snapshot per phase
    ("stride15", {"stride": 15}),        # 2x duplication retained
    ("stride1",  {"stride": 1}),         # full 30x duplication (original)
    ("no_e2",    {"no_e2": True}),      # interference edges
    ("no_e3",    {"no_e3": True}),      # handover edges
    ("no_e4",    {"no_e4": True}),      # core aggregation node
    ("no_moe",   {"no_moe": True}),     # MoE heads -> plain MLP
    ("no_gru",   {"no_gru": True}),      # recurrent update -> overwrite
    ("no_attn",  {"no_attn": True}),     # E2 kept, attention removed
    ("rounds4",  {"rounds": 4}),        # T=10 -> 4
    ("starved",  {"starve_gnb": True}),
]


def collect(all_rot, args):
    """Mean and spread per target across rotations and inits."""
    out = {}
    for tgt, met in (("energy", "R2"), ("delay", "R2"),
                     ("jitter", "R2"), ("packet_loss", "AUROC")):
        g = [m[tgt][met] for rot in all_rot.values() for m in rot["gnn"]]
        row = {"gnn": float(np.mean(g)), "gnn_sd": float(np.std(g))}
        if not args.no_baseline and all_rot and all_rot[list(all_rot)[0]]["blind"]:
            b = [m[tgt][met] for rot in all_rot.values() for m in rot["blind"]]
            gaps = np.array(g) - np.array(b)
            row.update(blind=float(np.mean(b)), blind_sd=float(np.std(b)),
                       gap=float(gaps.mean()),
                       pos=int((gaps > 0).sum()), n=int(len(gaps)))
        out[tgt] = row
    return out


def sweep(args):
    import copy
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
        rot_seeds = [args.test_seed] if args.test_seed else list(SEEDS)
        all_rot = {s: one_rotation(s, a) for s in rot_seeds}
        summarise(all_rot, a)
        results[name] = collect(all_rot, a)
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            with open(os.path.join(args.save, f"results_{name}.json"), "w") as f:
                json.dump({str(k): v for k, v in all_rot.items()}, f, indent=2)

    # ---- comparison table -------------------------------------------------
    print("\n" + "=" * 96)
    print("ABLATION SUMMARY -- graph-vs-blind gap per configuration")
    print("=" * 96)
    hdr = f"{'config':<10}"
    for t in ("energy", "delay", "jitter", "packet_loss"):
        hdr += f"{t[:9]:>21}"
    print(hdr)
    print(f"{'':<10}" + f"{'GNN    gap  pos':>21}" * 4)
    print("-" * 96)
    for name in results:
        line = f"{name:<10}"
        for t in ("energy", "delay", "jitter", "packet_loss"):
            r = results[name][t]
            if "gap" in r:
                line += f"{r['gnn']:>8.3f} {r['gap']:>+7.3f} {r['pos']:>2}/{r['n']:<2}"
            else:
                line += f"{r['gnn']:>8.3f} {'':>12}"
        print(line)

    if "full" in results:
        print("\nChange in GNN score relative to 'full' (negative = component helped):")
        for name in results:
            if name == "full":
                continue
            d = [results[name][t]["gnn"] - results["full"][t]["gnn"]
                 for t in ("energy", "delay", "jitter", "packet_loss")]
            print(f"  {name:<10} E {d[0]:+.3f}  D {d[1]:+.3f}  "
                  f"J {d[2]:+.3f}  PL {d[3]:+.3f}")
        print("\nNot architecture ablations -- read separately:")
        print("  starved            changes the gNB inputs; its gap is the number of interest")
        print("  stride15/stride1   change label duplication, so the dataset size differs")
        print("                     (stride 30 -> ~612 train graphs, 15 -> ~1224, 1 -> ~18360)")

    if args.save:
        with open(os.path.join(args.save, "sweep_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsweep summary -> {args.save}/sweep_summary.json")


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
    ap.add_argument("--test-seed", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of each training run held out for "
                         "validation, rounded to whole 30-snapshot phases")
    ap.add_argument("--kpi-1s", action="store_true",
                    help="use kpi_targets_1s.csv (1 s windows, ~6700 labels "
                         "per run) instead of kpi_targets.csv (3 s phases, "
                         "2400). Sets --stride to 10 unless given explicitly, "
                         "since 1 s = 10 snapshots.")
    ap.add_argument("--log-kpi", action="store_true",
                    help="fit delay and jitter on log1p(value) instead of the "
                         "raw value. Intended for --kpi-1s, where the target "
                         "spans two orders of magnitude and MSE chases a few "
                         "congested windows. R2 is then reported in both the "
                         "training space and physical space; only R2_phys is "
                         "comparable with non-log runs.")
    ap.add_argument("--mask-filled", action="store_true",
                    help="exclude forward-filled KPI rows (n_packets == 0) "
                         "from the delay/jitter loss. Only meaningful with "
                         "--kpi-1s, where imputation reaches 5-8%%.")
    ap.add_argument("--stride", type=int, default=None,
                    help="keep 1 snapshot per STRIDE. 30 = one per phase, "
                         "which removes the 30x KPI label duplication. "
                         "1 reproduces the original behaviour. Defaults to "
                         "30, or 10 with --kpi-1s.")
    ap.add_argument("--no-e1", action="store_true")
    ap.add_argument("--no-e2", action="store_true")
    ap.add_argument("--no-e3", action="store_true")
    ap.add_argument("--no-e4", action="store_true")
    ap.add_argument("--no-moe", action="store_true")
    ap.add_argument("--deep-energy", action="store_true",
                    help="give the energy head a wider/deeper plain MLP "
                         "(2x hidden width, 3 layers) instead of MoE/plain, "
                         "keeping the shared encoder untouched -- tests "
                         "whether energy's head is capacity-starved by the "
                         "joint 4-target objective (see RF-beats-gnn-on-"
                         "energy finding in src/v7_5g3e/SWEEP_RESULTS.md)")
    ap.add_argument("--no-gru", action="store_true",
                    help="replace the recurrent update with a plain "
                         "overwrite, making the encoder a weight-tied "
                         "stack of convolutions")
    ap.add_argument("--no-attn", action="store_true",
                    help="keep the interference edges but drop GATv2 "
                         "attention over them (all relations use SAGEConv)")
    ap.add_argument("--starve-gnb", action="store_true",
                    help="drop offered_ul_bps and num_connected_ues from the "
                         "gNB features; load must then be aggregated from UEs")
    ap.add_argument("--use-lag", action="store_true",
                    help="append each node's own previous-snapshot label(s) "
                         "to its input features (gnb: own prev energy_W; "
                         "ue: own prev [delay_ms, jitter_ms]) -- the fair "
                         "gnn/blind counterpart to baseline_classical.py's "
                         "RF+lag diagnostic. CAUTION: 06c's delay/jitter/PL "
                         "labels are broadcast across each 3s phase's 30 "
                         "snapshots (see wdt_data.py's KPI_FILE comment), so "
                         "lag_delay_ms/lag_jitter_ms are ~equal to the "
                         "current label for 96.7% of rows -- near-total "
                         "leakage, not a real temporal feature, for THOSE "
                         "two targets specifically. energy_targets.csv has "
                         "no such broadcast (7.9% exact-repeat rate, i.e. "
                         "genuinely fresh each snapshot), so lag_energy_w is "
                         "legitimate. Use --use-lag-energy-only for a run "
                         "with no leakage pathway anywhere in the graph. "
                         "Each run's first snapshot has no valid lag and is "
                         "dropped from train/val/test rather than faked.")
    ap.add_argument("--use-lag-energy-only", action="store_true",
                    help="like --use-lag but gnb only (own prev energy_W); "
                         "ue gets no lag feature at all. Use this whenever "
                         "the UE-side leakage described under --use-lag "
                         "must not be allowed to reach ANY prediction, "
                         "including energy via ue->gnb message passing in "
                         "the gnn (not an issue for blind, which never "
                         "aggregates across nodes, but gnn does).")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--save", type=str, default="checkpoints")
    ap.add_argument("--save-all", action="store_true",
                    help="save every init, not just the first")
    ap.add_argument("--sweep", action="store_true",
                    help="run every ablation config in turn and compare")
    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="with --sweep, restrict to these config names")
    ap.add_argument("--tag", type=str, default="full")
    args = ap.parse_args()

    # One snapshot per label window: 3 s phases -> 30, 1 s windows -> 10.
    # Set before anything reads the data, because wdt_data picks the file
    # up from the environment at import-time of load_runs.
    if args.stride is None:
        args.stride = 10 if args.kpi_1s else 30
    if args.kpi_1s:
        os.environ["WDT_KPI_FILE"] = "kpi_targets_1s.csv"
        import wdt_data
        wdt_data.KPI_FILE = "kpi_targets_1s.csv"
        print(f"using 1 s KPI labels, stride {args.stride}"
              + ("  (filled rows masked from KPI loss)" if args.mask_filled else ""))

    if args.sweep:
        sweep(args)
        return

    rot_seeds = [args.test_seed] if args.test_seed else list(SEEDS)
    all_rot = {s: one_rotation(s, args) for s in rot_seeds}
    summarise(all_rot, args)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, f"results_{args.tag}.json"), "w") as f:
            json.dump({str(k): v for k, v in all_rot.items()}, f, indent=2)
        print(f"\nresults -> {args.save}/results_{args.tag}.json")


if __name__ == "__main__":
    main()