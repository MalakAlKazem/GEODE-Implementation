"""
ensemble_eval.py -- score the average of several models' predictions,
rather than averaging their scores.

    python3 ensemble_eval.py --tag final

What changes
------------
Every result so far reports the mean of three separately-scored models:
train three initialisations, compute R2 for each, average the three R2s.
That describes how good a typical model is.

Ensembling averages the PREDICTIONS first and scores once. Because the
three initialisations settle in different local optima, they tend to err in
different directions, so averaging cancels part of the error. The ensemble
is usually better than the average member, and never much worse.

Nothing is retrained. The checkpoints already exist.

What this costs
---------------
Every number recorded so far changes, so tables and text have to be
regenerated. And the twin becomes three models rather than one, which is a
deployment choice to justify rather than assume -- three forward passes is
still fast enough for the verification loop, but it should be stated.

Both the ensemble and the mean member are reported, so the gain is visible
and the honest comparison is available.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from c1_graph import edge_types_present
from wdt_checkpoint import load_checkpoint
from wdt_data import (load_runs, SnapshotDataset, Normalizer,
                      LabelNormalizer, compute_bin_edges, GNB_COLS)

BASE = os.environ.get("GEODE_S06C", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "scenario06c"))
SEEDS = (61, 62, 63)
POWERS = ("p37", "p43", "p46")
SLEEP = {61: [0, 1, 2, 3, 4, 5], 62: [0, 1, 2, 5], 63: [0, 1, 2, 3, 4, 5]}
PHASE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_paths_for(seeds):
    p = {}
    for s in seeds:
        for pw in POWERS:
            p[f"s{s}_{pw}"] = os.path.join(BASE, f"extracted_s06c{s}_{pw}")
        for k in SLEEP[s]:
            p[f"s{s}_p43_sleep{k}"] = os.path.join(
                BASE, f"extracted_s06c{s}_p43_sleep{k}")
    return p


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


@torch.no_grad()
def collect(model, loader):
    """Raw (normalised-space) predictions and truths for one model."""
    model.eval()
    out = defaultdict(list)
    for batch in loader:
        batch = batch.to(DEVICE)
        ea = {r: batch[r].edge_attr for r in batch.edge_types
              if hasattr(batch[r], "edge_attr") and batch[r].edge_attr is not None}
        o = model(batch.x_dict, batch.edge_index_dict, ea)
        out["energy_p"].append(o["energy"].squeeze(-1).cpu().numpy())
        out["delay_p"].append(o["delay"].squeeze(-1).cpu().numpy())
        out["jitter_p"].append(o["jitter"].squeeze(-1).cpu().numpy())
        out["pl_p"].append(torch.sigmoid(o["pl_binary_logit"]).cpu().numpy())
        out["energy_t"].append(batch["gnb"].y.squeeze(-1).cpu().numpy())
        out["delay_t"].append(batch["ue"].y[:, 0].cpu().numpy())
        out["jitter_t"].append(batch["ue"].y[:, 1].cpu().numpy())
        out["pl_t"].append(batch["ue"].pl_binary.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate_rotation(test_seed, ckpts, args):
    """All initialisations of one rotation, individually and ensembled."""
    model0, ctx = load_checkpoint(ckpts[0], DEVICE)
    flags = ctx["flags"]
    data = load_runs(run_paths_for(SEEDS))

    tag_snaps = (data["gnb"][["run_tag", "snapshot_id"]].drop_duplicates()
                 .groupby("run_tag").snapshot_id
                 .apply(lambda s: np.sort(s.unique())).to_dict())
    te_ids = []
    for t in [t for t in tag_snaps if t.startswith(f"s{test_seed}_")]:
        te_ids.extend(tag_snaps[t][args.stride // 2::args.stride].tolist())
    te_ids = sorted(te_ids)

    # Normalisers come from the checkpoint, not refitted: they must be the
    # ones the weights were trained under.
    gnb_norm = ctx["gnb_norm"]
    ue_norm = ctx["ue_norm"]
    energy_norm = ctx["energy_norm"]
    kpi_norms = ctx["kpi_norm"]
    edges = ctx["pl_bin_edges"]

    ds = SnapshotDataset(data, te_ids, gnb_norm, ue_norm, energy_norm,
                         kpi_norms, edges, **flags)
    loader = DataLoader(ds, batch_size=args.batch)

    per_model = []
    preds = defaultdict(list)
    truth = None
    for ck in ckpts:
        m, _ = load_checkpoint(ck, DEVICE)
        r = collect(m, loader)
        if truth is None:
            truth = {k: r[k] for k in ("energy_t", "delay_t", "jitter_t", "pl_t")}
        for k in ("energy_p", "delay_p", "jitter_p", "pl_p"):
            preds[k].append(r[k])
        per_model.append({
            "energy": metrics(r["energy_t"], r["energy_p"], energy_norm),
            "delay": metrics(r["delay_t"], r["delay_p"], kpi_norms["delay"]),
            "jitter": metrics(r["jitter_t"], r["jitter_p"], kpi_norms["jitter"]),
            "packet_loss": {"AUROC": auroc(r["pl_t"], r["pl_p"])},
        })

    ens = {k: np.mean(np.stack(v), axis=0) for k, v in preds.items()}
    ensemble = {
        "energy": metrics(truth["energy_t"], ens["energy_p"], energy_norm),
        "delay": metrics(truth["delay_t"], ens["delay_p"], kpi_norms["delay"]),
        "jitter": metrics(truth["jitter_t"], ens["jitter_p"], kpi_norms["jitter"]),
        "packet_loss": {"AUROC": auroc(truth["pl_t"], ens["pl_p"])},
    }
    return per_model, ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--kind", default="gnn", choices=["gnn", "blind"])
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    pat = os.path.join(args.ckpt_dir, f"wdt_{args.tag}_{args.kind}_test*_init*.pt")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"no checkpoints matching {pat}")
        print("the final run needs --save-all for every initialisation to exist")
        return

    by_seed = defaultdict(list)
    for f in files:
        m = re.search(r"_test(\d+)_init(\d+)\.pt$", f)
        if m:
            by_seed[int(m.group(1))].append(f)
    print(f"found {len(files)} checkpoints across {len(by_seed)} rotations")
    for s, fs in sorted(by_seed.items()):
        print(f"  test seed {s}: {len(fs)} initialisations")
    if all(len(fs) < 2 for fs in by_seed.values()):
        print("\nonly one initialisation per rotation -- nothing to ensemble")
        return

    all_members, all_ens = [], []
    for seed, ckpts in sorted(by_seed.items()):
        if len(ckpts) < 2:
            print(f"\n  (seed {seed} has one checkpoint, skipping)")
            continue
        print(f"\n=== test seed {seed}, {len(ckpts)} models ===")
        members, ens = evaluate_rotation(seed, ckpts, args)
        for i, m in enumerate(members):
            print(f"  member {i+1}   E {m['energy']['R2']:.3f}  "
                  f"D {m['delay']['R2']:.3f}  J {m['jitter']['R2']:.3f}  "
                  f"PL {m['packet_loss']['AUROC']:.3f}")
        print(f"  ENSEMBLE   E {ens['energy']['R2']:.3f}  "
              f"D {ens['delay']['R2']:.3f}  J {ens['jitter']['R2']:.3f}  "
              f"PL {ens['packet_loss']['AUROC']:.3f}")
        all_members.extend(members)
        all_ens.append(ens)

    if not all_ens:
        return

    print("\n" + "=" * 76)
    print("ENSEMBLE vs MEAN MEMBER -- averaged over rotations")
    print("=" * 76)
    print(f"{'target':<14}{'mean member':>14}{'ensemble':>12}{'gain':>10}")
    for tgt, key in (("energy", "R2"), ("delay", "R2"),
                     ("jitter", "R2"), ("packet_loss", "AUROC")):
        mem = np.mean([m[tgt][key] for m in all_members])
        en = np.mean([e[tgt][key] for e in all_ens])
        print(f"{tgt:<14}{mem:>14.4f}{en:>12.4f}{en-mem:>+10.4f}")

    print(f"\n{'target':<14}{'member MAPE':>14}{'ensemble MAPE':>15}")
    for tgt in ("energy", "delay", "jitter"):
        mem = np.mean([m[tgt]["MAPE"] for m in all_members])
        en = np.mean([e[tgt]["MAPE"] for e in all_ens])
        print(f"{tgt:<14}{mem:>13.1f}%{en:>14.1f}%")

    print("\nThe ensemble averages predictions before scoring; the member")
    print("column averages scores. A positive gain means the three models err")
    print("in different directions and part of the error cancels.")


if __name__ == "__main__":
    main()