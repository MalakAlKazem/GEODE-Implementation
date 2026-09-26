"""
cmoa_5g3e.py -- C4: CMOA-DT verification loop for 5G3E, using the v7
model (blind control, ablation-tested, lag+delta-aware) instead of
v3's WirelessDT6Gv3. Ported from src/v3/cmoa_v3.py's C-M-O-A structure:

  C -- Candidate Generation : gNBs with E_hat_i > E_thresh (top-K by predicted energy)
  M -- Modify               : build counterfactual graph G'_i (zero gNB's own
                               features, remove its serving + backhaul edges)
  O -- Observe               : run the model on G'_i -> predicted KPIs under sleep
  A -- Approve                : safe iff D_hat' <= D_max AND J_hat' <= J_max
                               for every remaining (non-sleeping) gNB

Two real differences from v3's version
------------------------------------------
1. Runs on the actual held-out Day 3 test set, not a validation split of
   the same training day -- this is the digital twin exercised on data
   it never trained on, which is the honest way to evaluate it.
2. Packet loss is DROPPED from the safety check. v3 kept a PL <= 1%
   constraint from the PDF spec, but packet_loss is a measured constant
   (0.0) across the entire 5G3E dataset (see c3_heads.py's docstring) --
   there is no real PL signal to check a sleep action against here, and
   keeping a fake constraint that always trivially passes would be
   worse than dropping it and saying why.

predict_delta handling
------------------------
If the loaded checkpoint used predict_delta, the model's raw head
output is a CHANGE from the previous snapshot, not the KPI itself.
Reconstruction (lag + predicted delta) happens the same way here as in
train_wdt_5g3e.py's run_epoch/denorm_metrics. The lag values themselves
are NOT touched by the counterfactual -- a gNB's real recent history
doesn't change just because we're asking "what if it slept starting
now" -- only its current features and edges do.
"""

import copy
import csv
import os
import time

import torch
from torch_geometric.loader import DataLoader

from wdt_data import load_norm_stats, load_split, SnapshotDataset, relations_for
from wdt_model import WirelessDT6G

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_5g3e")
OUT_DIR = os.path.join(os.path.dirname(__file__), "cmoa_out")

# -- C: candidate generation ------------------------------------------------
E_THRESH = 480.0       # W -- same threshold v3 used; train-set energy mean
                       # is ~506.6W (norm_stats.pt), so this is consistent
K_CANDIDATES = 2

# -- A: QoS safety thresholds (v3's testbed-adapted values; PL dropped) ----
D_MAX_MS = 180.0
J_MAX_MS = 20.0

# -- Energy saving (3GPP TR 38.840) -----------------------------------------
E_SLEEP_FRAC = 0.12

# -- Drift detection --------------------------------------------------------
DRIFT_WINDOW = 100
DRIFT_THRESH = 0.20

# -- DT synchronisation -------------------------------------------------------
ALPHA_SYNC = 0.10


def load_model_from_checkpoint(path):
    ck = torch.load(path, weights_only=False)
    model = WirelessDT6G(
        ck["in_dims"], ck["relations"], hidden=ck["hidden"], rounds=ck["rounds"],
        use_moe=ck["use_moe"], use_gru=ck["use_gru"], dropout=ck["dropout"],
        attention_on=(("gnb", "interferes", "gnb") if ck.get("use_attn", True) else None),
        predict_delta=ck.get("predict_delta", False),
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def make_counterfactual(data, gnb_idx, norms):
    """M-step: build the graph as it would look with gnb_idx asleep.

    Calibration note (see CMOA_RESULTS.md for the full diagnostic) --
    an earlier version zeroed the sleeping gNB's own 5 features, which
    fed the model THREE combinations that never occur in real training
    data at once: an impossible all-zero site one-hot (real gNBs are
    always at exactly one site), a mean_rsrp normalising to -15.9
    against an observed range of [-5.2, 2.4], and nof_ue set to a value
    5 std devs from the ONLY value nof_ue ever takes in training (it is
    a hardcoded constant, 5, in every real row -- the model never saw
    it vary and almost certainly learned zero sensitivity to it, so
    there is no calibrated substitute value to reach for). This is the
    same shape of bug as 06c's C4 originally had (see its own
    documented transmit-power miscalibration) -- an input the model
    was never calibrated to respond to, not a physically wrong one.

    Fix: leave the sleeping gNB's own features COMPLETELY untouched.
    No invented value, zero out-of-distribution risk. The sleep signal
    comes entirely from removing its serving and backhaul edges --
    exactly the mechanism the no_e4 ablation already proved is what
    the model actually learned to use for energy, so nothing about the
    twin's real reasoning is lost by not also corrupting its local
    features.

    Stranded UEs are reassigned to another ACTIVE gNB at the same
    physical site (site membership read from the gNB's own, untouched,
    one-hot -- the only locality signal available; 5G3E has no UE
    position data, unlike 06c's Euclidean nearest-cell reassignment).
    The receiving gNB's mean_rsrp is recomputed from its new UE set,
    since that IS a feature the model has learned to use.
    """
    d = copy.deepcopy(data)
    n_gnb = d["gnb"].x.shape[0]
    site_id = d["gnb"].x[:, 1:4].argmax(dim=1)
    active = torch.ones(n_gnb, dtype=torch.bool)
    active[gnb_idx] = False

    if ("ue", "connects", "gnb") in d.edge_types:
        ei = d["ue", "connects", "gnb"].edge_index
        stranded = ei[1] == gnb_idx
        if stranded.any():
            same_site = active & (site_id == site_id[gnb_idx])
            pool = same_site.nonzero(as_tuple=True)[0]
            if len(pool) == 0:
                pool = active.nonzero(as_tuple=True)[0]  # no same-site gNB active -- fall back
            stranded_ue = ei[0][stranded]
            new_targets = pool[torch.arange(len(stranded_ue)) % len(pool)]
            new_ei = torch.stack([stranded_ue, new_targets])
            ei = torch.cat([ei[:, ~stranded], new_ei], dim=1)
            d["ue", "connects", "gnb"].edge_index = ei
            d["gnb", "rev_connects", "ue"].edge_index = ei.flip(0)

            ue_rsrp_raw = d["ue"].x[:, 0] * norms["ue"].std_[0] + norms["ue"].mean_[0]
            for g in torch.unique(new_targets).tolist():
                served = ei[0][ei[1] == g]
                if len(served) > 0:
                    new_mean_raw = ue_rsrp_raw[served].mean().item()
                    d["gnb"].x[g, 4] = float((new_mean_raw - norms["gnb"].mean_[4])
                                             / norms["gnb"].std_[4])

    if ("gnb", "backhaul", "srv") in d.edge_types:
        ei = d["gnb", "backhaul", "srv"].edge_index
        d["gnb", "backhaul", "srv"].edge_index = ei[:, ei[0] != gnb_idx]
    if ("srv", "rev_backhaul", "gnb") in d.edge_types:
        ei = d["srv", "rev_backhaul", "gnb"].edge_index
        d["srv", "rev_backhaul", "gnb"].edge_index = ei[:, ei[1] != gnb_idx]

    return d


def predict_physical(model, data, y_norms):
    """Run the model, reconstruct absolute predictions (lag + delta if
    predict_delta), return [N_gnb, 3] energy_W/delay_ms/jitter_ms."""
    predict_delta = model.heads.predict_delta
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    if predict_delta:
        lag = data["gnb"].lag_y_norm
        e = lag[:, 0] + out["energy"].squeeze(-1)
        d = lag[:, 1] + out["delay"].squeeze(-1)
        j = lag[:, 2] + out["jitter"]
    else:
        e = out["energy"].squeeze(-1)
        d = out["delay"].squeeze(-1)
        j = out["jitter"]
    e = y_norms["energy_W"].inverse(e)
    d = y_norms["delay_ms"].inverse(d)
    j = y_norms["jitter_ms"].inverse(j)
    return torch.stack([e, d, j], dim=1)


def qos_safe(y_pred_phys, gnb_idx, n_gnb):
    active = torch.ones(n_gnb, dtype=torch.bool)
    active[gnb_idx] = False
    y_act = y_pred_phys[active]
    d_max = y_act[:, 1].max().item()
    j_max = y_act[:, 2].max().item()
    safe = (d_max <= D_MAX_MS) and (j_max <= J_MAX_MS)
    return safe, {"d_max_pred": d_max, "j_max_pred": j_max}


def cmoa_step(model, data, norms, snap_id):
    y_norms = norms["y"]
    n_gnb = data["gnb"].x.size(0)

    y_pred = predict_physical(model, data, y_norms)   # [n_gnb, 3] baseline
    y_true = data["gnb"].y_raw[:, :3]

    e_pred = y_pred[:, 0]
    topk_e, topk_idx = torch.topk(e_pred, min(K_CANDIDATES, n_gnb))
    candidates = [(idx.item(), topk_e[k].item()) for k, idx in enumerate(topk_idx)
                 if topk_e[k].item() > E_THRESH]

    approved, rejected = [], []
    total_delta_e, total_active_e = 0.0, e_pred.sum().item()

    for gnb_idx, e_val in candidates:
        data_prime = make_counterfactual(data, gnb_idx, norms)
        y_pred_prime = predict_physical(model, data_prime, y_norms)
        safe, qos_vals = qos_safe(y_pred_prime, gnb_idx, n_gnb)

        e_sleep = E_SLEEP_FRAC * e_val
        delta_e = e_val - e_sleep
        record = {"gnb_idx": gnb_idx, "e_active": round(e_val, 2),
                  "e_sleep": round(e_sleep, 2), "delta_e": round(delta_e, 2),
                  "safe": safe, **{k: round(v, 2) for k, v in qos_vals.items()}}
        (approved if safe else rejected).append(record)
        if safe:
            total_delta_e += delta_e

    esr_pct = (total_delta_e / total_active_e * 100.0) if total_active_e > 0 else 0.0
    eps = 1e-6
    e_mape = ((y_pred[:, 0] - y_true[:, 0]).abs() / (y_true[:, 0].abs() + eps)).mean().item() * 100

    return {"snap_id": snap_id, "n_candidates": len(candidates),
           "n_approved": len(approved), "n_rejected": len(rejected),
           "approved": approved, "rejected": rejected, "esr_pct": esr_pct,
           "total_delta_e": total_delta_e, "total_active_e": total_active_e,
           "e_mape": e_mape, "e_pred_mean": e_pred.mean().item()}


def run_cmoa_loop(model, dataset, norms, split_name="test"):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("  C4 -- CMOA-DT Verification Loop (5G3E v7)")
    print("=" * 72)
    print(f"  Split       : {split_name}  ({len(dataset)} snapshots)")
    print(f"  E_thresh    : {E_THRESH}W    K_cand={K_CANDIDATES}")
    print(f"  QoS limits  : D<={D_MAX_MS}ms  J<={J_MAX_MS}ms  "
         f"(PL dropped -- measured constant on this testbed)")
    print(f"  E_sleep     : {E_SLEEP_FRAC*100:.0f}% of active  (3GPP TR 38.840)")
    print("-" * 72)

    results, e_mape_history, dt_sync = [], [], None
    t0 = time.time()

    for snap_id in range(len(dataset)):
        data = dataset[snap_id]
        result = cmoa_step(model, data, norms, snap_id)
        results.append(result)

        e_obs = result["e_pred_mean"]
        dt_sync = e_obs if dt_sync is None else (1 - ALPHA_SYNC) * dt_sync + ALPHA_SYNC * e_obs

        e_mape_history.append(result["e_mape"])
        if len(e_mape_history) > DRIFT_WINDOW:
            e_mape_history.pop(0)
        rolling_mape = sum(e_mape_history) / len(e_mape_history)
        drift = rolling_mape >= DRIFT_THRESH * 100
        result["rolling_e_mape"], result["drift_flag"], result["dt_sync_e"] = \
            rolling_mape, drift, dt_sync

        if snap_id % 500 == 0 or snap_id < 3 or drift:
            print(f"  {snap_id:5d} | cand {result['n_candidates']:2d} | "
                 f"ok {result['n_approved']:2d} | rej {result['n_rejected']:2d} | "
                 f"ESR {result['esr_pct']:5.2f}% | E-MAPE {result['e_mape']:6.2f}% | "
                 f"roll {rolling_mape:6.2f}% | DT {dt_sync:6.1f}W | "
                 f"{'DRIFT' if drift else ''}")

    elapsed = time.time() - t0
    n = len(results)
    total_cand = sum(r["n_candidates"] for r in results)
    total_ok = sum(r["n_approved"] for r in results)
    total_rej = sum(r["n_rejected"] for r in results)
    mean_esr = sum(r["esr_pct"] for r in results) / n
    total_kwh = sum(r["total_delta_e"] for r in results) / 1000.0
    n_drift = sum(1 for r in results if r["drift_flag"])

    print("\n" + "=" * 72)
    print("  CMOA LOOP -- SUMMARY")
    print("=" * 72)
    print(f"  Snapshots processed  : {n}  ({elapsed:.1f}s, {elapsed/n*1000:.0f}ms/snap)")
    print(f"  Candidates generated : {total_cand}  ({total_cand/n:.2f}/snap)")
    print(f"  Actions approved     : {total_ok}  ({100*total_ok/max(total_cand,1):.1f}% of candidates)")
    print(f"  Actions rejected     : {total_rej}")
    print(f"  Mean ESR             : {mean_esr:.2f}%")
    print(f"  Cumulative dE        : {total_kwh:.2f} kWh over {n} snapshots")
    print(f"  Final rolling MAPE   : {rolling_mape:.2f}%  (drift threshold {DRIFT_THRESH*100:.0f}%)")
    print(f"  Drift events         : {n_drift} / {n} snapshots")
    print(f"  C5 trigger           : {'YES' if rolling_mape >= DRIFT_THRESH*100 else 'NO'}")

    log_path = os.path.join(OUT_DIR, f"cmoa_{split_name}_log.csv")
    fields = ["snap_id", "n_candidates", "n_approved", "n_rejected", "esr_pct",
             "total_delta_e", "total_active_e", "e_mape", "rolling_e_mape",
             "drift_flag", "dt_sync_e"]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"\n  Log: {log_path}")

    sample = next((r for r in results if r["n_approved"] > 0), None)
    if sample:
        a = sample["approved"][0]
        print(f"\n  Sample approved action (snap {sample['snap_id']}):")
        print(f"    Sleep gNB #{a['gnb_idx']}: E_active={a['e_active']}W -> "
             f"E_sleep={a['e_sleep']}W (dE={a['delta_e']}W saved)")
        print(f"    Post-sleep QoS: D_max={a['d_max_pred']:.1f}ms<={D_MAX_MS}ms  "
             f"J_max={a['j_max_pred']:.1f}ms<={J_MAX_MS}ms")

    sample_r = next((r for r in results if r["n_rejected"] > 0), None)
    if sample_r:
        rj = sample_r["rejected"][0]
        print(f"\n  Sample REJECTED action (snap {sample_r['snap_id']}):")
        print(f"    Candidate gNB #{rj['gnb_idx']}  E={rj['e_active']}W")
        violated = []
        if rj["d_max_pred"] > D_MAX_MS: violated.append(f"D={rj['d_max_pred']:.1f}>{D_MAX_MS}ms")
        if rj["j_max_pred"] > J_MAX_MS: violated.append(f"J={rj['j_max_pred']:.1f}>{J_MAX_MS}ms")
        print(f"    Violated: {', '.join(violated)}")

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default=os.path.join(CKPT_DIR, "wdt5g3e_full_delta_gnn_init1.pt"),
                    help="path to a v7 gnn checkpoint")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model, ck = load_model_from_checkpoint(args.checkpoint)
    print(f"  kind={ck['kind']}  rounds={ck['rounds']}  hidden={ck['hidden']}  "
         f"use_lag={ck.get('use_lag')}  predict_delta={ck.get('predict_delta')}  "
         f"best_epoch={ck['best_epoch']}")

    norms = load_norm_stats()
    split = load_split()
    ds = SnapshotDataset(split[args.split], norms, use_lag=ck.get("use_lag", False),
                        **ck["edge_flags"])

    run_cmoa_loop(model, ds, norms, split_name=args.split)


if __name__ == "__main__":
    main()
