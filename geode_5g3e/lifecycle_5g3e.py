"""
lifecycle_5g3e.py -- C5: DT lifecycle fine-tuning for 5G3E v7. Ported
from src/v3/lifecycle_v3.py, adapted to the v7 model.

Freeze target is simpler than v3's module-name list: WirelessDT6G
already separates the whole HMPGNN encoder (model.encoder) from the
four prediction heads (model.heads) as two top-level attributes, so
freezing the encoder and leaving heads trainable is just those two.

Three phases, same as v3
--------------------------
1. Monitor: rolling energy MAPE over the real Day3 test set. We already
   know from cmoa_5g3e.py's run that this never crosses the 20% drift
   threshold (rolling MAPE stayed ~0.13% throughout) -- summarised here
   rather than re-run at full cost.
2. Injected-drift demo: v3's own showcased C5 result used an ARTIFICIAL
   drift (shift energy by +4.5 sigma, delay by +2.0 sigma) because no
   natural drift occurred on its real Day2 test either -- the mechanism
   has to be demonstrated somehow. Same approach here: shift the last
   FT_N_SAMPLES test snapshots' energy/delay labels, confirm the rolling
   MAPE crosses the trigger threshold, then fine-tune and show recovery.
3. Before/after comparison on the drifted set.

Fine-tuning happens on the same drifted snapshots it's evaluated on --
this is a demonstration of the MECHANISM (does freeze+refit recover
accuracy at all), not a generalisation test, matching v3's own framing.
"""

import copy
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from wdt_data import load_norm_stats, load_split, SnapshotDataset
from cmoa_5g3e import load_model_from_checkpoint, CKPT_DIR
import os

DRIFT_WINDOW = 100
DRIFT_THRESH = 0.20
MONITOR_BATCH = 32

FT_N_SAMPLES = 1000
FT_LR = 1e-4
FT_EPOCHS = 30
FT_PATIENCE = 5
FT_BATCH = 32

ENERGY_SHIFT_SIGMA = 4.5   # matches v3's injected-drift demo
DELAY_SHIFT_SIGMA = 2.0

W = (0.35, 0.30, 0.20, 0.15)  # energy, delay, jitter, PL loss weights


def energy_mape_phys(y_pred_phys, y_true_phys):
    return ((y_pred_phys - y_true_phys).abs() / (y_true_phys.abs() + 1e-6)).mean().item() * 100


def predict_energy_delay(model, data, y_norms):
    predict_delta = model.heads.predict_delta
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    if predict_delta:
        lag = data["gnb"].lag_y_norm
        e = lag[:, 0] + out["energy"].squeeze(-1)
        d = lag[:, 1] + out["delay"].squeeze(-1)
    else:
        e = out["energy"].squeeze(-1)
        d = out["delay"].squeeze(-1)
    return y_norms["energy_W"].inverse(e), y_norms["delay_ms"].inverse(d)


def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.heads.parameters():
        p.requires_grad_(True)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable


def inject_drift(dataset, norms):
    """Shift energy/delay labels (normalised AND raw) on every graph in
    the dataset by several std devs -- simulates the network having
    drifted away from what the DT was trained on. Lag features (if
    present) are NOT shifted, matching the idea that the drift is new,
    not something the model could already have adapted to via lag."""
    e_std = norms["y"]["energy_W"].std_
    d_std = norms["y"]["delay_ms"].std_
    e_shift_raw = ENERGY_SHIFT_SIGMA * e_std
    d_shift_raw = DELAY_SHIFT_SIGMA * d_std
    e_shift_norm = ENERGY_SHIFT_SIGMA  # a shift of N std devs is exactly N in z-score space
    d_shift_norm = DELAY_SHIFT_SIGMA

    drifted = []
    for g in dataset.graphs:
        d = copy.deepcopy(g)
        d["gnb"].y_raw[:, 0] += e_shift_raw
        d["gnb"].y_raw[:, 1] += d_shift_raw
        d["gnb"].y[:, 0] += e_shift_norm
        d["gnb"].y[:, 1] += d_shift_norm
        drifted.append(d)
    return drifted, e_shift_raw, d_shift_raw


def rolling_mape_over(model, graphs, y_norms, window=DRIFT_WINDOW, batch=MONITOR_BATCH):
    """Batched rolling energy MAPE, same cadence as cmoa_5g3e.py's
    monitoring loop."""
    loader = DataLoader(graphs, batch_size=batch, shuffle=False)
    hist, trigger_at = [], None
    seen = 0
    roll_window = max(1, window // batch)
    for data in loader:
        e_pred, _ = predict_energy_delay(model, data, y_norms)
        e_true = data["gnb"].y_raw[:, 0]
        em = energy_mape_phys(e_pred, e_true)
        hist.append(em)
        if len(hist) > roll_window:
            hist.pop(0)
        rolling = sum(hist) / len(hist)
        seen += data["gnb"].x.shape[0] // 13  # snapshots in this batch (13 gNBs each)
        if rolling >= DRIFT_THRESH * 100 and trigger_at is None:
            trigger_at = seen
    return (sum(hist) / len(hist)) if hist else 0.0, trigger_at


def finetune(model, graphs, y_norms):
    frozen, trainable = freeze_encoder(model)
    total = frozen + trainable
    print(f"  Encoder frozen  : {frozen:,} params ({100*frozen/total:.1f}%)")
    print(f"  Heads trainable : {trainable:,} params ({100*trainable/total:.1f}%)")
    print(f"  lr={FT_LR}  batch={FT_BATCH}  max_epochs={FT_EPOCHS}  patience={FT_PATIENCE}")

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=FT_LR, weight_decay=1e-4)
    loader = DataLoader(graphs, batch_size=FT_BATCH, shuffle=True)
    predict_delta = model.heads.predict_delta

    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(1, FT_EPOCHS + 1):
        model.train()
        t0, tot_loss, n = time.time(), 0.0, 0
        for data in loader:
            opt.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            y = data["gnb"].y
            if predict_delta:
                lag = data["gnb"].lag_y_norm
                l_e = F.huber_loss(out["energy"].squeeze(-1), y[:, 0] - lag[:, 0])
                l_d = F.mse_loss(out["delay"].squeeze(-1), y[:, 1] - lag[:, 1])
                l_j = F.mse_loss(out["jitter"], y[:, 2] - lag[:, 2])
            else:
                l_e = F.huber_loss(out["energy"].squeeze(-1), y[:, 0])
                l_d = F.mse_loss(out["delay"].squeeze(-1), y[:, 1])
                l_j = torch.zeros((), device=y.device)  # jitter head structure varies; energy/delay are what C5 cares about
            loss = W[0]*l_e + W[1]*l_d + W[2]*l_j
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            tot_loss += float(loss); n += 1

        model.eval()
        e_pred_all, e_true_all = [], []
        with torch.no_grad():
            for data in DataLoader(graphs, batch_size=64, shuffle=False):
                e_pred, _ = predict_energy_delay(model, data, y_norms)
                e_pred_all.append(e_pred); e_true_all.append(data["gnb"].y_raw[:, 0])
        em = energy_mape_phys(torch.cat(e_pred_all), torch.cat(e_true_all))
        avg_loss = tot_loss / n
        print(f"  epoch {ep:3d}  loss {avg_loss:.4f}  E-MAPE {em:6.2f}%  ({time.time()-t0:.1f}s)")

        if avg_loss < best_loss:
            best_loss, bad = avg_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                print(f"  [early stop at epoch {ep}]")
                break

    model.load_state_dict(best_state)
    return best_loss


def main():
    ckpt_path = os.path.join(CKPT_DIR, "wdt5g3e_full_delta_gnn_init1.pt")
    print(f"Loading model from {ckpt_path}")
    model, ck = load_model_from_checkpoint(ckpt_path)
    print(f"  rounds={ck['rounds']}  use_lag={ck.get('use_lag')}  "
         f"predict_delta={ck.get('predict_delta')}")

    norms = load_norm_stats()
    split = load_split()
    test_ds = SnapshotDataset(split["test"], norms, use_lag=ck.get("use_lag", False),
                              **ck["edge_flags"])

    print("\n" + "=" * 70)
    print("  PHASE 1 -- Natural drift check (real Day 3 test set)")
    print("=" * 70)
    print("  Already measured by cmoa_5g3e.py's full run: rolling energy MAPE")
    print("  stayed ~0.13% across all 6,973 snapshots. NO natural drift --")
    print("  the DT trained on Day1+2 generalises accurately to Day 3.")
    print("  C5's trigger mechanism therefore needs to be demonstrated with")
    print("  INJECTED drift, same as v3's own showcased result.")

    n_total = len(test_ds)
    ft_start = max(0, n_total - FT_N_SAMPLES)
    tail_graphs = test_ds.graphs[ft_start:]
    print(f"\n  Using last {len(tail_graphs)} test snapshots for the drift demo "
         f"(snaps {ft_start}..{n_total-1})")

    print("\n" + "=" * 70)
    print(f"  PHASE 2a -- Inject drift (+{ENERGY_SHIFT_SIGMA} sigma energy, "
         f"+{DELAY_SHIFT_SIGMA} sigma delay) and confirm C5 triggers")
    print("=" * 70)
    drifted_graphs, e_shift, d_shift = inject_drift(
        type("D", (), {"graphs": tail_graphs})(), norms)
    print(f"  Energy shift: +{e_shift:.1f}W   Delay shift: +{d_shift:.1f}ms")

    rolling_before, trigger_at = rolling_mape_over(model, drifted_graphs, norms["y"])
    print(f"  Rolling energy MAPE on drifted tail: {rolling_before:.2f}%  "
         f"(trigger threshold {DRIFT_THRESH*100:.0f}%)")
    if trigger_at is not None:
        print(f"  C5 TRIGGERED at ~snapshot {trigger_at} of the drifted tail")
    else:
        print("  C5 did NOT trigger -- drift magnitude may need to be larger")

    print("\n" + "=" * 70)
    print("  PHASE 2b -- Freeze encoder, fine-tune heads on the drifted tail")
    print("=" * 70)
    model_before = copy.deepcopy(model)
    finetune(model, drifted_graphs, norms["y"])

    print("\n" + "=" * 70)
    print("  PHASE 3 -- Before vs after fine-tuning, on the drifted tail")
    print("=" * 70)

    def mape_of(m):
        m.eval()
        ep, et = [], []
        with torch.no_grad():
            for data in DataLoader(drifted_graphs, batch_size=64, shuffle=False):
                e_pred, _ = predict_energy_delay(m, data, norms["y"])
                ep.append(e_pred); et.append(data["gnb"].y_raw[:, 0])
        return energy_mape_phys(torch.cat(ep), torch.cat(et))

    e_before = mape_of(model_before)
    e_after = mape_of(model)
    print(f"  Energy MAPE before fine-tuning: {e_before:.2f}%")
    print(f"  Energy MAPE after  fine-tuning: {e_after:.2f}%")
    print(f"  Recovered: {e_before - e_after:+.2f} points "
         f"({'RECOVERED below trigger' if e_after < DRIFT_THRESH*100 else 'still above trigger'})")


if __name__ == "__main__":
    main()
