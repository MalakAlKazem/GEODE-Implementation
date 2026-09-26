"""
c5_drift_test.py -- does lifecycle correction help under genuine
distribution shift?

    python3 c5_drift_test.py --checkpoint checkpoints/wdt_final_gnn_test61_init1.pt

The gap this closes
-------------------
c5_lifecycle.py replayed a held-out topology of the SAME dataset the twin
was trained on: six base stations, 500 m spacing, 25 resource blocks. The
twin's representation was still valid there; only the conditions varied,
becoming more congested in later traffic phases. Correction was measured
under non-stationarity, not under staleness, and it did harm.

A lifecycle exists for a twin whose representation has become WRONG for the
network it now sees. This replays scenario07 instead: four base stations,
1000 m spacing, 50 resource blocks, a different congestion regime. Nothing
is retrained beforehand -- retraining would remove the very condition being
tested. The graph encoder accepts a different node count unchanged, which is
what makes the test possible at all.

Two outcomes, both reportable
-----------------------------
If correction helps here, the lifecycle has a positive result and the
weakness of the earlier section closes: fine-tuning is beneficial under real
shift and harmful under mere non-stationarity, which is a usable operating
rule.

If correction still harms, the finding becomes stronger rather than weaker:
head-only correction on a small buffer fails even when the representation is
genuinely wrong, which is a substantive claim about the mechanism instead of
an untested one.

What to expect, and what not to read into it
--------------------------------------------
Initial accuracy will be poor, and that is the point rather than a fault.
Beyond the topology difference, the feature normalisers were fitted on
scenario06c statistics while scenario07 has a 2000x2000 m map, different
offered load and 50 resource blocks, so the inputs are out of distribution
in a second way. A deployed twin meeting a new network faces exactly that.
The question is not whether error is high but whether correction REDUCES it.

The trigger therefore has to be set relative to the error actually observed
here, not to the value used on scenario06c. The script measures first and
reports what a sensible trigger would be, then runs with it.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from c1_graph import build_graph, build_interference_edges
from c5_lifecycle import Lifecycle, read_run
from wdt_checkpoint import load_checkpoint, predict

S07 = os.environ.get("GEODE_S07", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "scenario07"))
S06C = os.environ.get("GEODE_S06C", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "scenario06c"))


def stream(folder, ctx, stride, gnb_norm, ue_norm):
    """Yield (graph, true_delay, true_energy) per retained snapshot."""
    run = read_run(folder)
    snaps = np.sort(run["gnb"].snapshot_id.unique())[stride // 2::stride]
    snap0 = run["gnb"][run["gnb"].snapshot_id == snaps[0]]
    interference = build_interference_edges(snap0)
    flags = ctx["flags"]

    for sid in snaps:
        gr = run["gnb"][run["gnb"].snapshot_id == sid].sort_values("gnb_index")
        ur = run["ue"][run["ue"].snapshot_id == sid].sort_values("ue_index")
        sr = run["serving"][run["serving"].snapshot_id == sid]
        kr = run["kpi"][run["kpi"].snapshot_id == sid].sort_values("ue_index")
        er = run["energy"][run["energy"].snapshot_id == sid].sort_values("gnb_index")
        if gr.empty or ur.empty or kr.empty or er.empty:
            continue
        gx = torch.tensor(gnb_norm.transform(gr), dtype=torch.float32)
        ux = torch.tensor(ue_norm.transform(ur), dtype=torch.float32)
        g = build_graph(gr, ur, sr, gx, ux, interference=interference,
                        handover=None, use_e1=flags["use_e1"],
                        use_e2=flags["use_e2"], use_e3=flags["use_e3"],
                        use_e4=flags["use_e4"])
        yield g, kr.delay_ms.to_numpy(), er.estimated_gnb_power_w.to_numpy()


def measure_only(model, ctx, items):
    """Error with no correction at all -- the control this test needs."""
    d_errs, e_errs = [], []
    for g, td, te in items:
        p = predict(model, ctx, g)
        pd_ms = np.asarray(p["delay_ms"]).ravel()
        pe_w = np.asarray(p["energy_w"]).ravel()
        nz = np.abs(td) > 1e-6
        if nz.any():
            d_errs.append(np.abs((td[nz] - pd_ms[nz]) / td[nz]).mean() * 100)
        e_errs.append(np.abs((te - pe_w) / te).mean() * 100)
    return np.array(d_errs), np.array(e_errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", default=S07,
                    help="folder containing the replay runs. Defaults to "
                         "scenario07. scenario06b's extracted runs happen to "
                         "live inside the scenario07 folder, so the same "
                         "default serves both.")
    ap.add_argument("--label", default=None,
                    help="what to call the replayed network in the output")
    ap.add_argument("--runs", nargs="+",
                    default=["extracted_s22_p43", "extracted_s23_p43",
                             "extracted_s24_p43"],
                    help="scenario07 runs replayed as one operational stream")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--trigger", type=float, default=None,
                    help="rolling delay MAPE that fires correction; measured "
                         "from the data if not given")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--min-buffer", type=int, default=30)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--out", default="c5_drift.csv")
    args = ap.parse_args()

    model, ctx = load_checkpoint(args.checkpoint)
    print(f"twin trained on scenario06c seeds {ctx['train_seeds']} "
          f"(6 gNB, 500 m ISD, 25 RB)")
    label = args.label or ("scenario07 (4 gNB, 1000 m ISD, 50 RB)"
                           if args.base == S07 and "s06b" not in args.runs[0]
                           else "the replayed network")
    print(f"replaying {label} -- never seen, nothing retrained\n")

    folders = [os.path.join(args.base, r) for r in args.runs]
    missing = [f for f in folders if not os.path.isdir(f)]
    if missing:
        print("missing folders:")
        for m in missing:
            print("  " + m)
        return
    folders = [f for f in folders if os.path.isdir(f)]

    gnb_norm, ue_norm = ctx["gnb_norm"], ctx["ue_norm"]

    # ---- build the stream once and hold it, so the no-correction control
    # ---- and the lifecycle run see identical inputs
    items = []
    for f in folders:
        got = list(stream(f, ctx, args.stride, gnb_norm, ue_norm))
        print(f"  {os.path.basename(f)}: {len(got)} cycles")
        items.extend(got)
    if not items:
        print("no usable snapshots")
        return
    print(f"  total {len(items)} cycles\n")

    # ---- control: no correction ----
    print("CONTROL -- no correction, error on the shifted network")
    d0, e0 = measure_only(model, ctx, items)
    print(f"  delay MAPE   mean {d0.mean():>7.1f}%   "
          f"first quarter {d0[:len(d0)//4].mean():>7.1f}%   "
          f"last quarter {d0[-len(d0)//4:].mean():>7.1f}%")
    print(f"  energy MAPE  mean {e0.mean():>7.1f}%   "
          f"first quarter {e0[:len(e0)//4].mean():>7.1f}%   "
          f"last quarter {e0[-len(e0)//4:].mean():>7.1f}%")

    # Compare against the twin on its own dataset, to show the shift is real
    own = os.path.join(S06C, f"extracted_s06c{ctx['test_seed']}_p43")
    if os.path.isdir(own):
        own_items = list(stream(own, ctx, args.stride, gnb_norm, ue_norm))
        d_own, e_own = measure_only(model, ctx, own_items)
        print(f"\n  for reference, same twin on its own held-out topology:")
        print(f"    delay MAPE {d_own.mean():.1f}%   "
              f"energy MAPE {e_own.mean():.1f}%")
        print(f"  the difference is the distribution shift this test needs")

    # ---- trigger, set from the data rather than carried over ----
    trigger = args.trigger
    if trigger is None:
        roll = pd.Series(d0).rolling(args.window).mean().dropna()
        trigger = float(np.percentile(roll, 25)) if len(roll) else d0.mean()
        print(f"\n  trigger not given; using the 25th percentile of the rolling "
              f"mean, {trigger:.1f}%")
        print(f"  (a value from the scenario06c run would be meaningless here, "
              f"since the error scale differs)")

    # ---- lifecycle run ----
    model, ctx = load_checkpoint(args.checkpoint)   # fresh weights
    lc = Lifecycle(model, ctx, window=args.window, mape_trigger=trigger,
                   ft_lr=args.ft_lr, ft_epochs=args.ft_epochs,
                   holdout=args.holdout, min_buffer=args.min_buffer)

    print(f"\nLIFECYCLE -- correction enabled, trigger {trigger:.1f}%")
    for t, (g, td, te) in enumerate(items):
        rec = lc.step(t, g, td, te)
        if rec["finetuned"]:
            v = ("helped" if rec["ft_gain"] < 0
                 else "HARMED" if rec["ft_gain"] > 0 else "no change")
            print(f"    t={t:>4}  rolling {rec['rolling_delay_mape']:>7.1f}% "
                  f"-> correction, held-out MAPE {rec['ft_gain']:+.1f}pp ({v})")

    h = pd.DataFrame(lc.history)
    h.to_csv(args.out, index=False)

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  cycles {len(h)}   corrections {len(lc.events)}")
    print(f"\n  {'':<22}{'no correction':>15}{'with correction':>17}")
    print(f"  {'delay MAPE, mean':<22}{d0.mean():>14.1f}%{h.delay_mape.mean():>16.1f}%")
    q = len(h) // 4
    print(f"  {'delay MAPE, last qtr':<22}"
          f"{d0[-q:].mean():>14.1f}%{h.delay_mape[-q:].mean():>16.1f}%")
    print(f"  {'energy MAPE, mean':<22}"
          f"{e0.mean():>14.1f}%{h.energy_mape.mean():>16.1f}%")

    if lc.events:
        gains = [e["gain"] for e in lc.events if np.isfinite(e["gain"])]
        if gains:
            helped = sum(1 for g_ in gains if g_ < 0)
            print(f"\n  corrections reducing held-out error: {helped}/{len(gains)}"
                  f"   mean change {np.mean(gains):+.1f} pp")
        better = h.delay_mape.mean() < d0.mean()
        print(f"\n  VERDICT: correction {'HELPS' if better else 'does NOT help'} "
              f"under genuine distribution shift.")
        if better:
            print("  Together with the scenario06c result this gives an operating")
            print("  rule: correct when the representation is wrong, not when")
            print("  conditions are merely harder.")
        else:
            print("  Head-only correction on a small buffer fails even when the")
            print("  representation is genuinely wrong. That is a stronger claim")
            print("  about the mechanism than the earlier result, not a weaker one.")
    else:
        print(f"\n  no corrections fired -- lower --trigger below "
              f"{trigger:.1f}% to exercise the mechanism")


if __name__ == "__main__":
    main()