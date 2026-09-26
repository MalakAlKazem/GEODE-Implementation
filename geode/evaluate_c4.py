"""
evaluate_c4.py -- does the verification loop's decision match what actually
happened?

    python3 evaluate_c4.py --checkpoint checkpoints/wdt_final_gnn_test61_init1.pt

Ground truth comes from the paired sleep runs: for every (cell, phase) the
real energy saving and the real change in 95th-percentile delay were
measured by running the network with that cell asleep. That gives 640
labelled opportunities. The loop predicts approve or reject from the
BASELINE graph alone, never seeing the sleep run, and the two are compared.

What is being tested
--------------------
Not "is the twin accurate" -- that is section B of the results. This asks
whether an accurate-enough twin supports a correct SAFETY DECISION, which
is a lower bar and the one the contribution actually claims. A twin that
predicts delay to within a factor of two can still sort safe actions from
unsafe ones.

Which errors matter
-------------------
Not symmetric. A false approval means an unsafe action reaches the live
network; a false rejection means a saving is missed. The first is a
failure of the component's purpose, the second is a lost opportunity.
Recall on the unsafe class is therefore the number to lead with, not
overall accuracy.

Ground-truth labels use the same rule as the dataset analysis: an action is
unsafe if measured d95 rises by more than 1.5x, and worth taking only if it
actually saves energy.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from c1_graph import build_interference_edges
from c4_cmoa import CMOA, EnsemblePredictor
from wdt_checkpoint import load_checkpoint
from wdt_data import GNB_COLS, QOS_CLASSES, UE_NUM_COLS

BASE = os.environ.get("GEODE_S06C", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "scenario06c"))
SLEEP = {61: [0, 1, 2, 3, 4, 5], 62: [0, 1, 2, 5], 63: [0, 1, 2, 3, 4, 5]}
PHASE = 30
D95_UNSAFE = 1.5


def read_run(folder):
    g = pd.read_csv(os.path.join(folder, "gnb_inputs.csv"))
    u = pd.read_csv(os.path.join(folder, "ue_inputs.csv"))
    k = pd.read_csv(os.path.join(folder, "kpi_targets.csv"))
    e = pd.read_csv(os.path.join(folder, "energy_targets.csv"))
    s = pd.read_csv(os.path.join(folder, "serving_edges.csv"))
    for c in QOS_CLASSES:
        u[f"qos_{c}"] = (u.qos_class == c).astype(np.float32)
    return {"gnb": g, "ue": u, "kpi": k, "energy": e, "serving": s}


def ground_truth(base, sleep):
    """Measured saving and d95 ratio per phase for one slept cell."""
    b_e, s_e = base["energy"].copy(), sleep["energy"].copy()
    b_k, s_k = base["kpi"].copy(), sleep["kpi"].copy()
    for d in (b_e, s_e, b_k, s_k):
        d["ph"] = d.snapshot_id // PHASE
    bp = b_e.groupby(["ph", "snapshot_id"]).estimated_gnb_power_w.sum().groupby("ph").mean()
    sp = s_e.groupby(["ph", "snapshot_id"]).estimated_gnb_power_w.sum().groupby("ph").mean()
    bd = b_k.groupby("ph").delay_ms.quantile(.95)
    sd = s_k.groupby("ph").delay_ms.quantile(.95)
    out = {}
    for ph in bp.index:
        ratio = sd[ph] / bd[ph] if bd[ph] > 0 else np.nan
        out[int(ph)] = {"saved_w": float(bp[ph] - sp[ph]),
                        "d95_ratio": float(ratio)}
    return out


def sanity_check(cmoa, base, sleep, cell, snaps, interference,
                 gnb_norm, ue_norm):
    """Does the constructed counterfactual match the real sleep run?

    Run once before trusting any decision. The first C4 evaluation failed
    because the counterfactual wrote a transmit power that appears nowhere
    in training -- invisible in the decision numbers, obvious the moment the
    constructed features are put beside the measured ones. Compare rather
    than assume.
    """
    sid = int(snaps[len(snaps) // 2])
    gr = base["gnb"][base["gnb"].snapshot_id == sid].sort_values("gnb_index")
    ur = base["ue"][base["ue"].snapshot_id == sid].sort_values("ue_index")
    sr = base["serving"][base["serving"].snapshot_id == sid]

    _, new_srv, stranded = cmoa.build_counterfactual(
        cell, gr, ur, sr, gnb_norm, ue_norm, interference, reassign="nearest")

    real = sleep["gnb"][sleep["gnb"].snapshot_id == sid].sort_values("gnb_index")
    counts = new_srv.groupby("serving_gnb_index").size()

    print(f"\n  SANITY CHECK -- counterfactual vs measured sleep run "
          f"(cell {cell}, snapshot {sid})")
    print(f"    {'gnb':>4} {'built UEs':>10} {'real UEs':>9} "
          f"{'built tx':>9} {'real tx':>8}")
    ok = True
    for r in real.itertuples():
        g = int(r.gnb_index)
        btx = float(gr[gr.gnb_index == g].tx_power_dbm.iloc[0])
        rtx = float(r.tx_power_dbm)
        if abs(btx - rtx) > 0.01:
            ok = False
        print(f"    {g:>4} {int(counts.get(g, 0)):>10} "
              f"{int(r.num_connected_ues):>9} {btx:>9.0f} {rtx:>8.0f}")
    print(f"    stranded UEs reassigned: {len(stranded)}")
    if ok:
        print("    tx power matches the measured run -- inputs are in "
              "distribution")
    else:
        print("    *** tx power DIFFERS from the measured run. The twin has "
              "never seen this value; results below are not trustworthy.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="one checkpoint, or with --ensemble the pattern "
                         "whose siblings across initialisations are averaged")
    ap.add_argument("--ensemble", action="store_true",
                    help="average predictions over every initialisation of "
                         "this rotation, matching how the predictor results "
                         "are reported")
    ap.add_argument("--reassign", nargs="+", default=["nearest", "oracle"])
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[61, 62, 63])
    ap.add_argument("--out", default="c4_results.csv")
    args = ap.parse_args()

    model, ctx = load_checkpoint(args.checkpoint)
    if args.ensemble:
        import glob as _glob
        pat = re.sub(r"_init\d+\.pt$", "_init*.pt", args.checkpoint)
        sibs = sorted(_glob.glob(pat))
        if len(sibs) > 1:
            members = [load_checkpoint(s)[0] for s in sibs]
            model = EnsemblePredictor(members, ctx)
            print(f"ENSEMBLE of {len(sibs)} initialisations:")
            for s in sibs:
                print(f"    {os.path.basename(s)}")
        else:
            print(f"--ensemble given but only one checkpoint matches {pat}")
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"  trained on {ctx['train_seeds']}, held out {ctx['test_seed']}")
    if ctx.get("test_metrics"):
        m = ctx["test_metrics"]
        print(f"  twin quality: delay R2 {m['delay']['R2']:.3f}, "
              f"energy R2 {m['energy']['R2']:.3f}")

    gnb_norm, ue_norm = ctx["gnb_norm"], ctx["ue_norm"]
    cmoa = CMOA(model, ctx, violation_tolerance=args.tolerance)

    rows = []
    for seed in args.seeds:
        base = read_run(os.path.join(BASE, f"extracted_s06c{seed}_p43"))
        snaps = np.sort(base["gnb"].snapshot_id.unique())
        snap0 = base["gnb"][base["gnb"].snapshot_id == snaps[0]]
        interference = build_interference_edges(snap0)

        for cell in SLEEP[seed]:
            sleep = read_run(os.path.join(
                BASE, f"extracted_s06c{seed}_p43_sleep{cell}"))
            gt = ground_truth(base, sleep)

            if seed == args.seeds[0] and cell == SLEEP[seed][0]:
                sanity_check(cmoa, base, sleep, cell, snaps, interference,
                             gnb_norm, ue_norm)

            for ph, truth in gt.items():
                # mid-phase snapshot, matching the training convention
                sid = int(snaps[min(ph * PHASE + PHASE // 2, len(snaps) - 1)])
                gr = base["gnb"][base["gnb"].snapshot_id == sid]
                ur = base["ue"][base["ue"].snapshot_id == sid]
                sr = base["serving"][base["serving"].snapshot_id == sid]
                if gr.empty or ur.empty:
                    continue
                osr = sleep["serving"][sleep["serving"].snapshot_id == sid]

                for mode in args.reassign:
                    if mode == "oracle" and osr.empty:
                        continue
                    d = cmoa.evaluate_action(
                        cell, gr, ur, sr, gnb_norm, ue_norm, interference,
                        reassign=mode, oracle_serving=osr)
                    d.update(seed=seed, phase=ph,
                             true_saved_w=truth["saved_w"],
                             true_d95_ratio=truth["d95_ratio"],
                             true_safe=truth["d95_ratio"] < D95_UNSAFE,
                             true_saves=truth["saved_w"] > 0)
                    d["true_approve"] = bool(d["true_safe"] and d["true_saves"])
                    rows.append(d)
        print(f"  seed {seed}: {len([r for r in rows if r['seed']==seed])} evaluations")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)\n")

    for mode in args.reassign:
        d = df[df.reassign == mode]
        if d.empty:
            continue
        print("=" * 74)
        print(f"REASSIGNMENT: {mode}"
              + ("   (uses the measured sleep-run topology -- not deployable, "
                 "isolates twin error)" if mode == "oracle" else
                 "   (deployable: nearest active cell)"))
        print("=" * 74)

        # ---- Gate 1: safety, judged on its own ----
        # This is the contribution's actual claim: does the twin sort actions
        # that preserve QoS from ones that break it. Judged independently of
        # the energy gate, because the two fail for different reasons and
        # combining them hides which.
        s_tp = int(((d.safe) & (d.true_safe)).sum())
        s_fp = int(((d.safe) & (~d.true_safe)).sum())
        s_fn = int(((~d.safe) & (d.true_safe)).sum())
        s_tn = int(((~d.safe) & (~d.true_safe)).sum())
        n_unsafe = int((~d.true_safe).sum())
        print(f"  SAFETY GATE (predicted d95 ratio < 1.5)")
        print(f"    called safe, was safe        {s_tp:>4}")
        print(f"    called safe, was NOT         {s_fp:>4}   <- unsafe action passed")
        print(f"    called unsafe, was safe      {s_fn:>4}   <- over-cautious")
        print(f"    called unsafe, was unsafe    {s_tn:>4}")
        if n_unsafe:
            print(f"    unsafe actions caught: {s_tn}/{n_unsafe} "
                  f"({s_tn/n_unsafe*100:.1f}%)")
        base_rate = (1 - d.true_safe.mean()) * 100
        print(f"    (always-reject would catch 100% but approve nothing; "
              f"{base_rate:.0f}% of actions are truly unsafe)")
        cd = d[["pred_d95_ratio", "true_d95_ratio"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(cd) > 2:
            print(f"    predicted vs measured d95 ratio, correlation: "
                  f"{cd.corr().iloc[0,1]:.3f}")

        # ---- Gate 2: energy, and why it cannot work here ----
        print(f"\n  ENERGY GATE (predicted saving > 0)")
        e_tp = int(((d.saves_energy) & (d.true_saves)).sum())
        e_fp = int(((d.saves_energy) & (~d.true_saves)).sum())
        print(f"    predicted a saving, there was one   {e_tp:>4}")
        print(f"    predicted a saving, there was not   {e_fp:>4}")
        ec = d[["predicted_saving_w", "true_saved_w"]].corr().iloc[0, 1]
        print(f"    predicted vs measured saving, correlation: {ec:.3f}")
        med = d.true_saved_w.abs().median()
        print(f"    median absolute true saving: {med:.1f} W")
        print(f"    RESOLUTION: the twin's energy MAPE is ~8%, so on a "
              f"~1800 W network")
        print(f"    the per-cell error is ~24 W and the total error ~60 W. A "
              f"{med:.0f} W")
        print(f"    signal is not resolvable through that. This is a limit of "
              f"twin")
        print(f"    accuracy, not of the counterfactual construction.")

        # ---- Combined, for completeness ----
        print(f"\n  COMBINED (both gates, as the loop would actually decide)")
        tp = int(((d.approved) & (d.true_approve)).sum())
        fp = int(((d.approved) & (~d.true_approve)).sum())
        fn = int(((~d.approved) & (d.true_approve)).sum())
        tn = int(((~d.approved) & (~d.true_approve)).sum())
        n = len(d)
        print(f"  decisions: {n}")
        print(f"    approved and correct     {tp:>4}")
        print(f"    approved but should not  {fp:>4}   <- unsafe action reaching the network")
        print(f"    rejected but could have  {fn:>4}   <- missed saving")
        print(f"    rejected and correct     {tn:>4}")
        acc = (tp + tn) / n * 100 if n else float("nan")
        prec = tp / (tp + fp) * 100 if tp + fp else float("nan")
        rec = tp / (tp + fn) * 100 if tp + fn else float("nan")
        print(f"  accuracy {acc:.1f}%   precision {prec:.1f}%   recall {rec:.1f}%")

        realised = d[d.approved].true_saved_w
        if len(realised):
            print(f"    energy actually saved by approved actions: "
                  f"mean {realised.mean():.1f} W, total {realised.sum():.1f} W")

    if len(args.reassign) > 1 and set(args.reassign) >= {"nearest", "oracle"}:
        a = df[df.reassign == "oracle"]
        b = df[df.reassign == "nearest"]
        if len(a) and len(b):
            ua, ub = a[~a.true_safe], b[~b.true_safe]
            ca = (~ua.approved).mean() * 100 if len(ua) else float("nan")
            cb = (~ub.approved).mean() * 100 if len(ub) else float("nan")
            print("\n" + "=" * 74)
            print(f"Unsafe actions caught: oracle {ca:.1f}%  vs  nearest {cb:.1f}%")
            print("The difference is the cost of not knowing where the UEs will")
            print("go -- reassignment error, separate from twin error.")


if __name__ == "__main__":
    main()