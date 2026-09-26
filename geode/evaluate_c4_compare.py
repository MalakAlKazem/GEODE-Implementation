"""
evaluate_c4_compare.py -- does the graph help predict REAL measured
cell-sleep effects better than a blind (no-message-passing) control?

Motivation
----------
The ordinary held-out-snapshot comparison (train_wdt_s06c.py's own
R2 table) tests "predict another ordinary snapshot." Once a lag
feature is available that comparison is dominated by inertia (this
instant looks like the last one) -- not the regime a digital twin
actually has to work in for C4, where it must predict the effect of
an action (a cell going to sleep) it has never seen play out, on
NEIGHBOURING cells whose own inputs changed because of that action.
evaluate_c4.py already has exactly this test built, against 640 real
paired sleep-run measurements (not a synthetic counterfactual) -- it
was just never run gnn-vs-blind head to head, or with any temporal
feature. This script does both, split into two LEAKAGE-CLEAN gates.

Why two separate checkpoint sets, not one
------------------------------------------
06c's delay_ms/jitter_ms/packet_loss labels are broadcast across each
3-second phase's 30 snapshots (see wdt_data.py's KPI_FILE comment) --
confirmed empirically: 0/2400 phases have more than one distinct
delay value. A "previous-snapshot own delay" lag feature is therefore
~equal to the current label for 96.7% of rows: near-total leakage,
not a real temporal signal, for those three targets. energy_targets.csv
has no such broadcast (7.9% exact-repeat rate -- genuinely fresh each
snapshot), so lag_energy_w is legitimate.

So:
  SAFETY GATE (delay/jitter d95 ratio) -- uses the "nolag_recover"
    checkpoints (no lag feature anywhere). Any lag feature here would
    be near-total leakage and the comparison would be meaningless.
  ENERGY GATE (real measured saving_w) -- uses the "energy_lag_only"
    checkpoints (gnb gets its own previous energy_W; UE gets NOTHING).
    Deliberately not the earlier "full lag" checkpoints, which also
    carry the leaky UE-side lag_delay_ms/lag_jitter_ms -- for gnn
    specifically (not blind, which never aggregates across nodes)
    those could reach the energy prediction via ue->gnb message
    passing. This config has zero leakage pathway anywhere.

Lag values at evaluation time: no oracle
------------------------------------------
For both gates, every node's lag input (baseline graph AND
counterfactual/sleep graph) is that node's own REAL value from the
BASELINE (active) run's previous snapshot -- never from the sleep
run's own history. This matches c4_cmoa.py's own "nearest" vs
"oracle" distinction: a real decision-time system only has real
already-observed history from BEFORE the hypothetical action, never
a peek at what its own reading would be once already asleep. This
does mean the sleeping cell's own post-sleep energy prediction is
mildly out-of-distribution (training's sleep-run snapshots are
steady-state-already-asleep for their whole duration, so the model
never saw "lag=recent-active, current=asleep" during training) --
a genuine, reportable limit of the lag feature under counterfactual
reasoning, not something hidden here.

Per-seed checkpoint discipline (leak prevention)
--------------------------------------------------
Each rotation's checkpoint (test_seed=S) was trained on the OTHER
two seeds -- including their sleep-run data. Evaluating it on seed
S's own sleep events is the only leak-free combination; pooling one
checkpoint across all 3 seeds (as a naive `--seeds 61 62 63` run
against a single checkpoint would do) would test 2/3 of the seeds on
data the model trained on. This script always matches checkpoint
test_seed to the seed being evaluated.

    python3 evaluate_c4_compare.py
"""

import os

import numpy as np
import pandas as pd
import torch

from c1_graph import build_interference_edges
from c4_cmoa import CMOA
from wdt_checkpoint import load_checkpoint
from evaluate_c4 import read_run, ground_truth, sanity_check, BASE, SLEEP, PHASE, D95_UNSAFE

SEEDS = (61, 62, 63)


def add_energy_lag_to_run(base):
    """Adds lag_energy_w to EVERY row of base["gnb"] in place (not just
    the phase-representative snapshots the main loop samples) --
    lag_energy_w = that gNB's own real estimated_gnb_power_w one raw
    snapshot_id earlier, in this same BASELINE (active) run. A run's
    first snapshot has no earlier reading and falls back to its own
    CURRENT reading ("assume no change", not a fabricated number).

    Whole-run, not per-phase-slice: sanity_check() (called from
    run_gate) independently re-slices base["gnb"] at its own chosen
    snapshot, bypassing any per-slice augmentation -- augmenting the
    full dataframe once, before any slicing happens, means every
    downstream slice (this script's own loop AND sanity_check's)
    already has the column, with no special-casing needed at the call
    site. Mutates and returns base["gnb"]; base["energy"] is untouched.
    """
    e = base["energy"].sort_values(["gnb_index", "snapshot_id"])
    e = e.assign(lag_energy_w=e.groupby("gnb_index")
                 .estimated_gnb_power_w.shift(1))
    e["lag_energy_w"] = e["lag_energy_w"].fillna(e["estimated_gnb_power_w"])
    base["gnb"] = base["gnb"].merge(
        e[["snapshot_id", "gnb_index", "lag_energy_w"]],
        on=["snapshot_id", "gnb_index"], how="left")
    missing = base["gnb"]["lag_energy_w"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} gnb rows got no lag_energy_w at all -- "
            f"energy_targets.csv is missing rows gnb_inputs.csv has.")
    return base["gnb"]


def run_gate(kind, tag, seeds=SEEDS, reassign=("nearest", "oracle"),
             use_energy_lag=False, tolerance=0.05):
    """One (kind, tag) checkpoint set, evaluated leak-free per seed
    (checkpoint test_seed always matches the seed being evaluated),
    pooled across all seeds. Returns the pooled results DataFrame.
    """
    rows = []
    for test_seed in seeds:
        ckpt_path = f"checkpoints/wdt_{tag}_{kind}_test{test_seed}_init1.pt"
        model, ctx = load_checkpoint(ckpt_path)
        gnb_norm, ue_norm = ctx["gnb_norm"], ctx["ue_norm"]
        cmoa = CMOA(model, ctx, violation_tolerance=tolerance)

        base = read_run(os.path.join(BASE, f"extracted_s06c{test_seed}_p43"))
        if use_energy_lag:
            add_energy_lag_to_run(base)  # mutates base["gnb"] in place,
            # BEFORE any slicing -- so every downstream slice (this
            # loop's own gr AND sanity_check's independent slice) already
            # carries lag_energy_w, with no per-call-site special-casing.
        snaps = np.sort(base["gnb"].snapshot_id.unique())
        snap0 = base["gnb"][base["gnb"].snapshot_id == snaps[0]]
        interference = build_interference_edges(snap0)

        for cell in SLEEP[test_seed]:
            sleep = read_run(os.path.join(
                BASE, f"extracted_s06c{test_seed}_p43_sleep{cell}"))
            gt = ground_truth(base, sleep)

            if test_seed == seeds[0] and cell == SLEEP[test_seed][0]:
                sanity_check(cmoa, base, sleep, cell, snaps, interference,
                             gnb_norm, ue_norm)

            for ph, truth in gt.items():
                sid = int(snaps[min(ph * PHASE + PHASE // 2, len(snaps) - 1)])
                gr = base["gnb"][base["gnb"].snapshot_id == sid]
                ur = base["ue"][base["ue"].snapshot_id == sid]
                sr = base["serving"][base["serving"].snapshot_id == sid]
                if gr.empty or ur.empty:
                    continue
                osr = sleep["serving"][sleep["serving"].snapshot_id == sid]

                for mode in reassign:
                    if mode == "oracle" and osr.empty:
                        continue
                    d = cmoa.evaluate_action(
                        cell, gr, ur, sr, gnb_norm, ue_norm, interference,
                        reassign=mode, oracle_serving=osr)
                    d.update(kind=kind, tag=tag, seed=test_seed, phase=ph,
                             true_saved_w=truth["saved_w"],
                             true_d95_ratio=truth["d95_ratio"],
                             true_safe=truth["d95_ratio"] < D95_UNSAFE,
                             true_saves=truth["saved_w"] > 0)
                    d["true_approve"] = bool(d["true_safe"] and d["true_saves"])
                    rows.append(d)
        print(f"    {kind:<6} test_seed {test_seed}: "
              f"{len([r for r in rows if r['seed'] == test_seed])} rows")

    return pd.DataFrame(rows)


def safety_summary(label, d):
    """d95 correlation + unsafe-action recall, for one model's pooled
    results on one reassignment mode."""
    cd = d[["pred_d95_ratio", "true_d95_ratio"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    corr = cd.corr().iloc[0, 1] if len(cd) > 2 else float("nan")
    n_unsafe = int((~d.true_safe).sum())
    caught = int(((~d.safe) & (~d.true_safe)).sum())
    recall = caught / n_unsafe * 100 if n_unsafe else float("nan")
    fp = int(((d.safe) & (~d.true_safe)).sum())  # unsafe action passed
    print(f"    {label:<12} d95 corr {corr:+.3f}   "
          f"unsafe caught {caught}/{n_unsafe} ({recall:.1f}%)   "
          f"unsafe-passed(FP) {fp}")
    return {"corr": corr, "recall": recall, "fp": fp}


def energy_summary(label, d):
    ec = d[["predicted_saving_w", "true_saved_w"]].corr().iloc[0, 1]
    mae = (d.predicted_saving_w - d.true_saved_w).abs().mean()
    print(f"    {label:<12} saving corr {ec:+.3f}   "
          f"MAE {mae:.1f} W   median|true| {d.true_saved_w.abs().median():.1f} W")
    return {"corr": ec, "mae": mae}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety-tag", default="nolag_recover",
                    help="checkpoint tag for the no-lag safety-gate models")
    ap.add_argument("--energy-tag", default="energy_lag_only",
                    help="checkpoint tag for the energy-lag energy-gate models")
    ap.add_argument("--out-prefix", default="c4_compare",
                    help="prefix for the written CSVs, so runs against "
                         "different architectures don't overwrite each other")
    args = ap.parse_args()

    print("=" * 78)
    print(f"SAFETY GATE (delay/jitter d95 ratio) -- {args.safety_tag} checkpoints")
    print("  (no lag feature anywhere -- delay/jitter lag would be leakage,")
    print("   see module docstring)")
    print("=" * 78)
    safety = {}
    for kind in ("gnn", "blind"):
        print(f"\n  loading {kind}...")
        safety[kind] = run_gate(kind, args.safety_tag, use_energy_lag=False)

    for mode in ("nearest", "oracle"):
        print(f"\n  -- reassignment: {mode} --")
        res = {}
        for kind in ("gnn", "blind"):
            d = safety[kind][safety[kind].reassign == mode]
            if len(d):
                res[kind] = safety_summary(kind, d)
        if "gnn" in res and "blind" in res:
            print(f"    -> d95 corr:   gnn {res['gnn']['corr']:+.3f}  vs  "
                  f"blind {res['blind']['corr']:+.3f}   "
                  f"(higher = better tracks real neighbour QoS shift)")
            print(f"    -> unsafe recall: gnn {res['gnn']['recall']:.1f}%  vs  "
                  f"blind {res['blind']['recall']:.1f}%")

    print("\n" + "=" * 78)
    print(f"ENERGY GATE (real measured saving_w) -- {args.energy_tag} checkpoints")
    print("  (gnb: own prev energy_W only; ue: nothing -- zero leakage path)")
    print("=" * 78)
    energy = {}
    for kind in ("gnn", "blind"):
        print(f"\n  loading {kind}...")
        energy[kind] = run_gate(kind, args.energy_tag, use_energy_lag=True)

    for mode in ("nearest", "oracle"):
        print(f"\n  -- reassignment: {mode} --")
        res = {}
        for kind in ("gnn", "blind"):
            d = energy[kind][energy[kind].reassign == mode]
            if len(d):
                res[kind] = energy_summary(kind, d)
        if "gnn" in res and "blind" in res:
            print(f"    -> saving corr:  gnn {res['gnn']['corr']:+.3f}  vs  "
                  f"blind {res['blind']['corr']:+.3f}")

    for kind, df in safety.items():
        df.to_csv(f"{args.out_prefix}_safety_{kind}.csv", index=False)
    for kind, df in energy.items():
        df.to_csv(f"{args.out_prefix}_energy_{kind}.csv", index=False)
    print(f"\nwrote {args.out_prefix}_safety_" + "{gnn,blind}.csv, "
          f"{args.out_prefix}_energy_" + "{gnn,blind}.csv")


if __name__ == "__main__":
    main()
