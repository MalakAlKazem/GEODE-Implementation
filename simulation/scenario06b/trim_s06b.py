"""
trim_s06b.py -- removes each seed's handover-settling window.

Per-seed windows measured from that seed's OWN per-phase handover
counts, never copied between seeds (the scenario07 lesson: windows
ranged 300-510 snapshots there, and applying one constant would have
left churn in the training data).

  s51: 6 phases (peak 17 handovers at ph2, <=5 by ph5)
  s52: 6 phases (peak 16 at ph2, settled by ph5)
  s53: 8 phases (peak 23 at ph3, still 8-9 at ph5)

Delay could NOT be used as the settling signal here, unlike scenario07:
scenario06b saturates during most traffic phases, so congestion delay
(200-2000ms) swamps any handover-settling signature. Handover count is
the only clean indicator.

Trims all 7 per-run files, including sinr_features.csv and the
kpi_filled flag column added by repair_kpi_grid_s06b.py.
"""

import os

import pandas as pd

TRIMS = {51: 180, 52: 180, 53: 240}
RUNS = ["p37", "p43", "p46"]
FILES = ["gnb_inputs.csv", "ue_inputs.csv", "serving_edges.csv",
         "energy_targets.csv", "energy_diagnostics.csv",
         "kpi_targets.csv", "sinr_features.csv"]

for seed, cut in TRIMS.items():
    print(f"=== seed {seed}: removing first {cut} snapshots ({cut//30} phases) ===")
    for run in RUNS:
        d = f"results/extracted_s06b{seed}_{run}"
        if not os.path.isdir(d):
            print(f"  {run}: missing, skipped")
            continue
        for fname in FILES:
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                print(f"  {run}/{fname}: MISSING")
                continue
            df = pd.read_csv(p)
            before = len(df)
            df = df[df.snapshot_id >= cut].copy()
            df["snapshot_id"] -= cut
            df.to_csv(p, index=False)
            print(f"  {run}/{fname}: {before} -> {len(df)}")
    print()
