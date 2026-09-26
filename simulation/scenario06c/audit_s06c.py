"""
audit_s06c.py -- data quality audit for scenario06c.

    python3 audit_s06c.py

Checks every extracted run for the failure modes that would corrupt
training silently rather than raising an error:

  1. nulls, NaN, inf in any column used as a feature or label
  2. index completeness -- every snapshot must have all 6 gNBs and all
     60 UEs, or a graph will be built with missing nodes and the label
     tensor will not line up with the feature tensor
  3. serving_gnb_index inside 0..NUM_GNB-1, and every UE served
  4. qos_class restricted to the three expected strings, since the
     one-hot silently produces all-zeros for an unrecognised value
  5. constant columns -- zero variance means a dead input the
     normaliser will divide by a floored std of 1.0
  6. physical plausibility bounds
  7. snapshot-id agreement across the five files of a run
  8. duplicate (snapshot_id, index) keys
  9. kpi_filled share, so imputed rows stay visible

Nothing here modifies data. It reports.
"""

import glob
import os
import sys
import numpy as np
import pandas as pd

NUM_GNB = 6
NUM_UE = 60
QOS = {"eMBB", "URLLC", "mMTC"}

# (column, low, high, note) -- generous bounds; the point is to catch
# corruption and unit mistakes, not to second-guess the simulator.
BOUNDS = {
    "ue_inputs.csv": [
        ("pos_x", -1.0, 1600.0, "map is 1500 m wide"),
        ("pos_y", -1.0, 1100.0, "map is 1000 m tall"),
        ("speed_mps", 0.0, 30.0, "mobility classes top out at 25"),
        ("offered_ul_bps", 0.0, 1e9, ""),
        ("sinr_db", -60.0, 45.0, "merge clips at 40"),
    ],
    "gnb_inputs.csv": [
        ("pos_x", -1.0, 1600.0, ""),
        ("pos_y", -1.0, 1100.0, ""),
        ("tx_power_dbm", 0.0, 60.0, ""),
        ("num_connected_ues", 0, NUM_UE, ""),
        ("offered_ul_bps", 0.0, 1e10, ""),
        ("num_bands", 1, 500, ""),
    ],
    "kpi_targets.csv": [
        ("delay_ms", 0.0, 1e5, ""),
        ("jitter_ms", 0.0, 1e5, ""),
        ("packet_loss", 0.0, 1.0, "a fraction, not a percentage"),
    ],
    "energy_targets.csv": [
        ("estimated_gnb_power_w", 0.0, 5000.0, ""),
    ],
}

FILES = ["gnb_inputs.csv", "ue_inputs.csv", "kpi_targets.csv",
         "energy_targets.csv", "serving_edges.csv"]

problems = []
notes = []


def flag(run, msg):
    problems.append(f"{run}: {msg}")


def note(run, msg):
    notes.append(f"{run}: {msg}")


def audit_run(folder):
    run = os.path.basename(folder.rstrip("/\\"))
    d = {}
    for f in FILES:
        p = os.path.join(folder, f)
        if not os.path.exists(p):
            flag(run, f"MISSING FILE {f}")
            return
        d[f] = pd.read_csv(p)

    # ---- 1. nulls / non-finite --------------------------------------------
    for f, df in d.items():
        n = df.isnull().sum()
        bad = n[n > 0]
        if len(bad):
            flag(run, f"{f} nulls: {bad.to_dict()}")
        for c in df.select_dtypes(include=[np.number]).columns:
            v = df[c].to_numpy()
            if not np.isfinite(v).all():
                flag(run, f"{f}.{c} has {int((~np.isfinite(v)).sum())} "
                          f"non-finite values")

    # ---- 7. snapshot agreement -------------------------------------------
    snaps = {f: set(df.snapshot_id.unique()) for f, df in d.items()}
    ref = snaps["gnb_inputs.csv"]
    for f, s in snaps.items():
        if s != ref:
            flag(run, f"{f} snapshot set differs from gnb_inputs "
                      f"(only-here {len(s - ref)}, missing {len(ref - s)})")

    # ---- 2. index completeness -------------------------------------------
    g_per = d["gnb_inputs.csv"].groupby("snapshot_id").gnb_index.nunique()
    if not (g_per == NUM_GNB).all():
        flag(run, f"gnb count per snapshot not always {NUM_GNB}: "
                  f"min {int(g_per.min())} max {int(g_per.max())}")
    u_per = d["ue_inputs.csv"].groupby("snapshot_id").ue_index.nunique()
    if not (u_per == NUM_UE).all():
        flag(run, f"UE count per snapshot not always {NUM_UE}: "
                  f"min {int(u_per.min())} max {int(u_per.max())}")
    k_per = d["kpi_targets.csv"].groupby("snapshot_id").ue_index.nunique()
    if not (k_per == NUM_UE).all():
        flag(run, f"KPI rows per snapshot not always {NUM_UE}: "
                  f"min {int(k_per.min())}")
    e_per = d["energy_targets.csv"].groupby("snapshot_id").gnb_index.nunique()
    if not (e_per == NUM_GNB).all():
        flag(run, f"energy rows per snapshot not always {NUM_GNB}")

    # ---- 8. duplicate keys ------------------------------------------------
    for f, keys in (("gnb_inputs.csv", ["snapshot_id", "gnb_index"]),
                    ("ue_inputs.csv", ["snapshot_id", "ue_index"]),
                    ("kpi_targets.csv", ["snapshot_id", "ue_index"]),
                    ("energy_targets.csv", ["snapshot_id", "gnb_index"]),
                    ("serving_edges.csv", ["snapshot_id", "ue_index"])):
        n = d[f].duplicated(subset=keys).sum()
        if n:
            flag(run, f"{f} has {n} duplicate {keys}")

    # ---- 3. serving validity ---------------------------------------------
    se = d["serving_edges.csv"]
    sv = se.serving_gnb_index
    if sv.isnull().any():
        flag(run, f"serving_edges has {int(sv.isnull().sum())} null serving cells")
    finite = sv.dropna()
    if len(finite) and (finite.min() < 0 or finite.max() > NUM_GNB - 1):
        flag(run, f"serving_gnb_index out of range "
                  f"[{finite.min()}, {finite.max()}]")
    srv_per = se.groupby("snapshot_id").ue_index.nunique()
    if not (srv_per == NUM_UE).all():
        note(run, f"not every UE has a serving cell in every snapshot "
                  f"(min {int(srv_per.min())}/{NUM_UE}) -- expected in sleep runs")

    # ---- 4. qos_class ----------------------------------------------------
    vals = set(d["ue_inputs.csv"].qos_class.unique())
    if not vals <= QOS:
        flag(run, f"unexpected qos_class values: {vals - QOS}")

    # ---- 5. constant columns ---------------------------------------------
    for f in ("gnb_inputs.csv", "ue_inputs.csv"):
        for c in d[f].select_dtypes(include=[np.number]).columns:
            if c in ("snapshot_id", "gnb_index", "ue_index"):
                continue
            if d[f][c].nunique() == 1:
                note(run, f"{f}.{c} is constant ({d[f][c].iloc[0]}) "
                          f"-- dead input if used as a feature")

    # ---- 6. bounds -------------------------------------------------------
    for f, checks in BOUNDS.items():
        for c, lo, hi, why in checks:
            if c not in d[f].columns:
                flag(run, f"{f} missing expected column {c}")
                continue
            v = d[f][c].to_numpy()
            out = ((v < lo) | (v > hi)).sum()
            if out:
                flag(run, f"{f}.{c}: {int(out)} values outside [{lo}, {hi}]"
                          + (f" ({why})" if why else "")
                          + f" -- observed [{np.nanmin(v):.3g}, {np.nanmax(v):.3g}]")

    # ---- 9. imputation share ---------------------------------------------
    if "kpi_filled" in d["kpi_targets.csv"].columns:
        s = d["kpi_targets.csv"].kpi_filled.mean() * 100
        if s > 0:
            note(run, f"kpi_filled {s:.2f}% of rows imputed")

    return len(d["gnb_inputs.csv"].snapshot_id.unique())


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results"
    folders = sorted(glob.glob(os.path.join(base, "extracted_s06c*")))
    if not folders:
        print(f"no extracted_s06c* folders under {base}")
        return
    print(f"auditing {len(folders)} runs under {base}\n")
    for f in folders:
        n = audit_run(f)
        if n:
            print(f"  {os.path.basename(f):<34} {n} snapshots")

    print("\n" + "=" * 78)
    if problems:
        print(f"PROBLEMS ({len(problems)}) -- these would corrupt training")
        print("=" * 78)
        for p in problems:
            print("  ! " + p)
    else:
        print("NO PROBLEMS FOUND")
    print()
    if notes:
        print(f"NOTES ({len(notes)}) -- expected or benign, listed for the record")
        print("-" * 78)
        seen = set()
        for n in notes:
            key = n.split(":", 1)[1]
            if key in seen:
                continue
            seen.add(key)
            print("  - " + n + ("  [and others]" if notes.count(n) > 1 else ""))
        print(f"\n  ({len(notes)} notes total, deduplicated above)")


if __name__ == "__main__":
    main()
