"""
repair_kpi_grid_1s_s06c.py -- complete the (snapshot, UE) grid for the
1 s KPI files.

    python3 repair_kpi_grid_1s_s06c.py

Same purpose as repair_kpi_grid_s06c.py, but the gaps are larger and
arise for a different reason, so the handling differs in one respect.

At 3 s resolution a gap meant a UE sent nothing for a whole phase --
about 0.1% of rows. At 1 s resolution it means a UE sent nothing during
one particular second, which at Low-phase rates (~7 packets/s) happens
often: 5.5% to 8.3% of rows across the 25 runs.

Forward-fill is still the right choice over zero-fill -- a UE that
transmitted nothing has no measured latency, and its last observed value
is the best available stand-in, whereas a delay of 0 ms would be a false
measurement that drags the mean down. But at 8% the imputation is no
longer negligible, so:

  - every imputed row is flagged in kpi_filled, as before
  - n_packets is set to 0 on imputed rows, so they can be excluded or
    down-weighted by that column alone
  - the per-run imputation share is printed, not just a total

Whether to train on imputed rows or drop them is a decision for the
training script, not this one. This script makes the grid complete and
makes the imputation visible.
"""

import glob
import os

import numpy as np
import pandas as pd

NUM_UES = 60
COLS = ["delay_ms", "jitter_ms", "packet_loss"]
IN_NAME = "kpi_targets_1s.csv"

total_rows = total_filled = 0
print(f"{'run':<34} {'rows':>7} {'filled':>8} {'share':>7}  {'delay mean':>11}")
print("-" * 74)

for d in sorted(glob.glob("results/extracted_s06c*_p*")):
    path = os.path.join(d, IN_NAME)
    if not os.path.exists(path):
        continue
    k = pd.read_csv(path)
    for c in ("kpi_filled",):
        if c in k.columns:
            k = k.drop(columns=[c])

    snaps = sorted(pd.read_csv(os.path.join(d, "gnb_inputs.csv"))
                   .snapshot_id.unique())
    full = pd.MultiIndex.from_product([snaps, range(NUM_UES)],
                                      names=["snapshot_id", "ue_index"])
    k = k.set_index(["snapshot_id", "ue_index"]).reindex(full)

    missing = k[COLS[0]].isnull()
    n_missing = int(missing.sum())

    for c in COLS:
        k[c] = (k[c].groupby(level="ue_index").ffill()
                    .groupby(level="ue_index").bfill())

    # kpi_window_id belongs to the snapshot, not the measurement, so it is
    # recomputed rather than carried across from a neighbouring row.
    k["kpi_window_id"] = (k.index.get_level_values("snapshot_id")
                          * 0.1 // 1.0).astype(int)

    # n_packets = 0 marks a row with no measurement behind it, which is a
    # stronger and more usable signal than the flag alone.
    if "n_packets" not in k.columns:
        k["n_packets"] = np.nan
    k["n_packets"] = k["n_packets"].fillna(0).astype(int)
    k.loc[missing, "n_packets"] = 0

    k["kpi_filled"] = missing.astype(int).values
    k = k.reset_index()

    if k[COLS].isnull().any().any():
        raise RuntimeError(f"{d}: nulls remain after fill -- a UE has no KPI "
                           f"data in any window. NOT written.")

    k.to_csv(path, index=False)
    total_rows += len(k)
    total_filled += n_missing
    print(f"{os.path.basename(d):<34} {len(k):>7} {n_missing:>8} "
          f"{n_missing/len(k)*100:>6.2f}%  {k.delay_ms.mean():>10.1f}ms")

print("-" * 74)
print(f"{'TOTAL':<34} {total_rows:>7} {total_filled:>8} "
      f"{total_filled/total_rows*100:>6.2f}%")
print(f"\nImputed rows carry kpi_filled=1 and n_packets=0. At this share the")
print(f"imputation is not negligible -- consider excluding these rows from")
print(f"the KPI loss rather than training on forward-filled targets.")
