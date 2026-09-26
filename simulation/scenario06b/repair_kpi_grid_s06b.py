"""
repair_kpi_grid_s06b.py

Some (UE, phase) combos have no delay/jitter/loss because that UE sent
zero packets during that 3s phase -- the app records no endToEndDelay
vector at all, so the extractor dropped the row rather than inventing a
zero. That leaves kpi_targets.csv short of the full 1200x60 grid (worst
case 240 rows = 8 combos = 0.33%), which the dataset builder assumes.

Fix: reindex to the complete grid and forward-fill per UE, then
back-fill any leading gap. Rationale: a UE that transmitted nothing has
no measured latency, and its last observed value is the best available
stand-in -- the same convention this pipeline already uses for
serving_cell (extract_scenario07.py) and cqi_ul. A `kpi_filled` flag
marks every imputed row so the choice stays visible in the data and can
be excluded from evaluation later if wanted.

Not zero-filled: delay 0 would be a false measurement and would drag
the mean down; 'unknown, assume unchanged' is the honest reading.
"""

import glob
import os

import pandas as pd

NUM_UES = 60
COLS = ["delay_ms", "jitter_ms", "packet_loss"]

for d in sorted(glob.glob("results/extracted_s06b*_p*")):
    path = os.path.join(d, "kpi_targets.csv")
    k = pd.read_csv(path)
    if "kpi_filled" in k.columns:
        k = k.drop(columns=["kpi_filled"])

    snaps = sorted(pd.read_csv(os.path.join(d, "gnb_inputs.csv")).snapshot_id.unique())
    full = pd.MultiIndex.from_product([snaps, range(NUM_UES)],
                                      names=["snapshot_id", "ue_index"])

    k = k.set_index(["snapshot_id", "ue_index"]).reindex(full)
    missing = k[COLS[0]].isnull()
    n_missing = int(missing.sum())

    for c in COLS:
        k[c] = (k[c].groupby(level="ue_index").ffill()
                    .groupby(level="ue_index").bfill())

    k["kpi_filled"] = missing.astype(int).values
    k = k.reset_index()

    if k[COLS].isnull().any().any():
        raise RuntimeError(f"{d}: nulls remain after fill -- a UE has no KPI "
                           f"data in any phase. NOT written.")

    k.to_csv(path, index=False)
    print(f"{os.path.basename(d):<24} {len(k)} rows  filled {n_missing} "
          f"({n_missing/len(k)*100:.2f}%)  delay mean {k.delay_ms.mean():.1f}ms")
