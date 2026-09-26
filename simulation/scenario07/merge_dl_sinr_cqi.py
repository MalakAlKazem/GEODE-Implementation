"""
merge_dl_sinr_cqi.py

Merges dl_sinr_features.csv and cqi_features.csv into ue_inputs.csv,
adding columns sinr_dl_db, cqi_ul, cqi_ul_filled. Additive and
idempotent: existing columns are untouched and re-running replaces
rather than duplicates.

Run in Ubuntu from the scenario07 folder, then copy ue_inputs.csv to
the Windows project folder.

Clipping sinr_dl_db at 40 dB
----------------------------
Same reason as UL SINR: 3GPP MCS selection saturates around 22-30 dB,
so 40 dB and 110 dB (the observed max) drive identical scheduling and
identical throughput. Leaving the tail in would skew mean/std
normalization and compress the range where SINR discriminates. The raw
dl_sinr_features.csv is preserved, so the clip is reversible.

Two caveats to carry into interpretation
----------------------------------------
1. sinr_dl_db means track tx power almost exactly (28 / 34 / 37 dB for
   p37 / p43 / p46, a 9 dB span matching the 9 dB power difference), so
   this feature partly re-encodes tx_power_dbm, which is already a gNB
   input. Its new information is the per-UE, per-snapshot variation,
   not the level.
2. cqi_ul is 34-38% forward-filled, because averageCqiUl is reported
   per transmission and quiet UEs produce no samples in some snapshots.
   It is also a quantized (1-15) function of SINR, which the model
   already has at full resolution. cqi_ul_filled marks imputed rows so
   the model can distinguish measured from stale, and so the
   imputation stays visible in the data rather than hidden.
"""

import glob
import os

import pandas as pd

CLIP_DL_DB = 40.0
NEW_COLS = ["sinr_dl_db", "cqi_ul", "cqi_ul_filled"]

for d in sorted(glob.glob("results/extracted_*")):
    ue_path = os.path.join(d, "ue_inputs.csv")
    dl_path = os.path.join(d, "dl_sinr_features.csv")
    cqi_path = os.path.join(d, "cqi_features.csv")

    if not (os.path.exists(dl_path) and os.path.exists(cqi_path)):
        print(f"{os.path.basename(d):<22} missing feature file, skipped")
        continue

    ue = pd.read_csv(ue_path)
    ue = ue.drop(columns=[c for c in NEW_COLS if c in ue.columns])
    n_before = len(ue)

    dl = pd.read_csv(dl_path)
    n_clipped = int((dl.sinr_dl_db > CLIP_DL_DB).sum())
    dl["sinr_dl_db"] = dl.sinr_dl_db.clip(upper=CLIP_DL_DB)

    cqi = pd.read_csv(cqi_path)

    merged = (ue.merge(dl, on=["snapshot_id", "ue_index"], how="left")
                .merge(cqi, on=["snapshot_id", "ue_index"], how="left"))

    if len(merged) != n_before:
        raise RuntimeError(
            f"{d}: row count changed {n_before} -> {len(merged)}. Duplicate "
            f"keys somewhere. NOT written.")
    nulls = {c: int(merged[c].isnull().sum()) for c in NEW_COLS}
    if any(nulls.values()):
        raise RuntimeError(f"{d}: nulls after merge {nulls}. NOT written.")

    merged.to_csv(ue_path, index=False)
    print(f"{os.path.basename(d):<22} {len(merged)} rows  "
          f"dl_clipped {n_clipped} ({n_clipped/len(dl)*100:.2f}%)  "
          f"dl_sinr mean {merged.sinr_dl_db.mean():.2f}  "
          f"cqi mean {merged.cqi_ul.mean():.2f}  "
          f"cqi stale {merged.cqi_ul_filled.mean()*100:.1f}%")
