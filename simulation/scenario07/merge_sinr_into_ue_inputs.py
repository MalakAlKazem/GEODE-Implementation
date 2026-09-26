"""
merge_sinr_into_ue_inputs.py

Adds sinr_db from sinr_features.csv into ue_inputs.csv as a UE input
feature. Additive: existing columns untouched, so every script that
reads ue_inputs.csv keeps working.

Clipping at 40 dB. Rationale, not convenience: 3GPP MCS selection
saturates around 22-30 dB SINR -- the highest modulation/coding scheme
is already chosen well below 40 dB, so 40 dB and 90 dB give identical
scheduling behaviour and identical throughput. The difference carries
no actionable information, while an unclipped tail (max observed 96 dB
against a median of 18) would skew mean/std normalization and compress
the range where SINR genuinely matters, roughly -10 to 30 dB.

2.5-5% of samples exceed 40 dB. These are real, not a bug: during
L-phases (sendInterval exponential(0.15s)) most of the 50 RBs are idle,
so a UE transmitting with no co-channel interference is noise-limited
and its SINR can be very large.

Raw sinr_features.csv is left intact, so the clip is reversible.
"""

import glob
import os

import pandas as pd

CLIP_DB = 40.0

for d in sorted(glob.glob("results/extracted_*")):
    ue_path = os.path.join(d, "ue_inputs.csv")
    sinr_path = os.path.join(d, "sinr_features.csv")
    if not os.path.exists(sinr_path):
        print(f"{os.path.basename(d):<22} no sinr_features.csv, skipped")
        continue

    ue = pd.read_csv(ue_path)
    sinr = pd.read_csv(sinr_path)

    if "sinr_db" in ue.columns:
        ue = ue.drop(columns=["sinr_db"])   # idempotent re-run

    n_before = len(ue)
    n_clipped = int((sinr.sinr_db > CLIP_DB).sum())
    sinr["sinr_db"] = sinr.sinr_db.clip(upper=CLIP_DB)

    merged = ue.merge(sinr, on=["snapshot_id", "ue_index"], how="left")

    if len(merged) != n_before:
        raise RuntimeError(
            f"{d}: row count changed {n_before} -> {len(merged)}. Duplicate "
            f"keys in one of the files. NOT written.")
    n_null = int(merged.sinr_db.isnull().sum())
    if n_null:
        raise RuntimeError(
            f"{d}: {n_null} rows got no SINR value. Missing (snapshot,ue) "
            f"pairs. NOT written -- investigate before training.")

    merged.to_csv(ue_path, index=False)
    print(f"{os.path.basename(d):<22} {len(merged)} rows, "
          f"clipped {n_clipped} ({n_clipped/len(sinr)*100:.2f}%), "
          f"sinr mean {merged.sinr_db.mean():.2f} "
          f"min {merged.sinr_db.min():.2f} max {merged.sinr_db.max():.2f}")
