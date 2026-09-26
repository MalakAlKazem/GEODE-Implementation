"""
extract_sinr_scenario07.py

Extracts per-UE uplink SINR from the existing `.vec` files as a new
GNN INPUT feature. No re-simulation needed -- `**.vector-recording =
true` was already set, so every run recorded this.

Usage:  SEED=27 python3 extract_sinr_scenario07.py
        SEED=21 python3 extract_sinr_scenario07.py   (original folders)

Which statistic, and the trap to avoid
--------------------------------------
TWO different modules publish a statistic named `measuredSinrUl:vector`:
    ue[N].cellularNic.channelModel[0]     <- LTE stack, count=0, EMPTY
    ue[N].cellularNic.nrChannelModel[0]   <- 5G NR stack, count=20000
Filtering on the name alone silently returns nothing usable. Same class
of trap as the SCTP `endToEndDelay:vector` collision found earlier in
this project. This script filters on `nrChannelModel` explicitly and
raises if the row count looks wrong rather than writing a bad CSV.

UL, not DL: our KPI targets are uplink, and UL SINR carries real
variance (per-UE means 11-27 dB, individual samples down to -4.75 dB)
whereas DL sits uniformly high (~35 dB mean, min 10.8) -- consistent
with the earlier finding that DL signal quality in this topology is too
good to be informative.

Granularity -- genuinely per-snapshot, unlike the KPI labels
------------------------------------------------------------
20,000 samples over 120s is one every ~6ms, so ~16 samples land inside
each 100ms snapshot. This aggregates to a REAL per-snapshot value.
Contrast with delay/jitter/packet_loss, which are one value per 3s
traffic phase broadcast across 30 identical snapshots. This is the
first UE input feature besides position that actually varies every
snapshot.

Why this is a legitimate input and not leakage
----------------------------------------------
SINR is a function of geometry, tx power, and interference -- not of
`load_used`, which is what the energy label is computed from. For the
KPI targets it is causally upstream (poor SINR -> retransmissions ->
delay), and it is exactly what a real network measures and reports.
So it belongs on the input side under the same non-circularity rule
this pipeline has followed throughout.

Honest limitation to record: under a CMOA counterfactual ("what if I
sleep gNB k?"), real SINR would change in ways the model cannot infer
from a measured value. This feature is sound for prediction and adds
an assumption to the what-if loop. Worth stating explicitly rather
than discovering later.

Trim alignment
--------------
The `.vec` covers the full 0-120s. The extracted CSVs were trimmed by a
per-seed amount (measured from each seed's own handover/delay table,
never assumed). SINR snapshots are shifted by the same amount and
negative indices dropped, so `snapshot_id` lines up with the other
CSVs. TRIMS below must match what `trim_sNN.py` actually applied.

Output: `sinr_features.csv` with columns snapshot_id, ue_index, sinr_db
-- the schema `data_loader.py` already looks for.
"""

import os
import re
import sqlite3
import subprocess
import tempfile

import pandas as pd

SEED = int(os.environ.get("SEED", "61"))
NUM_UES = 60      # scenario06b
WINDOW_S = 0.1
STAT = "measuredSinrUl:vector"

# Snapshots removed from the front of each seed's CSVs. Must match the
# trim actually applied by trim_sNN.py, or SINR will be misaligned
# against every other file by the difference.
# scenario06b is extracted UNTRIMMED, so no offset here. Trim is applied
# later by trim_s06b.py, per seed, from its own measured settling window.


# Repetition that completed per run (see s06b_successful_runs.txt).
REPS = {
    (61, "p37") : 0,
    (61, "p43"): 0, (61, "p46"): 0,
    (61, "p43_sleep0"): 0,
    (61, "p43_sleep1"): 0,
    (61, "p43_sleep2"): 0,
    (61, "p43_sleep3"): 0,
    (61, "p43_sleep4"): 0,
    (61, "p43_sleep5"): 0,
    (62, "p37") : 0,
    (62, "p43"): 0, (62, "p46"): 0,
    (62, "p43_sleep0"): 0,
    (62, "p43_sleep1"): 0,
    (62, "p43_sleep2"): 0,
    (62, "p43_sleep5"): 0,
    (63, "p37") : 0,
    (63, "p43"): 0, (63, "p46"): 0,
    (63, "p43_sleep0"): 0,
    (63, "p43_sleep1"): 0,
    (63, "p43_sleep2"): 0,
    (63, "p43_sleep3"): 0,
    (63, "p43_sleep4"): 0,
    (63, "p43_sleep5"): 0,
    # s62 sleep3/sleep4 never completed: deterministic NrMacGnb
    # handover crash, reproduced identically over r=0..9.
}

TRIMS = {s: 0 for (s, _) in REPS}   # no settling window in 06c

# Seed 26's p43 run crashed (Simu5G NrMacGnb handover race, deterministic
# at t=11.05s) and was dropped.
POWERS = {}
for (_s, _r) in REPS:
    POWERS.setdefault(_s, []).append(_r)
POWERS = {k: tuple(v) for k, v in POWERS.items()}


def config_name(seed: int, power: str) -> str:
    return f"Scenario06Cs{seed}_{power}"


def out_dir(seed: int, power: str) -> str:
    return f"results/extracted_s06c{seed}_{power}"


def run_scavetool(args: list) -> None:
    r = subprocess.run(["opp_scavetool"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"opp_scavetool failed: {' '.join(args)}\n{r.stderr}")


def extract_one(vec_file: str, trim: int) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        sqlite_path = tmp.name

    try:
        run_scavetool([
            "export", "-F", "SqliteVectorFile", "-o", sqlite_path,
            "-f", f"name =~ {STAT}", vec_file,
        ])
        conn = sqlite3.connect(sqlite_path)
        df = pd.read_sql(
            """
            SELECT vector.moduleName AS moduleName,
                   vectorData.simtimeRaw / 1e12 AS time_s,
                   vectorData.value AS sinr_db
            FROM vectorData
            JOIN vector ON vectorData.vectorId = vector.vectorId
            """,
            conn,
        )
        conn.close()
    finally:
        os.remove(sqlite_path)

    # Keep ONLY UE-side NR-stack rows. Two separate filters are needed:
    #   - `nrChannelModel`: the LTE `channelModel` publishes the same stat
    #     name with zero samples (the trap this script exists to avoid).
    #   - `.ue[`: gNB modules ALSO carry nrChannelModel, and their module
    #     names contain no `ue[N]`, so leaving them in makes the ue_index
    #     extraction produce NaN and abort the run. Found while testing
    #     this script against real module strings.
    before = len(df)
    df = df[df.moduleName.str.contains("nrChannelModel", regex=False)
            & df.moduleName.str.contains(".ue[", regex=False)].copy()
    if df.empty:
        raise RuntimeError(
            f"No UE nrChannelModel rows found in {vec_file}. Total rows before "
            f"filtering: {before}. Check the module path -- do NOT proceed."
        )

    ue = df.moduleName.str.extract(r"\.ue\[(\d+)\]\.", expand=False)
    if ue.isnull().any():
        raise RuntimeError("Some moduleName values did not match the ue[N] pattern.")
    df["ue_index"] = ue.astype(int)

    df["snapshot_id"] = (df.time_s / WINDOW_S).astype(int) - trim
    df = df[df.snapshot_id >= 0]

    agg = (df.groupby(["snapshot_id", "ue_index"], as_index=False)
             .agg(sinr_db=("sinr_db", "mean"),
                  n_samples=("sinr_db", "size")))
    return agg


def main():
    trim = TRIMS[SEED]
    powers = POWERS.get(SEED, ("p37", "p43", "p46"))
    print(f"seed {SEED}, trim {trim} snapshots, runs {powers}\n")

    for power in powers:
        vec = f"results/{config_name(SEED, power)}/{REPS[(SEED, power)]}.vec"
        target_dir = out_dir(SEED, power)
        print(f"=== {power} ===")
        print(f"  reading {vec}")

        agg = extract_one(vec, trim)

        # Align to the snapshot range the other CSVs actually have.
        ref = pd.read_csv(f"{target_dir}/gnb_inputs.csv")
        valid = set(ref.snapshot_id.unique())
        agg = agg[agg.snapshot_id.isin(valid)]

        n_snap = agg.snapshot_id.nunique()
        expected = len(valid) * NUM_UES
        print(f"  snapshots {n_snap} (CSVs have {len(valid)}), "
              f"rows {len(agg)} (expect {expected})")
        print(f"  samples per (ue, snapshot): "
              f"min {agg.n_samples.min()} median {int(agg.n_samples.median())} "
              f"max {agg.n_samples.max()}")
        print(f"  sinr_db: mean {agg.sinr_db.mean():.2f} "
              f"min {agg.sinr_db.min():.2f} max {agg.sinr_db.max():.2f}")

        if len(agg) != expected:
            print(f"  WARNING: row count mismatch -- {expected - len(agg)} "
                  f"(ue, snapshot) pairs have no SINR samples. They are "
                  f"DROPPED, not zero-filled. Check before training.")

        path = f"{target_dir}/sinr_features.csv"
        agg[["snapshot_id", "ue_index", "sinr_db"]].to_csv(path, index=False)
        print(f"  wrote {path}\n")


if __name__ == "__main__":
    main()
