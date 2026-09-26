"""
extract_sinr_scenario07.py

Extracts per-UE uplink SINR from the existing .vec files as a new GNN
INPUT feature. No re-simulation needed -- vector-recording was already on.

Usage:  SEED=27 python3 extract_sinr_scenario07.py

TRAP: two modules publish a stat named measuredSinrUl:vector --
  ue[N].cellularNic.channelModel[0]    LTE stack, count=0, EMPTY
  ue[N].cellularNic.nrChannelModel[0]  5G NR stack, count=20000
and gNB modules also carry nrChannelModel. So BOTH filters are needed:
'nrChannelModel' AND '.ue['. Same class of trap as the SCTP
endToEndDelay collision found earlier in this project.

UL not DL: KPI targets are uplink, and UL SINR has real variance
(per-UE means 11-27 dB) while DL sits uniformly high (~35 dB).

Granularity: 20000 samples / 120s = one per ~6ms, so ~16 per 100ms
snapshot. This is a REAL per-snapshot value, unlike delay/jitter/loss
which are one value per 3s phase broadcast across 30 snapshots.

Not leakage: SINR depends on geometry, tx power and interference, not
on load_used. Causally upstream of the KPIs. Limitation to record: in a
CMOA counterfactual, sleeping a gNB would change SINR in ways the model
cannot infer from a measured value.

Output: sinr_features.csv (snapshot_id, ue_index, sinr_db) -- the schema
data_loader.py already looks for.
"""

import os
import sqlite3
import subprocess
import tempfile

import pandas as pd

SEED = 24
NUM_UES = 30
WINDOW_S = 0.1
STAT = "measuredSinrUl:vector"

# Must match what trim_sNN.py actually applied, or SINR lands
# misaligned against every other CSV by the difference.
TRIMS = {24: 0}

# seed 26 p43 crashed (Simu5G NrMacGnb handover race at t=11.05s)
POWERS = {}


def config_name(seed, power):
    return f"Scenario07s{seed}_{power}"


def out_dir(seed, power):
    return f"results/extracted_sleep_s{seed}_{power}"


def run_scavetool(args):
    r = subprocess.run(["opp_scavetool"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"opp_scavetool failed: {' '.join(args)}\n{r.stderr}")


def extract_one(vec_file, trim):
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

    before = len(df)
    df = df[df.moduleName.str.contains("nrChannelModel", regex=False)
            & df.moduleName.str.contains(".ue[", regex=False)].copy()
    if df.empty:
        raise RuntimeError(
            f"No UE nrChannelModel rows in {vec_file}. Rows before filter: "
            f"{before}. Check the module path -- do NOT proceed.")

    ue = df.moduleName.str.extract(r"\.ue\[(\d+)\]\.", expand=False)
    if ue.isnull().any():
        raise RuntimeError("Some moduleName values did not match ue[N].")
    df["ue_index"] = ue.astype(int)

    df["snapshot_id"] = (df.time_s / WINDOW_S).astype(int) - trim
    df = df[df.snapshot_id >= 0]

    return (df.groupby(["snapshot_id", "ue_index"], as_index=False)
              .agg(sinr_db=("sinr_db", "mean"), n_samples=("sinr_db", "size")))


def main():
    trim = TRIMS[SEED]
    powers = ["p37","p37_sleep0","p37_sleep1","p37_sleep2","p37_sleep3","p43","p43_sleep0","p43_sleep1","p43_sleep2","p43_sleep3","p46","p46_sleep0","p46_sleep1","p46_sleep2","p46_sleep3"]
    print(f"seed {SEED}, trim {trim} snapshots, runs {powers}\n")

    for power in powers:
        vec = f"results/{config_name(SEED, power)}/{ {23:{"p43_sleep3":1,"p46_sleep3":1},24:{"p37_sleep1":2}}.get(SEED,{}).get(power,0) }.vec"
        target_dir = out_dir(SEED, power)
        print(f"=== {power} ===")
        print(f"  reading {vec}")

        agg = extract_one(vec, trim)

        ref = pd.read_csv(f"{target_dir}/gnb_inputs.csv")
        valid = set(ref.snapshot_id.unique())
        agg = agg[agg.snapshot_id.isin(valid)]

        expected = len(valid) * NUM_UES
        print(f"  snapshots {agg.snapshot_id.nunique()} (CSVs have {len(valid)}), "
              f"rows {len(agg)} (expect {expected})")
        print(f"  samples per (ue,snapshot): min {agg.n_samples.min()} "
              f"median {int(agg.n_samples.median())} max {agg.n_samples.max()}")
        print(f"  sinr_db: mean {agg.sinr_db.mean():.2f} "
              f"min {agg.sinr_db.min():.2f} max {agg.sinr_db.max():.2f}")

        if len(agg) != expected:
            print(f"  WARNING: {expected - len(agg)} (ue,snapshot) pairs have "
                  f"no SINR samples. DROPPED, not zero-filled. Check before training.")

        path = f"{target_dir}/sinr_features.csv"
        agg[["snapshot_id", "ue_index", "sinr_db"]].to_csv(path, index=False)
        print(f"  wrote {path}\n")


if __name__ == "__main__":
    main()
