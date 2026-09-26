"""
extract_dl_sinr_cqi_scenario07.py

Extracts two more per-UE channel features from the existing `.vec`
files. No re-simulation. Run per seed:

    SEED=27 python3 extract_dl_sinr_cqi_scenario07.py

What is available, and what is not
----------------------------------
Confirmed by querying the vec index:
  nrChannelModel[0] measuredSinrDl  count=20000  -> AVAILABLE
  nrPhy             averageCqiUl    count=4712-11827 -> AVAILABLE, SPARSE
  nrPhy             averageCqiDl    count=0      -> empty, no DL traffic exists
  rsrp / rssi                       absent       -> NOT RECORDED AT ALL
So RSRP cannot be filled from this data; it needs either a different
statistic enabled or computing from path loss. DL delay/jitter/loss
likewise need actual DL traffic in the `.ini` and a re-simulation.

Same two-filter trap as UL SINR: the LTE `channelModel` publishes
`measuredSinrDl` with count=0, and gNB modules also carry
`nrChannelModel`. Both `nrChannelModel` AND `.ue[` are required.

Expectation, stated up front so the result is interpretable
-----------------------------------------------------------
DL SINR per-UE means span roughly 31-37 dB against UL's 11-27 dB, so
it carries noticeably less spread across UEs. It is worth testing but
is a weaker candidate than UL SINR was. If it adds nothing, the honest
conclusion is that channel information is saturated after UL SINR, not
that DL is irrelevant.

CQI sparsity, handled explicitly
--------------------------------
averageCqiUl is reported per transmission, so a UE that sends little
during an L-phase produces few or no samples in a given snapshot.
Sample counts vary 4712-11827 across UEs (vs SINR's uniform 20000),
i.e. roughly 4-10 per snapshot with real gaps. Gaps are FORWARD-FILLED
within each UE, which is the physically correct choice: a reported CQI
persists as the scheduler's working value until the next report. Any
leading gap before a UE's first report is back-filled. The script
prints how many values were filled so the imputation is visible rather
than hidden.

Outputs, both aligned to the trimmed snapshot range:
  dl_sinr_features.csv  (snapshot_id, ue_index, sinr_dl_db)
  cqi_features.csv      (snapshot_id, ue_index, cqi_ul, cqi_ul_filled)
"""

import os
import sqlite3
import subprocess
import tempfile

import numpy as np
import pandas as pd

SEED = int(os.environ.get("SEED", "27"))
NUM_UES = 30
WINDOW_S = 0.1

TRIMS = {21: 370, 22: 360, 23: 360, 24: 360, 25: 510, 26: 420, 27: 300}
POWERS = {26: ("p37", "p46")}

SPECS = [
    # (stat name, module substring, output column, output filename)
    ("measuredSinrDl:vector", "nrChannelModel", "sinr_dl_db", "dl_sinr_features.csv"),
    ("averageCqiUl:vector",   "nrPhy",          "cqi_ul",     "cqi_features.csv"),
]


def config_name(seed, power):
    return f"Scenario07_{power}" if seed == 21 else f"Scenario07s{seed}_{power}"


def out_dir(seed, power):
    return f"results/extracted_{power}" if seed == 21 else f"results/extracted_s{seed}_{power}"


def read_stat(vec_file, stat, module_substr):
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        path = tmp.name
    try:
        r = subprocess.run(
            ["opp_scavetool", "export", "-F", "SqliteVectorFile", "-o", path,
             "-f", f"name =~ {stat}", vec_file],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"opp_scavetool failed on {vec_file}\n{r.stderr}")
        conn = sqlite3.connect(path)
        df = pd.read_sql(
            "SELECT vector.moduleName AS moduleName, "
            "vectorData.simtimeRaw / 1e12 AS time_s, vectorData.value AS val "
            "FROM vectorData JOIN vector ON vectorData.vectorId = vector.vectorId",
            conn)
        conn.close()
    finally:
        os.remove(path)

    before = len(df)
    df = df[df.moduleName.str.contains(module_substr, regex=False)
            & df.moduleName.str.contains(".ue[", regex=False)].copy()
    if df.empty:
        raise RuntimeError(
            f"No UE {module_substr} rows for {stat} in {vec_file} "
            f"({before} rows before filtering). Do NOT proceed.")

    ue = df.moduleName.str.extract(r"\.ue\[(\d+)\]\.", expand=False)
    if ue.isnull().any():
        raise RuntimeError(f"moduleName did not match ue[N] for {stat}.")
    df["ue_index"] = ue.astype(int)
    return df


def main():
    trim = TRIMS[SEED]
    powers = POWERS.get(SEED, ("p37", "p43", "p46"))
    print(f"seed {SEED}, trim {trim} snapshots, runs {powers}\n")

    for power in powers:
        vec = f"results/{config_name(SEED, power)}/0.vec"
        target = out_dir(SEED, power)
        valid = set(pd.read_csv(f"{target}/gnb_inputs.csv").snapshot_id.unique())
        full_index = pd.MultiIndex.from_product(
            [sorted(valid), range(NUM_UES)], names=["snapshot_id", "ue_index"])
        print(f"=== {power} ===")

        for stat, module_substr, col, fname in SPECS:
            df = read_stat(vec, stat, module_substr)
            df["snapshot_id"] = (df.time_s / WINDOW_S).astype(int) - trim
            df = df[df.snapshot_id.isin(valid)]

            raw = (df.groupby(["snapshot_id", "ue_index"])
                     .val.mean()
                     .reindex(full_index))
            missing_mask = raw.isnull()
            n_missing = int(missing_mask.sum())

            # Forward-fill within each UE: a reported value persists as
            # the working value until the next report. Back-fill covers
            # any leading gap before a UE's first report.
            filled = (raw.groupby(level="ue_index").ffill()
                         .groupby(level="ue_index").bfill())

            out = filled.reset_index(name=col)
            if out[col].isnull().any():
                raise RuntimeError(
                    f"{fname}: {int(out[col].isnull().sum())} values still "
                    f"null after fill -- a UE has no samples at all. NOT written.")

            if col == "cqi_ul":
                out["cqi_ul_filled"] = missing_mask.astype(int).values

            print(f"  {col:<11} rows {len(out)}  filled {n_missing} "
                  f"({n_missing/len(out)*100:.1f}%)  "
                  f"mean {out[col].mean():.2f} "
                  f"min {out[col].min():.2f} max {out[col].max():.2f}")
            out.to_csv(f"{target}/{fname}", index=False)
        print()


if __name__ == "__main__":
    main()
