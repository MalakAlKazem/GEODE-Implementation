"""
verify_sleep.py -- confirms every sleep run actually put its cell to sleep.

    SEED=22 python3 verify_sleep.py

Why this exists as a separate, mandatory step
---------------------------------------------
OMNeT++ accepts an ini parameter whose path matches no module WITHOUT
warning. If `*.gnbK.cellularNic.phy.eNodeBTxPower` were slightly wrong,
every "sleep" run would complete normally with the cell transmitting at
full power, and the resulting dataset would look entirely believable
while containing no sleep events at all -- CMOA would then be trained
on a lie. The path was verified once by hand on seed 22 p43; this
checks it held for every run rather than assuming.

Test: read servingCell from each sleep run and confirm cell (k+1) has
essentially no UEs after the settling window, while the baseline run
does have UEs on it. Both halves matter -- if the baseline also shows
no UEs on cell k, the cell was empty anyway and the pair carries no
counterfactual information.
"""

import os
import sqlite3
import subprocess
import tempfile

import pandas as pd

SEED = int(os.environ.get("SEED", "22"))
POWERS = (37, 43, 46)
NUM_GNB = 4
SETTLE_T = 40.0          # ignore the handover-settling window
TOL = 0.05               # residue scales with the cell load and with 1dBm not being zero power

REPS = {}
log = "sleep_successful_runs.txt"
if os.path.exists(log):
    for line in open(log):
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("r="):
            REPS[parts[0]] = int(parts[1][2:])


def serving_counts(cfg):
    rep = REPS.get(cfg, 0)
    vec = f"results/{cfg}/{rep}.vec"
    if not os.path.exists(vec):
        return None
    # A crashed run still writes everything up to the crash, so post-settling
    # data can exist in a partial file -- that is how a truncated sleep2 run
    # passed this check earlier. The .sca is only written at finish(), so its
    # size distinguishes complete (~10MB) from crashed (~0.5MB) reliably.
    sca = vec.replace(".vec", ".sca")
    if not os.path.exists(sca) or os.path.getsize(sca) < 5_000_000:
        print(f"  {cfg}: INCOMPLETE run (.sca too small -- crashed before finish)")
        return None
    tmp = tempfile.mktemp(suffix=".sqlite")
    try:
        subprocess.run(["opp_scavetool", "export", "-F", "SqliteVectorFile",
                        "-o", tmp, "-f", "name =~ servingCell:vector", vec],
                       capture_output=True)
        conn = sqlite3.connect(tmp)
        d = pd.read_sql(
            "SELECT v.moduleName, d.simtimeRaw/1e12 t, d.value "
            "FROM vectorData d JOIN vector v ON d.vectorId=v.vectorId", conn)
        conn.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    d = d[d.moduleName.str.contains("nrPhy", regex=False)
          & d.moduleName.str.contains(".ue[", regex=False)]
    d = d[d.t > SETTLE_T]
    if d.empty:
        return None
    return d.value.value_counts(normalize=True)


print(f"seed {SEED}: verifying sleep took effect (t > {SETTLE_T}s)\n")
all_ok = True
for p in POWERS:
    base = serving_counts(f"Scenario07s{SEED}_p{p}")
    if base is None:
        print(f"p{p}: baseline missing, skipped")
        continue
    for k in range(NUM_GNB):
        cfg = f"Scenario07s{SEED}_p{p}_sleep{k}"
        sl = serving_counts(cfg)
        if sl is None:
            print(f"  {cfg}: MISSING")
            all_ok = False
            continue
        cell = k + 1
        base_share = base.get(cell, 0.0)
        sleep_share = sl.get(cell, 0.0)
        if sleep_share > TOL:
            verdict = "FAIL -- cell still serving, parameter path did not apply"
            all_ok = False
        elif base_share < 0.05:
            verdict = "WEAK -- cell had almost no UEs even when awake"
        else:
            verdict = "ok"
        print(f"  {cfg}: gnb{k} share awake {base_share*100:5.1f}% "
              f"-> asleep {sleep_share*100:4.1f}%   {verdict}")
    print()

print("ALL RUNS VERIFIED" if all_ok else
      "PROBLEMS FOUND -- do not extract until resolved")
