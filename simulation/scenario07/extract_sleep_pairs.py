"""
extract_sleep_pairs.py -- extracts baseline + forced-sleep runs into
paired CSV sets for CMOA training.

    SEED=22 python3 extract_sleep_pairs.py

Reads PAIRS below (edit to match what verify_sleep.py passed) and writes
one folder per run: results/extracted_sleep_s{SEED}_p{P}_{tag}/ where
tag is "base" or "sleep{k}".

THE BUG THIS SCRIPT EXISTS TO AVOID
------------------------------------
extract_scenario07.py infers sleep as:
    sleeping = 1 if (avg_rb_ul == 0 and window_start < 0.5) else 0
That heuristic was written for scenario07's startup transient. In a
forced-sleep run the cell is off for the WHOLE 120s, so `window_start <
0.5` is false after the first half-second and the sleeping gNB would be
labelled ACTIVE at ~260W instead of the 150W idle floor. Energy deltas
between the pair would come out as zero and the entire CMOA dataset
would be meaningless while looking perfectly well-formed.

Here the flag comes from the CONFIG, not from the data:
    is_sleeping = 1 for the forced gNB, every snapshot
    p_gnb       = N_TRX * 75.0 = 150W for that gNB
Everything else is unchanged from extract_scenario07.py so the two are
directly comparable.

WHAT CMOA GETS FROM THIS
------------------------
For an identical network state (same seed, same UE trajectories, same
traffic schedule) you have measured energy and measured
delay/jitter/loss both with cell k serving and with cell k asleep. The
load did NOT cause the sleep -- it was imposed -- which is precisely
the correlation that has to be broken before "what if I sleep this
cell?" is answerable. Nothing is imputed: neighbours genuinely absorb
the traffic and the handovers genuinely occur (verified: the slept
cell's serving-share drops from ~25% to 0%).

Two extra columns in gnb_inputs.csv:
    is_sleeping      0/1 per gNB per snapshot -- CMOA's control knob
    forced_sleep_gnb which gNB this RUN slept (-1 for baseline)
"""

import math
import os
import random
import sqlite3
import subprocess

import numpy as np
import pandas as pd

SEED = int(os.environ.get("SEED", "22"))

# (power, sleep_gnb or None for baseline, repetition). Only runs that
# verify_sleep.py confirmed complete AND actually lost their cell.
PAIRS = [
    (37, None, 0), (37, 0, 0), (37, 1, 2), (37, 2, 0), (37, 3, 0),
    (43, None, 0), (43, 0, 0), (43, 1, 0), (43, 2, 0), (43, 3, 0),
    (46, None, 0), (46, 0, 0), (46, 1, 0), (46, 2, 0), (46, 3, 0),
]

NUM_UES = 30
NUM_GNB = 4
MAX_RBS = 50
N_TRX = 2
P0_W = 130.0
P_SLEEP_W = 75.0          # per TRX; 150W total, matches extract_scenario07
DELTA_P = 4.7
N_SNAPSHOTS = 1200
WINDOW = 0.1
MAP_SIZE = 2000.0         # scenario07 area is square, unlike scenario06b
SQLITE_TMP = f"/tmp/sleep_s{SEED}.sqlite"

GNB_POS = {0: (500, 500, 30), 1: (1500, 500, 30),
           2: (500, 1500, 30), 3: (1500, 1500, 30)}

# ---- RNG replay, identical draw order to gen_ini.py ----
random.seed(SEED)


def closest_gnb(x, y):
    return min(GNB_POS, key=lambda g: math.hypot(x - GNB_POS[g][0], y - GNB_POS[g][1]))


ue_mobility, ue_home_gnb = {}, {}
for u in range(NUM_UES):
    x = random.uniform(200, 1800)
    y = random.uniform(200, 1800)
    speed = random.uniform(3, 8)
    heading = random.uniform(0, 360)
    ue_mobility[u] = {"x0": x, "y0": y, "speed": speed, "heading_deg": heading}
    ue_home_gnb[u] = closest_gnb(x, y)

QOS_WEIGHTS = {"eMBB": 0.4, "URLLC": 0.3, "mMTC": 0.3}


def assign_qos():
    r = random.random()
    cum = 0.0
    for cls, w in QOS_WEIGHTS.items():
        cum += w
        if r < cum:
            return cls
    return "eMBB"


qos_assignments = {u: assign_qos() for u in range(NUM_UES)}
print(f"seed {SEED}: replay check -- UE0 home_gnb={ue_home_gnb[0]} "
      f"qos={qos_assignments[0]}, UE29 home_gnb={ue_home_gnb[29]} "
      f"qos={qos_assignments[29]}  (diff vs ue_assignment_s{SEED}.txt)")


def reflect_1d(pos, size):
    period = 2 * size
    m = pos % period
    return period - m if m > size else m


def reconstruct_position(u, t):
    m = ue_mobility[u]
    hr = math.radians(m["heading_deg"])
    return (reflect_1d(m["x0"] + m["speed"] * math.cos(hr) * t, MAP_SIZE),
            reflect_1d(m["y0"] + m["speed"] * math.sin(hr) * t, MAP_SIZE))


def export_sqlite(vec, name_pattern, module_pattern=None):
    filt = f"name =~ {name_pattern}*"
    if module_pattern:
        filt += f" AND module =~ {module_pattern}"
    subprocess.run(["opp_scavetool", "export", "-T", "v", "-f", filt,
                    "-F", "SqliteVectorFile", "-o", SQLITE_TMP, vec],
                   capture_output=True)


def read_sqlite():
    conn = sqlite3.connect(SQLITE_TMP)
    df = pd.read_sql_query(
        """SELECT v.moduleName, d.simtimeRaw/1e12 AS time_s, d.value
           FROM vectorData d JOIN vector v ON d.vectorId = v.vectorId
           ORDER BY v.moduleName, d.simtimeRaw""", conn)
    conn.close()
    return df


starts = np.arange(0, N_SNAPSHOTS * WINDOW, WINDOW)
snapshots = pd.DataFrame({"snapshot_id": np.arange(N_SNAPSHOTS),
                          "window_start": starts, "window_end": starts + WINDOW})

for tx_dbm, sleep_gnb, rep in PAIRS:
    tag = "base" if sleep_gnb is None else f"sleep{sleep_gnb}"
    cfg = (f"Scenario07s{SEED}_p{tx_dbm}" if sleep_gnb is None
           else f"Scenario07s{SEED}_p{tx_dbm}_sleep{sleep_gnb}")
    vec = f"results/{cfg}/{rep}.vec"
    if not os.path.exists(vec):
        print(f"\n=== {cfg}: {vec} MISSING, skipped ===")
        continue

    print(f"\n=== {cfg} (tx={tx_dbm}dBm, sleeping={sleep_gnb}) ===")
    tx_w = 10 ** (tx_dbm / 10) / 1000
    out_dir = f"results/extracted_sleep_s{SEED}_p{tx_dbm}_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    export_sqlite(vec, "servingCell:vector", module_pattern="**.nrPhy")
    serving_all = read_sqlite()

    edge_rows = []
    for u in range(NUM_UES):
        us = serving_all[serving_all.moduleName.str.contains(
            f"ue[{u}].", regex=False)].sort_values("time_s")
        for _, sn in snapshots.iterrows():
            sub = us[(us.time_s >= sn.window_start) & (us.time_s < sn.window_end)]
            edge_rows.append({"snapshot_id": int(sn.snapshot_id), "ue_index": u,
                              "serving_cell_raw": sub.value.iloc[-1] if not sub.empty else np.nan})

    edge_df = pd.DataFrame(edge_rows).sort_values(["ue_index", "snapshot_id"])
    edge_df["serving_cell_raw"] = edge_df["serving_cell_raw"].replace(0, np.nan)
    edge_df["serving_cell_raw"] = edge_df.groupby("ue_index")["serving_cell_raw"].transform(
        lambda s: s.ffill().bfill())
    edge_df["serving_gnb_index"] = (edge_df["serving_cell_raw"] - 1).astype(int)
    edge_df["serving_gnb_id"] = "gnb_" + edge_df["serving_gnb_index"].astype(str)
    edge_df = edge_df.drop(columns=["serving_cell_raw"])
    edge_df.to_csv(f"{out_dir}/serving_edges.csv", index=False)

    if sleep_gnb is not None:
        stuck = (edge_df.serving_gnb_index == sleep_gnb).mean() * 100
        print(f"  UEs still on slept gnb{sleep_gnb}: {stuck:.2f}% "
              f"({'ok' if stuck < 2 else 'SUSPICIOUS -- check the config'})")

    export_sqlite(vec, "packetSent:vector", module_pattern="**.app[*]")
    tput_all = read_sqlite()

    ue_rows = []
    for u in range(NUM_UES):
        ut = tput_all[tput_all.moduleName.str.contains(f"ue[{u}].", regex=False)]
        for _, sn in snapshots.iterrows():
            t_mid = (sn.window_start + sn.window_end) / 2
            x, y = reconstruct_position(u, t_mid)
            sub = ut[(ut.time_s >= sn.window_start) & (ut.time_s < sn.window_end)]
            bps = (sub.value.sum() * 8 / WINDOW) if not sub.empty else 0.0
            ue_rows.append({
                "snapshot_id": int(sn.snapshot_id), "ue_index": u,
                "pos_x": x, "pos_y": y, "pos_z": 1.5,
                "speed_mps": ue_mobility[u]["speed"], "qos_class": qos_assignments[u],
                "home_gnb_index": ue_home_gnb[u],
                "offered_ul_bytes": int(bps * WINDOW / 8), "offered_ul_bps": round(bps, 4)})

    ue_df = pd.DataFrame(ue_rows)
    ue_df.to_csv(f"{out_dir}/ue_inputs.csv", index=False)

    off_lookup = {(int(r.snapshot_id), int(r.ue_index)): r.offered_ul_bps
                  for r in ue_df.itertuples()}
    edges_by_snap = {sid: g for sid, g in edge_df.groupby("snapshot_id")}

    export_sqlite(vec, "avgServedBlocksUl:vector")
    rb_all = read_sqlite()
    gnb_rb = {g: rb_all[rb_all.moduleName.str.contains(f"gnb{g}.", regex=False)]
                    [["time_s", "value"]].rename(columns={"value": "rb_ul"}).sort_values("time_s")
              for g in range(NUM_GNB)}

    gnb_rows, tgt_rows, diag_rows = [], [], []
    for _, sn in snapshots.iterrows():
        sid = int(sn.snapshot_id)
        se = edges_by_snap.get(sid)
        for g in range(NUM_GNB):
            rb = gnb_rb[g]
            sub = rb[(rb.time_s >= sn.window_start) & (rb.time_s < sn.window_end)]
            avg_rb = sub.rb_ul.mean() if not sub.empty else 0.0
            load = avg_rb / MAX_RBS
            ids = ([] if se is None
                   else se[se.serving_gnb_index == g].ue_index.tolist())
            bps = sum(off_lookup.get((sid, u), 0.0) for u in ids)

            # Flag from the CONFIG, not inferred from the data -- see header.
            is_sleeping = 1 if g == sleep_gnb else 0
            p = (N_TRX * P_SLEEP_W if is_sleeping
                 else N_TRX * P0_W + DELTA_P * tx_w * load * N_TRX)

            gnb_rows.append({
                "snapshot_id": sid, "gnb_index": g, "gnb_id": f"gnb_{g}",
                "pos_x": GNB_POS[g][0], "pos_y": GNB_POS[g][1], "pos_z": GNB_POS[g][2],
                "tx_power_dbm": tx_dbm, "tx_power_w": round(tx_w, 6),
                "num_bands": MAX_RBS, "num_trx": N_TRX,
                "scheduler_dl": "PF", "scheduler_ul": "PF",
                "num_connected_ues": len(ids),
                "offered_ul_bytes": int(bps * WINDOW / 8),
                "offered_ul_bps": round(bps, 4),
                "is_sleeping": is_sleeping,
                "forced_sleep_gnb": -1 if sleep_gnb is None else sleep_gnb})
            tgt_rows.append({"snapshot_id": sid, "gnb_index": g, "gnb_id": f"gnb_{g}",
                             "estimated_gnb_power_w": round(p, 6)})
            diag_rows.append({"snapshot_id": sid, "gnb_index": g,
                              "avg_rb_ul": round(avg_rb, 6), "load_used": round(load, 6),
                              "sleeping": is_sleeping})

    pd.DataFrame(gnb_rows).to_csv(f"{out_dir}/gnb_inputs.csv", index=False)
    pd.DataFrame(tgt_rows).to_csv(f"{out_dir}/energy_targets.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(f"{out_dir}/energy_diagnostics.csv", index=False)

    t = pd.DataFrame(tgt_rows)
    print(f"  energy: mean={t.estimated_gnb_power_w.mean():.2f} "
          f"min={t.estimated_gnb_power_w.min():.2f} "
          f"max={t.estimated_gnb_power_w.max():.2f}")
    print(f"  -> {out_dir}/")

if os.path.exists(SQLITE_TMP):
    os.remove(SQLITE_TMP)
print("\nDone. Next: KPI + SINR extraction for the same runs, then "
      "compare_pairs.py to see the energy/QoS trade-off per sleep action.")
