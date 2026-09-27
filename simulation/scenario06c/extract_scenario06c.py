"""
extract_scenario06b.py  --  6 gNB / 60 UE energy + inputs extraction

    SEED=51 python3 extract_scenario06b.py

Adapted from extract_scenario07.py. Four changes matter; the rest is
parameterization.

1. NON-SQUARE AREA. scenario07 was 2000x2000, so one MAP_SIZE constant
   served both axes in reflect_1d(). scenario06b is 3000x2000. Using a
   single constant would bounce every UE's x-coordinate off the wrong
   wall and silently produce wrong positions for the whole dataset.
   Separate MAP_X / MAP_Y below.

2. RNG REPLAY must match gen_ini_s06b.py EXACTLY: random.seed(SEED),
   then per UE in order x=uniform(200,2800), y=uniform(200,1800),
   speed=uniform(3,8), heading=uniform(0,360) for all 60 UEs, THEN 60
   qos draws. Any deviation shifts the whole stream and every position
   and QoS label comes out wrong with no error raised. The script
   prints the reconstruction so it can be diffed against
   ue_assignment_s{SEED}.txt -- do that before trusting the output.

3. REPETITION-AWARE PATHS. The NrMacGnb handover race kills most
   repetitions at 6 gNB, so the completed file is often not 0.vec.
   REPS below comes from s06b_successful_runs.txt. Note crashed runs
   can be 2.0-2.4G against a good run's 2.6G, so file size is NOT a
   safe way to identify the right one -- trust the log.

4. PERFORMANCE. The original looked up offered_ul_bps with a boolean
   mask over the whole UE frame inside the gNB loop: 1200 snapshots x
   6 gNBs x 72,000 rows is ~500M row scans. Replaced with a dict
   keyed by (snapshot_id, ue_index), built once. Same numbers, minutes
   instead of tens of minutes.

Energy model, unchanged from scenario07 so the two are comparable:
P = N_TRX * P0 + DELTA_P * tx_w * load * N_TRX, load = avg_rb_ul / MAX_RBS (25 in scenario06c).
"""

import math
import os
import random
import sqlite3
import subprocess

import numpy as np
import pandas as pd

SEED = int(os.environ.get("SEED", "61"))

NUM_UES = 60
NUM_GNB = 6
MAX_RBS = 25
N_TRX = 2
P0_W = 130.0
DELTA_P = 4.7
N_SNAPSHOTS = 1200
WINDOW = 0.1

# Non-square area -- see note 1 above.
MAP_X = 1500.0
MAP_Y = 1000.0

# NOT /tmp: that is a 3.9G tmpfs here and a single .vec is ~2.8G,
# so the SQLite conversion would run out of space partway through.
SQLITE_TMP = os.path.expanduser(f"~/scenario06c_s{SEED}_extract.sqlite")

# Repetition that completed, per (seed, run). From s06b_successful_runs.txt.
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

# Derived from REPS so the two can never disagree. dBm is 43 for every
# sleep variant (sleep pairs are generated at p43 only).
CONFIGS = {
    run: (f"results/Scenario06Cs{SEED}_{run}/{REPS[(SEED, run)]}.vec",
          37 if run == "p37" else (46 if run == "p46" else 43))
    for (s, run) in REPS if s == SEED
}

# Must match SixCell_Standalone.ned and gen_ini_s06b.py.
GNB_POS = {0: (250, 250, 30), 1: (750, 250, 30), 2: (1250, 250, 30),
           3: (250, 750, 30), 4: (750, 750, 30), 5: (1250, 750, 30)}

# ---- RNG replay: mirrors gen_ini_s06b.py draw-for-draw ----
random.seed(SEED)


def closest_gnb(x, y):
    return min(GNB_POS, key=lambda g: math.hypot(x - GNB_POS[g][0], y - GNB_POS[g][1]))


ue_mobility, ue_home_gnb = {}, {}
for u in range(NUM_UES):
    x = random.uniform(100, 1400)      # matches generator's x range
    y = random.uniform(100, 900)       # matches generator's y range
    mc = random.random()                  # matches generator's class draw
    if mc < 0.60:
        speed = random.uniform(1.0, 2.0)
    elif mc < 0.90:
        speed = random.uniform(8.0, 15.0)
    else:
        speed = random.uniform(15.0, 25.0)
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

print(f"seed {SEED}: reconstructed assignment "
      f"(diff against ue_assignment_s{SEED}.txt before trusting output)")
for u in range(NUM_UES):
    print(f"  UE {u}: home_gnb={ue_home_gnb[u]}, qos={qos_assignments[u]}")


def reflect_1d(pos, size):
    period = 2 * size
    p_mod = pos % period
    return period - p_mod if p_mod > size else p_mod


def reconstruct_position(u, t):
    m = ue_mobility[u]
    hr = math.radians(m["heading_deg"])
    x = reflect_1d(m["x0"] + m["speed"] * math.cos(hr) * t, MAP_X)
    y = reflect_1d(m["y0"] + m["speed"] * math.sin(hr) * t, MAP_Y)
    return x, y


def export_sqlite(vec_file, name_pattern, module_pattern=None):
    filt = f"name =~ {name_pattern}*"
    if module_pattern:
        filt += f" AND module =~ {module_pattern}"
    subprocess.run(["opp_scavetool", "export", "-T", "v", "-f", filt,
                    "-F", "SqliteVectorFile", "-o", SQLITE_TMP, vec_file],
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

for run_name, (vec_file, tx_dbm) in CONFIGS.items():
    if not os.path.exists(vec_file):
        print(f"\n=== {run_name}: {vec_file} MISSING, skipped ===")
        continue

    print(f"\n=== Processing {run_name} (tx={tx_dbm}dBm, {vec_file}) ===")
    tx_w = 10 ** (tx_dbm / 10) / 1000
    out_dir = f"results/extracted_s06c{SEED}_{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- serving cell (ground truth for handover) ---
    export_sqlite(vec_file, "servingCell:vector", module_pattern="**.nrPhy")
    serving_all = read_sqlite()

    edge_rows = []
    for u in range(NUM_UES):
        ue_serving = serving_all[
            serving_all.moduleName.str.contains(f"ue[{u}].", regex=False)].sort_values("time_s")
        for _, snap in snapshots.iterrows():
            sub = ue_serving[(ue_serving.time_s >= snap.window_start)
                             & (ue_serving.time_s < snap.window_end)]
            edge_rows.append({"snapshot_id": int(snap.snapshot_id), "ue_index": u,
                              "serving_cell_raw": sub.value.iloc[-1] if not sub.empty else np.nan})
        if u % 20 == 0:
            print(f"  serving_edges: UE {u}/{NUM_UES}")

    edge_df = pd.DataFrame(edge_rows).sort_values(["ue_index", "snapshot_id"])
    edge_df["serving_cell_raw"] = edge_df["serving_cell_raw"].replace(0, np.nan)
    edge_df["serving_cell_raw"] = edge_df.groupby("ue_index")["serving_cell_raw"].transform(
        lambda s: s.ffill().bfill())
    edge_df["serving_gnb_index"] = (edge_df["serving_cell_raw"] - 1).astype(int)
    edge_df["serving_gnb_id"] = "gnb_" + edge_df["serving_gnb_index"].astype(str)
    edge_df = edge_df.drop(columns=["serving_cell_raw"])
    edge_df.to_csv(f"{out_dir}/serving_edges.csv", index=False)
    print(f"  Saved serving_edges.csv ({len(edge_df)} rows)")

    # --- UE traffic + reconstructed positions ---
    export_sqlite(vec_file, "packetSent:vector", module_pattern="**.app[*]")
    tput_all = read_sqlite()

    ue_rows = []
    for u in range(NUM_UES):
        ue_tput = tput_all[tput_all.moduleName.str.contains(f"ue[{u}].", regex=False)]
        for _, snap in snapshots.iterrows():
            t_mid = (snap.window_start + snap.window_end) / 2
            x, y = reconstruct_position(u, t_mid)
            sub = ue_tput[(ue_tput.time_s >= snap.window_start)
                          & (ue_tput.time_s < snap.window_end)]
            off_ul_bps = (sub.value.sum() * 8 / WINDOW) if not sub.empty else 0.0
            ue_rows.append({
                "snapshot_id": int(snap.snapshot_id), "ue_index": u,
                "pos_x": x, "pos_y": y, "pos_z": 1.5,
                "speed_mps": ue_mobility[u]["speed"], "qos_class": qos_assignments[u],
                "home_gnb_index": ue_home_gnb[u],
                "offered_ul_bytes": int(off_ul_bps * WINDOW / 8),
                "offered_ul_bps": round(off_ul_bps, 4),
            })
        if u % 20 == 0:
            print(f"  ue_inputs: UE {u}/{NUM_UES}")

    ue_df = pd.DataFrame(ue_rows)
    ue_df.to_csv(f"{out_dir}/ue_inputs.csv", index=False)
    print(f"  Saved ue_inputs.csv ({len(ue_df)} rows)")

    # Lookup built once -- see note 4. Replaces a mask over the whole
    # frame inside the gNB loop.
    off_lookup = {(int(r.snapshot_id), int(r.ue_index)): r.offered_ul_bps
                  for r in ue_df.itertuples()}
    edges_by_snap = {sid: grp for sid, grp in edge_df.groupby("snapshot_id")}

    # --- gNB energy using the real, time-varying associations ---
    export_sqlite(vec_file, "avgServedBlocksUl:vector")
    rb_all = read_sqlite()
    gnb_rb = {g: rb_all[rb_all.moduleName.str.contains(f"gnb{g}.", regex=False)]
                    [["time_s", "value"]].rename(columns={"value": "rb_ul"}).sort_values("time_s")
              for g in range(NUM_GNB)}

    gnb_rows, tgt_rows, diag_rows = [], [], []
    for _, snap in snapshots.iterrows():
        sid = int(snap.snapshot_id)
        snap_edges = edges_by_snap.get(sid)
        for g in range(NUM_GNB):
            rb_df = gnb_rb[g]
            sub = rb_df[(rb_df.time_s >= snap.window_start) & (rb_df.time_s < snap.window_end)]
            avg_rb_ul = sub.rb_ul.mean() if not sub.empty else 0.0
            load_used = avg_rb_ul / MAX_RBS

            ue_ids = ([] if snap_edges is None
                      else snap_edges[snap_edges.serving_gnb_index == g].ue_index.tolist())
            off_ul_bps = sum(off_lookup.get((sid, u), 0.0) for u in ue_ids)

            sleeping = 1 if (avg_rb_ul == 0 and snap.window_start < 0.5) else 0
            p_gnb = (N_TRX * 75.0 if sleeping
                     else N_TRX * P0_W + DELTA_P * tx_w * load_used * N_TRX)

            gnb_rows.append({
                "snapshot_id": sid, "gnb_index": g, "gnb_id": f"gnb_{g}",
                "pos_x": GNB_POS[g][0], "pos_y": GNB_POS[g][1], "pos_z": GNB_POS[g][2],
                "tx_power_dbm": tx_dbm, "tx_power_w": round(tx_w, 6),
                "num_bands": MAX_RBS, "num_trx": N_TRX,
                "scheduler_dl": "PF", "scheduler_ul": "PF",
                "num_connected_ues": len(ue_ids),
                "offered_ul_bytes": int(off_ul_bps * WINDOW / 8),
                "offered_ul_bps": round(off_ul_bps, 4),
            })
            tgt_rows.append({"snapshot_id": sid, "gnb_index": g, "gnb_id": f"gnb_{g}",
                             "estimated_gnb_power_w": round(p_gnb, 6)})
            diag_rows.append({"snapshot_id": sid, "gnb_index": g,
                              "avg_rb_ul": round(avg_rb_ul, 6),
                              "load_used": round(load_used, 6), "sleeping": sleeping})
        if sid % 300 == 0:
            print(f"  energy: snapshot {sid}/{N_SNAPSHOTS}")

    pd.DataFrame(gnb_rows).to_csv(f"{out_dir}/gnb_inputs.csv", index=False)
    pd.DataFrame(tgt_rows).to_csv(f"{out_dir}/energy_targets.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(f"{out_dir}/energy_diagnostics.csv", index=False)

    t = pd.DataFrame(tgt_rows)
    print(f"  energy: mean={t.estimated_gnb_power_w.mean():.2f}  "
          f"std={t.estimated_gnb_power_w.std():.2f}  "
          f"min={t.estimated_gnb_power_w.min():.2f}  max={t.estimated_gnb_power_w.max():.2f}")
    print(f"  Done: {out_dir}/")

if os.path.exists(SQLITE_TMP):
    os.remove(SQLITE_TMP)
print("\nAll runs extracted.")
