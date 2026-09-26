import subprocess
import sqlite3
import pandas as pd
import numpy as np
import random
import math
import os

NUM_UES = 30
MAX_RBS = 50
N_TRX = 2
P0_W = 130.0
DELTA_P = 4.7
N_SNAPSHOTS = 1200
WINDOW = 0.1
MAP_SIZE = 2000.0
SQLITE_TMP = "/tmp/s25.sqlite"

CONFIGS = {
    "p37": ("results/Scenario07s25_p37/0.vec", 37),
    "p43": ("results/Scenario07s25_p43/0.vec", 43),
    "p46": ("results/Scenario07s25_p46/0.vec", 46),
}

GNB_POS = {0: (500, 500, 30), 1: (1500, 500, 30), 2: (500, 1500, 30), 3: (1500, 1500, 30)}

# --- EXACT reconstruction of the ini generator's random sequence ---
# (same seed=21, same call order: 30x position/mobility draws, THEN
# 30x qos draws -- must match exactly or qos_class would be wrong)
random.seed(25)

def closest_gnb(x, y):
    best_g, best_d = None, float('inf')
    for g, (gx, gy, gz) in GNB_POS.items():
        d = math.hypot(x - gx, y - gy)
        if d < best_d:
            best_d, best_g = d, g
    return best_g

ue_mobility = {}
ue_home_gnb = {}
for u in range(30):
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

qos_assignments = {u: assign_qos() for u in range(30)}

print("Reconstructed assignment (verify against your saved printout):")
for u in range(30):
    print(f"  UE {u}: home_gnb={ue_home_gnb[u]}, qos={qos_assignments[u]}")


def reflect_1d(pos, size):
    period = 2 * size
    p_mod = pos % period
    return period - p_mod if p_mod > size else p_mod

def reconstruct_position(u, t):
    m = ue_mobility[u]
    hr = math.radians(m["heading_deg"])
    vx = m["speed"] * math.cos(hr)
    vy = m["speed"] * math.sin(hr)
    x = reflect_1d(m["x0"] + vx * t, MAP_SIZE)
    y = reflect_1d(m["y0"] + vy * t, MAP_SIZE)
    return x, y


def export_sqlite(vec_file, name_pattern, module_pattern=None):
    filt = f"name =~ {name_pattern}*"
    if module_pattern:
        filt += f" AND module =~ {module_pattern}"
    cmd = ["opp_scavetool", "export", "-T", "v", "-f", filt,
           "-F", "SqliteVectorFile", "-o", SQLITE_TMP, vec_file]
    subprocess.run(cmd, capture_output=True)

def read_sqlite(module_contains=None):
    conn = sqlite3.connect(SQLITE_TMP)
    q = """SELECT v.moduleName, d.simtimeRaw/1e12 AS time_s, d.value
           FROM vectorData d JOIN vector v ON d.vectorId = v.vectorId"""
    if module_contains:
        q += f" WHERE v.moduleName LIKE '%{module_contains}%'"
    q += " ORDER BY v.moduleName, d.simtimeRaw"
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df

def make_snapshots():
    starts = np.arange(0, N_SNAPSHOTS * WINDOW, WINDOW)
    return pd.DataFrame({"snapshot_id": np.arange(N_SNAPSHOTS),
                          "window_start": starts, "window_end": starts + WINDOW})

snapshots = make_snapshots()

for run_name, (vec_file, tx_dbm) in CONFIGS.items():
    print(f"\n=== Processing {run_name} (tx={tx_dbm}dBm) ===")
    tx_w = 10 ** (tx_dbm / 10) / 1000
    out_dir = f"results/extracted_s25_{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Real servingCell (ground truth for handover) ---
    export_sqlite(vec_file, "servingCell:vector", module_pattern="**.nrPhy")
    serving_all = read_sqlite()

    edge_rows = []
    for u in range(NUM_UES):
        ue_serving = serving_all[serving_all.moduleName.str.contains(f"ue[{u}].", regex=False)]
        ue_serving = ue_serving.sort_values("time_s")
        for _, snap in snapshots.iterrows():
            sid = int(snap.snapshot_id)
            mask = (ue_serving.time_s >= snap.window_start) & (ue_serving.time_s < snap.window_end)
            sub = ue_serving[mask]
            serving_val = sub.value.iloc[-1] if not sub.empty else np.nan
            edge_rows.append({"snapshot_id": sid, "ue_index": u, "serving_cell_raw": serving_val})
        if u % 10 == 0:
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

    # --- UE traffic (offered_ul_bps) ---
    export_sqlite(vec_file, "packetSent:vector", module_pattern="**.app[*]")
    tput_all = read_sqlite()
    tput_by_u = {u: tput_all[tput_all.moduleName.str.contains(f"ue[{u}].", regex=False)]
                 for u in range(NUM_UES)}

    ue_rows = []
    for u in range(NUM_UES):
        ue_tput = tput_by_u[u]
        for _, snap in snapshots.iterrows():
            sid = int(snap.snapshot_id)
            t_mid = (snap.window_start + snap.window_end) / 2
            x, y = reconstruct_position(u, t_mid)
            mask = (ue_tput.time_s >= snap.window_start) & (ue_tput.time_s < snap.window_end)
            sub = ue_tput[mask]
            off_ul_bps = (sub.value.sum() * 8 / WINDOW) if not sub.empty else 0.0
            ue_rows.append({
                "snapshot_id": sid, "ue_index": u, "pos_x": x, "pos_y": y, "pos_z": 1.5,
                "speed_mps": ue_mobility[u]["speed"], "qos_class": qos_assignments[u],
                "home_gnb_index": ue_home_gnb[u],
                "offered_ul_bytes": int(off_ul_bps * WINDOW / 8),
                "offered_ul_bps": round(off_ul_bps, 4),
            })
        if u % 10 == 0:
            print(f"  ue_inputs: UE {u}/{NUM_UES}")

    ue_df = pd.DataFrame(ue_rows)
    ue_df.to_csv(f"{out_dir}/ue_inputs.csv", index=False)
    print(f"  Saved ue_inputs.csv ({len(ue_df)} rows)")

    # --- gNB energy (using REAL, time-varying serving associations) ---
    export_sqlite(vec_file, "avgServedBlocksUl:vector")
    rb_all = read_sqlite()
    gnb_rb = {g: rb_all[rb_all.moduleName.str.contains(f"gnb{g}.", regex=False)]
                  [["time_s", "value"]].rename(columns={"value": "rb_ul"}).sort_values("time_s")
              for g in range(4)}

    gnb_rows, tgt_rows, diag_rows = [], [], []
    for _, snap in snapshots.iterrows():
        sid = int(snap.snapshot_id)
        snap_edges = edge_df[edge_df.snapshot_id == sid]
        for g in range(4):
            rb_df = gnb_rb[g]
            mask = (rb_df.time_s >= snap.window_start) & (rb_df.time_s < snap.window_end)
            sub = rb_df[mask]
            avg_rb_ul = sub.rb_ul.mean() if not sub.empty else 0.0
            load_used = avg_rb_ul / MAX_RBS

            ue_ids = snap_edges[snap_edges.serving_gnb_index == g].ue_index.tolist()
            num_conn = len(ue_ids)
            off_ul_bps = ue_df[(ue_df.snapshot_id == sid) &
                                (ue_df.ue_index.isin(ue_ids))]["offered_ul_bps"].sum()

            sleeping = 1 if (avg_rb_ul == 0 and snap.window_start < 0.5) else 0
            if sleeping:
                p_gnb = N_TRX * 75.0
            else:
                p_gnb = N_TRX * P0_W + DELTA_P * tx_w * load_used * N_TRX
            e_window = p_gnb * WINDOW

            gnb_rows.append({
                "snapshot_id": sid, "gnb_index": g, "gnb_id": f"gnb_{g}",
                "pos_x": GNB_POS[g][0], "pos_y": GNB_POS[g][1], "pos_z": GNB_POS[g][2],
                "tx_power_dbm": tx_dbm, "tx_power_w": round(tx_w, 6),
                "num_bands": MAX_RBS, "num_trx": N_TRX,
                "scheduler_dl": "PF", "scheduler_ul": "PF",
                "num_connected_ues": num_conn,
                "offered_ul_bytes": int(off_ul_bps * WINDOW / 8),
                "offered_ul_bps": round(off_ul_bps, 4),
            })
            tgt_rows.append({"snapshot_id": sid, "gnb_index": g, "gnb_id": f"gnb_{g}",
                              "estimated_gnb_power_w": round(p_gnb, 6)})
            diag_rows.append({"snapshot_id": sid, "gnb_index": g,
                               "avg_rb_ul": round(avg_rb_ul, 6), "load_used": round(load_used, 6),
                               "sleeping": sleeping})

    pd.DataFrame(gnb_rows).to_csv(f"{out_dir}/gnb_inputs.csv", index=False)
    pd.DataFrame(tgt_rows).to_csv(f"{out_dir}/energy_targets.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(f"{out_dir}/energy_diagnostics.csv", index=False)

    tgt_df = pd.DataFrame(tgt_rows)
    print(f"  energy: mean={tgt_df.estimated_gnb_power_w.mean():.2f}  "
          f"std={tgt_df.estimated_gnb_power_w.std():.2f}  "
          f"min={tgt_df.estimated_gnb_power_w.min():.2f}  max={tgt_df.estimated_gnb_power_w.max():.2f}")
    print(f"  Done: {out_dir}/")

if os.path.exists(SQLITE_TMP):
    os.remove(SQLITE_TMP)
print("\nAll runs extracted.")
