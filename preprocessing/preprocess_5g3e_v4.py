# -*- coding: utf-8 -*-
"""
preprocess_5g3e_v3.py
---------------------
Fixed version:
  - dl_snr: inf -> fill with 70.0 (max finite observed), no NaN in stats
  - ul_ta / dist_km replaced with pusch_snr_db / pucch_snr_db from gNB JSON
  - Packet loss: jitter-relative-to-throughput (real variation, not all zeros)
  - Energy: normalized by max load per server

UE node features  [8]: rsrp, dl_snr, dl_brate, ul_brate,
                        dl_mcs, ul_mcs, pusch_snr_db, pucch_snr_db
gNB node features [4]: avg_latency, load_norm, energy_W, ue_count
Labels per UE     [4]: energy_W, delay_ms, jitter_bps, packet_loss
Labels per gNB    [1]: avg_latency_ms

Usage:
    python preprocess_5g3e_v3.py --path "C:\\path\\to\\version2" --out "processed_v3"
"""

import os, json, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    import torch
    from torch_geometric.data import HeteroData
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch / torch_geometric not found. Saving as .npz instead.")

parser = argparse.ArgumentParser()
parser.add_argument('--path',   default='.',          help='Root of 5G3E v2 dataset')
parser.add_argument('--out',    default='processed_v3')
parser.add_argument('--window', type=int, default=5,  help='Jitter rolling window')
args = parser.parse_args()

ROOT   = Path(args.path)
OUTDIR = Path(args.out)
OUTDIR.mkdir(parents=True, exist_ok=True)

SEP = '=' * 70
def log(msg): print(msg, flush=True)

# =========================================================================
# STEP 1 — FILE DISCOVERY
# =========================================================================
log(f'\n{SEP}\n  STEP 1: File Discovery\n{SEP}')

gnb_paths = sorted([p for p in ROOT.rglob('*.json') if 'gnb' in p.stem.lower()])
ue_paths  = sorted([p for p in ROOT.rglob('*.csv')
                    if 'ran_level' in str(p).lower()
                    and re.search(r'_ue\d+_', p.stem.lower())
                    and not p.name.endswith('.csv#')])
phy_paths = sorted([p for p in ROOT.rglob('*.csv') if 'server' in p.stem.lower()])

def gnb_id(p):
    return int(re.search(r'gnb(\d+)', p.stem.lower()).group(1))

def ue_id(p):
    return int(re.search(r'_ue(\d+)_', p.stem.lower()).group(1))

def server_id(p):
    return int(re.search(r'server_(\d+)', p.stem.lower()).group(1))

gnb_id_list = sorted([gnb_id(p) for p in gnb_paths])
ue_id_list  = sorted([ue_id(p)  for p in ue_paths])
gnb_idx = {gid: i for i, gid in enumerate(gnb_id_list)}
ue_idx  = {uid: i for i, uid in enumerate(ue_id_list)}

log(f'gNB files   : {len(gnb_paths)}   IDs: {gnb_id_list}')
log(f'UE  files   : {len(ue_paths)}   IDs (first 10): {ue_id_list[:10]}')
log(f'Server files: {len(phy_paths)}')

# =========================================================================
# STEP 2 — LOAD UE CSVs
# =========================================================================
log(f'\n{SEP}\n  STEP 2: Load UE CSVs\n{SEP}')

def load_ue_csv(path):
    try:
        df = pd.read_csv(path, nrows=2)
        df = pd.read_csv(path, sep=';') if len(df.columns) == 1 else pd.read_csv(path)
    except:
        df = pd.read_csv(path, sep=';')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Replace inf in dl_snr with 70.0 (max finite observed)
    if 'dl_snr' in df.columns:
        df['dl_snr'] = df['dl_snr'].replace([np.inf, -np.inf], 70.0)
    return df

ue_dataframes = {ue_id(p): load_ue_csv(p) for p in ue_paths}
log(f'Loaded {len(ue_dataframes)} UE DataFrames')

# =========================================================================
# STEP 3 — LOAD gNB JSONs
# =========================================================================
log(f'\n{SEP}\n  STEP 3: Load gNB JSONs\n{SEP}')

def load_ndjson(path):
    with open(path, encoding='utf-8') as f:
        content = f.read().strip()
    try:
        obj = json.loads(content)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try: records.append(json.loads(line))
            except: pass
    return records

gnb_records = {gnb_id(p): load_ndjson(p) for p in gnb_paths}
log(f'Loaded {len(gnb_records)} gNB record sets')
for gid in sorted(gnb_records)[:3]:
    log(f'  gnb{gid:02d}: {len(gnb_records[gid])} records')

# =========================================================================
# STEP 4 — LOAD PHYSICAL SERVER CSVs
# =========================================================================
log(f'\n{SEP}\n  STEP 4: Physical Servers\n{SEP}')

server_dfs = {server_id(p): pd.read_csv(p) for p in phy_paths}

# gNB -> server mapping (by load group)
GNB_TO_SERVER = {}
for gid in range(1,  6):  GNB_TO_SERVER[gid] = 1
for gid in range(6,  11): GNB_TO_SERVER[gid] = 2
for gid in range(11, 14): GNB_TO_SERVER[gid] = 3

for sid, df in sorted(server_dfs.items()):
    if 'node_load1' in df.columns:
        load  = pd.to_numeric(df['node_load1'], errors='coerce').dropna()
        max_l = load.max()
        norm  = load / max_l
        P_BS  = 130 + (600 - 130) * norm
        log(f'  Server {sid}: load_max={max_l:.2f}  '
            f'P_BS=[{P_BS.min():.1f},{P_BS.max():.1f}]W  mean={P_BS.mean():.1f}W')

# =========================================================================
# STEP 5 — UE-gNB ASSIGNMENT
# =========================================================================
log(f'\n{SEP}\n  STEP 5: UE-gNB Assignment\n{SEP}')

site_to_gnbs = defaultdict(list)
site_to_ues  = defaultdict(list)
for p in gnb_paths:
    site_to_gnbs[p.parent.name].append(gnb_id(p))
for p in ue_paths:
    site_to_ues[p.parent.name].append(ue_id(p))

UE_TO_GNB = {}
for site in sorted(site_to_gnbs):
    gnbs = sorted(site_to_gnbs[site])
    ues  = sorted(site_to_ues[site])
    for i, uid in enumerate(ues):
        UE_TO_GNB[uid] = gnbs[i % len(gnbs)]
    log(f'  {site}: {len(gnbs)} gNBs, {len(ues)} UEs')

gnb_to_ues = defaultdict(list)
for uid, gid in UE_TO_GNB.items():
    gnb_to_ues[gid].append(uid)
log(f'\nUEs per gNB: { {gid: len(v) for gid, v in sorted(gnb_to_ues.items())} }')

# =========================================================================
# STEP 6 — DERIVE LABELS
# =========================================================================
log(f'\n{SEP}\n  STEP 6: Derive Labels\n{SEP}')

# --- Jitter: rolling std of dl_brate per UE ---
ue_jitter = {}
for uid, df in ue_dataframes.items():
    if 'dl_brate' in df.columns:
        vals = pd.to_numeric(df['dl_brate'], errors='coerce')
        j = vals.rolling(window=args.window, min_periods=2).std().fillna(0)
        ue_jitter[uid] = j.values
    else:
        ue_jitter[uid] = np.zeros(len(df))

# --- Per-UE mean dl_brate (used for PL calculation) ---
ue_brate_mean = {}
for uid, df in ue_dataframes.items():
    if 'dl_brate' in df.columns:
        ue_brate_mean[uid] = float(pd.to_numeric(df['dl_brate'], errors='coerce').mean())
    else:
        ue_brate_mean[uid] = 1.0

# --- Synthetic PL: jitter relative to throughput ---
# High jitter / low throughput -> unstable link -> higher packet loss
def synthetic_pl(rsrp, rsrp_min=21.0, rsrp_max=39.0):
    # Lower RSRP (weaker signal) = higher packet loss
    rsrp_clipped = float(np.clip(rsrp, rsrp_min, rsrp_max))
    normalized = (rsrp_clipped - rsrp_min) / (rsrp_max - rsrp_min)
    pl = 0.5 * (1.0 - normalized)   # range [0, 0.5]
    return float(np.clip(pl, 0.001, 0.5))

log('Synthetic PL preview (jitter / brate_mean -> PL):')
for j, b in [(0, 100000), (50000, 100000), (200000, 300000), (600000, 500000), (1000000, 200000)]:
    log(f'  jitter={j:>8}  brate_mean={b:>8} -> PL={synthetic_pl(j, b):.5f}')

# --- Energy per gNB from server ---
gnb_energy = {}
for gid in gnb_id_list:
    sid = GNB_TO_SERVER[gid]
    df  = server_dfs[sid]
    if 'node_load1' in df.columns:
        load  = pd.to_numeric(df['node_load1'], errors='coerce').fillna(0)
        max_l = load.max()
        norm  = load / max_l if max_l > 0 else load
        gnb_energy[gid] = (130 + (600 - 130) * norm).values
    else:
        gnb_energy[gid] = np.full(2000, 300.0)

# --- pusch/pucch SNR from gNB JSON ue_list ---
log('Building pusch/pucch SNR lookup from gNB JSON ue_list...')
gnb_ue_snr = {}
for gid, records in gnb_records.items():
    snr_by_t = []
    for rec in records:
        ue_map = {}
        for ue_entry in rec.get('ue_list', []):
            cont  = ue_entry.get('ue_container', {})
            rnti  = cont.get('rnti')
            pusch = cont.get('pusch_snr_db', np.nan)
            pucch = cont.get('pucch_snr_db', np.nan)
            if rnti is not None:
                ue_map[rnti] = {'pusch': pusch, 'pucch': pucch}
        snr_by_t.append(ue_map)
    gnb_ue_snr[gid] = snr_by_t
log('pusch/pucch lookup built.')

log(f'Jitter: {len(ue_jitter)} UEs  |  Energy: {len(gnb_energy)} gNBs')

# =========================================================================
# STEP 7 — BUILD GRAPH SNAPSHOTS
# =========================================================================
log(f'\n{SEP}\n  STEP 7: Build Graph Snapshots\n{SEP}')

N_UE  = len(ue_id_list)
N_GNB = len(gnb_id_list)
n_ue_rows  = min(len(df) for df in ue_dataframes.values())
n_gnb_rows = min(len(r)  for r  in gnb_records.values())
N_SNAP = min(n_ue_rows, n_gnb_rows)
log(f'N_UE={N_UE}  N_GNB={N_GNB}  N_SNAP={N_SNAP}')

# Static edges: UE -> gNB
ue_src, gnb_dst = [], []
for uid, gid in UE_TO_GNB.items():
    ue_src.append(ue_idx[uid])
    gnb_dst.append(gnb_idx[gid])
edge_ue_gnb = np.array([ue_src, gnb_dst], dtype=np.int64)

# Static edges: gNB -> gNB (same site = interference)
gnb_src_i, gnb_dst_i = [], []
for site, gnbs in site_to_gnbs.items():
    for g1 in gnbs:
        for g2 in gnbs:
            if g1 != g2:
                gnb_src_i.append(gnb_idx[g1])
                gnb_dst_i.append(gnb_idx[g2])
edge_gnb_gnb = np.array([gnb_src_i, gnb_dst_i], dtype=np.int64)

log(f'UE->gNB edges: {edge_ue_gnb.shape[1]}  |  gNB->gNB edges: {edge_gnb_gnb.shape[1]}')

for t in range(N_SNAP):

    # ── UE node features & labels ─────────────────────────────────────
    ue_feat   = np.zeros((N_UE, 8), dtype=np.float32)
    ue_labels = np.zeros((N_UE, 4), dtype=np.float32)

    for uid in ue_id_list:
        ni  = ue_idx[uid]
        row = ue_dataframes[uid].iloc[t]
        gid = UE_TO_GNB[uid]

        rsrp     = float(row.get('rsrp',     0) or 0)
        dl_snr   = float(row.get('dl_snr',   70.0) or 70.0)
        dl_brate = float(row.get('dl_brate', 0) or 0)
        ul_brate = float(row.get('ul_brate', 0) or 0)
        dl_mcs   = float(row.get('dl_mcs',   0) or 0)
        ul_mcs   = float(row.get('ul_mcs',   0) or 0)

        # pusch/pucch: mean across all UEs on this gNB at timestep t
        ue_map = gnb_ue_snr[gid][min(t, len(gnb_ue_snr[gid]) - 1)]
        rntis  = list(ue_map.keys())
        if rntis:
            pusch = float(np.nanmean([ue_map[r]['pusch'] for r in rntis]))
            pucch = float(np.nanmean([ue_map[r]['pucch'] for r in rntis]))
        else:
            pusch, pucch = 65.0, 55.0

        ue_feat[ni] = [rsrp, dl_snr, dl_brate, ul_brate,
                       dl_mcs, ul_mcs, pusch, pucch]

        # Labels
        energy_t   = int(t * len(gnb_energy[gid]) / N_SNAP)
        energy_val = float(gnb_energy[gid][energy_t])

        rec        = gnb_records[gid][min(t, len(gnb_records[gid]) - 1)]
        delay_val  = float(rec.get('cell_metrics', {}).get('average_latency', 0) or 0)
        
        jitter_val = float(ue_jitter[uid][t]) if t < len(ue_jitter[uid]) else 0.0
        jitter_val = float(np.log1p(jitter_val))
        pl_val = synthetic_pl(rsrp)
        ue_labels[ni] = [energy_val, delay_val, jitter_val, pl_val]
    # ── gNB node features & labels ────────────────────────────────────
    gnb_feat   = np.zeros((N_GNB, 4), dtype=np.float32)
    gnb_labels = np.zeros((N_GNB, 1), dtype=np.float32)

    for gid in gnb_id_list:
        ni     = gnb_idx[gid]
        rec    = gnb_records[gid][min(t, len(gnb_records[gid]) - 1)]
        lat    = float(rec.get('cell_metrics', {}).get('average_latency', 0) or 0)
        n_ue_c = float(len(rec.get('ue_list', [])))

        sid    = GNB_TO_SERVER[gid]
        df_srv = server_dfs[sid]
        if 'node_load1' in df_srv.columns:
            srv_t  = int(t * len(df_srv) / N_SNAP)
            load   = pd.to_numeric(df_srv['node_load1'], errors='coerce').iloc[srv_t]
            max_l  = pd.to_numeric(df_srv['node_load1'], errors='coerce').max()
            load_n = float(load / max_l) if max_l > 0 else 0.0
        else:
            load_n = 0.0

        energy_t   = int(t * len(gnb_energy[gid]) / N_SNAP)
        energy_val = float(gnb_energy[gid][energy_t])

        gnb_feat[ni]   = [lat, load_n, energy_val, n_ue_c]
        gnb_labels[ni] = [lat]

    # ── Save snapshot ─────────────────────────────────────────────────
    snap = {
        'ue_feat':      ue_feat,
        'gnb_feat':     gnb_feat,
        'ue_labels':    ue_labels,
        'gnb_labels':   gnb_labels,
        'edge_ue_gnb':  edge_ue_gnb,
        'edge_gnb_gnb': edge_gnb_gnb,
        't':            np.array([t])
    }

    if HAS_TORCH:
        data = HeteroData()
        data['ue'].x  = torch.tensor(ue_feat)
        data['gnb'].x = torch.tensor(gnb_feat)
        data['ue'].y  = torch.tensor(ue_labels)
        data['gnb'].y = torch.tensor(gnb_labels)
        data['ue',  'connects',     'gnb'].edge_index = torch.tensor(edge_ue_gnb)
        data['gnb', 'rev_connects', 'ue' ].edge_index = torch.tensor(edge_ue_gnb[[1, 0]])
        data['gnb', 'interferes',   'gnb'].edge_index = torch.tensor(edge_gnb_gnb)
        torch.save(data, OUTDIR / f'snapshot_{t:04d}.pt')
    else:
        np.savez_compressed(OUTDIR / f'snapshot_{t:04d}.npz', **snap)

    if t % 50 == 0 or t == N_SNAP - 1:
        log(f'  t={t:4d}/{N_SNAP-1}  '
            f'energy={ue_labels[0,0]:.1f}W  '
            f'delay={ue_labels[0,1]:.1f}ms  '
            f'jitter={ue_labels[0,2]:.0f}  '
            f'pl={ue_labels[0,3]:.5f}')

# =========================================================================
# STEP 8 — TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# =========================================================================
log(f'\n{SEP}\n  STEP 8: Train/Val/Test Split\n{SEP}')

n_train   = int(0.70 * N_SNAP)
n_val     = int(0.15 * N_SNAP)
n_test    = N_SNAP - n_train - n_val
train_idx = list(range(n_train))
val_idx   = list(range(n_train, n_train + n_val))
test_idx  = list(range(n_train + n_val, N_SNAP))
log(f'Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}')

ext = '.pt' if HAS_TORCH else '.npz'
if HAS_TORCH:
    torch.save({'train': train_idx, 'val': val_idx, 'test': test_idx},
               OUTDIR / 'split.pt')
else:
    np.savez(OUTDIR / 'split.npz', train=train_idx, val=val_idx, test=test_idx)

# =========================================================================
# STEP 9 — NORMALIZATION STATISTICS  (train set only)
# =========================================================================
log(f'\n{SEP}\n  STEP 9: Normalization Statistics\n{SEP}')

all_ue, all_gnb, all_ue_y = [], [], []
for t in train_idx:
    fpath = OUTDIR / f'snapshot_{t:04d}{ext}'
    if HAS_TORCH:
        d = torch.load(fpath)
        all_ue.append(d['ue'].x.numpy())
        all_gnb.append(d['gnb'].x.numpy())
        all_ue_y.append(d['ue'].y.numpy())
    else:
        d = np.load(fpath)
        all_ue.append(d['ue_feat'])
        all_gnb.append(d['gnb_feat'])
        all_ue_y.append(d['ue_labels'])

ue_stack   = np.concatenate(all_ue,   axis=0)
gnb_stack  = np.concatenate(all_gnb,  axis=0)
ue_y_stack = np.concatenate(all_ue_y, axis=0)

ue_mean,  ue_std  = ue_stack.mean(0),   np.where(ue_stack.std(0)   > 0, ue_stack.std(0),   1.0)
gnb_mean, gnb_std = gnb_stack.mean(0),  np.where(gnb_stack.std(0)  > 0, gnb_stack.std(0),  1.0)
y_mean,   y_std   = ue_y_stack.mean(0), np.where(ue_y_stack.std(0) > 0, ue_y_stack.std(0), 1.0)

ue_col_names  = ['rsrp', 'dl_snr', 'dl_brate', 'ul_brate',
                 'dl_mcs', 'ul_mcs', 'pusch_snr', 'pucch_snr']
gnb_col_names = ['avg_latency', 'load_norm', 'energy_W', 'ue_count']
label_names   = ['energy_W', 'delay_ms', 'jitter_bps', 'packet_loss']

log('UE feature stats (mean | std):')
for i, name in enumerate(ue_col_names):
    log(f'  {name:<15}: mean={ue_mean[i]:.3f}  std={ue_std[i]:.3f}')

log('\ngNB feature stats:')
for i, name in enumerate(gnb_col_names):
    log(f'  {name:<15}: mean={gnb_mean[i]:.3f}  std={gnb_std[i]:.3f}')

log('\nLabel stats (UE):')
for i, name in enumerate(label_names):
    log(f'  {name:<15}: mean={y_mean[i]:.5f}  std={y_std[i]:.5f}')

stats = {
    'ue_mean':  ue_mean,  'ue_std':  ue_std,
    'gnb_mean': gnb_mean, 'gnb_std': gnb_std,
    'y_mean':   y_mean,   'y_std':   y_std,
    'ue_cols':  ue_col_names,
    'gnb_cols': gnb_col_names,
    'label_cols': label_names
}

if HAS_TORCH:
    torch.save(stats, OUTDIR / 'norm_stats.pt')
else:
    np.savez(OUTDIR / 'norm_stats.npz',
             ue_mean=ue_mean, ue_std=ue_std,
             gnb_mean=gnb_mean, gnb_std=gnb_std,
             y_mean=y_mean, y_std=y_std)

# =========================================================================
log(f'\n{SEP}')
log(f'  DONE  |  {N_SNAP} snapshots  ->  {OUTDIR.resolve()}')
log(f'  Format: {"HeteroData .pt" if HAS_TORCH else "numpy .npz"}')
log(SEP)
