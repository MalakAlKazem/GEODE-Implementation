# src/v3/preprocess_v3.py
# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
WirelessDT-6G v3  ─  C1: Graph Construction (Block 1 / 5)
==========================================================
Converts raw 5G3E telemetry into heterogeneous graph snapshots and saves
them as PyTorch Geometric HeteroData objects (.pt files).

WHY WE BUILD A HETEROGENEOUS GRAPH
────────────────────────────────────
The RAN has three physically distinct entity types:
  • UEs  – user devices measuring signal from their serving gNB
  • gNBs – base stations serving UEs, interfering with each other
  • Srvs – servers running gNB software; their CPU load = energy

A homogeneous GNN would force all these to share the same weights.
A *heterogeneous* GNN assigns each node type its OWN linear encoder
and each edge type its OWN message-passing kernel.

GRAPH STRUCTURE (one snapshot)
────────────────────────────────
Node types
  ue  [65 nodes, R^7] : [rsrp, dl_snr, dl_brate, ul_brate, dl_mcs, ul_mcs, ul_buff]
  gnb [13 nodes, R^5] : [nof_ue, site_0, site_1, site_2, mean_rsrp]
  srv [ 3 nodes, R^1] : [cpu_util]

Edge types
  (ue,  connects,    gnb)  UE served by gNB           — E1 serving link
  (gnb, rev_connects, ue)  reverse of above            — needed for bidir SAGE
  (gnb, interferes,  gnb)  same-site gNB pairs        — DGAT attention here (E2)
  (gnb, handover,    gnb)  cross-site gNB pairs       — CMOA sleep safety (E3)
  (gnb, backhaul,    srv)  gNB → server               — energy routes this way (E4)
  (srv, rev_backhaul,gnb)  reverse backhaul            — server messages gNBs

WHY EACH LABEL IS NOT LEAKY
─────────────────────────────
  energy_W    = 130 + 470 × cpu_util
    cpu_util is a SERVER node feature, not a gNB feature.
    Energy information must flow through the (srv→gnb) backhaul edge.
    The GNN must *learn* the routing — it cannot shortcut.

  delay_ms    = cell_metrics.average_latency (from gNB JSON)
    NOT in the gNB node feature vector (old code leaked this: gnb_feat[0]=lat).

  jitter_ms   = rolling_std(average_latency, window=5 gNB records)
    Derived ONLY from the delay label itself over consecutive snapshots.
    None of the node features are average_latency, so no leakage.

  packet_loss = mean( dl_nof_nok / (dl_nof_ok + dl_nof_nok) ) over ue_list
    Computed from gNB JSON counters, not from RSRP (old code leaked via
    the synthetic formula PL=0.5*(1-(rsrp-21)/18) where rsrp is an input).

WHAT THE OLD CODE GOT WRONG
─────────────────────────────
  OLD gNB features = [avg_latency, load_norm, nof_ue]
    avg_latency   → LEAKY: it equals the delay label
    load_norm     → LEAKY: energy label ∝ load_norm

  OLD jitter     = log1p(rolling_std(dl_brate))
    dl_brate is a UE *input* feature → model just reads it and takes std

  OLD PL          = 0.5*(1-(rsrp-21)/18)
    rsrp is a UE *input* feature → pure function of the input

  OLD predictions = per UE (65 outputs each with 4 labels)
    Energy/delay are gNB-level quantities averaged to UE — not meaningful.

  In v3 we fix all of this.

USAGE
──────
  python preprocess_5g3e.py                          # Day 1 only (default)
  python preprocess_5g3e.py --days 1 2               # Days 1 + 2
  python preprocess_5g3e.py --sample_every 5         # coarser sampling
  python preprocess_5g3e.py --days 1 2 --test_days 3 # thesis protocol

  Raw 5G3E folder (containing day_1/, day_2/, day_3/): $GEODE_5G3E_RAW,
  default data/5g3e_raw. Output: $GEODE_5G3E, default data/5g3e_processed.
"""

import argparse
import json
import os
import math
import re
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ─ CONFIGURATION
# All magic numbers and paths live here. Change once, pipeline updates.
# ══════════════════════════════════════════════════════════════════════════════

_REPO       = Path(__file__).resolve().parent.parent
BASE_RAW    = Path(os.environ.get('GEODE_5G3E_RAW', _REPO / 'data' / '5g3e_raw'))
OUT_DIR     = Path(os.environ.get('GEODE_5G3E', _REPO / 'data' / '5g3e_processed'))
SAMPLE_EVERY = 3      # keep 1 of every N gNB records → ~7 000 snapshots/day
JITTER_WIN   = 5      # rolling window (in gNB records) for std(latency) = jitter
N_CORES      = 32     # server CPU cores (cpu_0 to cpu_31 seen in UE CSV)
P_IDLE       = 130.0  # 3GPP TR 36.814 idle power (W)
P_DELTA      = 470.0  # 3GPP TR 36.814: P_max - P_idle = 600 - 130 = 470 W
VAL_FRAC     = 0.15   # fraction carved from end of train days for validation

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ─ STATIC NETWORK TOPOLOGY
# These mappings are derived from the 5G3E testbed design, NOT from the data.
# They define the STRUCTURE of every graph snapshot we build.
# ══════════════════════════════════════════════════════════════════════════════

# Site mapping: which gNBs are at which physical site?
# One-hot encoding per gNB: [is_site1_2, is_site3, is_site4]
# WHY: gNBs at the same site share physical space → strong interference.
#      The DGAT layer learns interference weights along intra-site edges.
GNB_SITE_NAME = {
    1:  'site1_2', 4:  'site1_2', 7:  'site1_2', 10: 'site1_2', 13: 'site1_2',
    2:  'site3',   5:  'site3',   8:  'site3',    11: 'site3',
    3:  'site4',   6:  'site4',   9:  'site4',    12: 'site4',
}
SITE_IDX = {'site1_2': 0, 'site3': 1, 'site4': 2}

# Backhaul mapping: which server runs each gNB's srsRAN software?
# SOURCE: 5G3E paper (Phung et al., 6GNet 2022):
#   "Three powerful servers connected in a triangle simulate 4 sites.
#    Sites 1 and 2 share one server, Site 3 has one, Site 4 has one."
# → Server assignment follows SITE, not sequential gNB numbering.
# WHY: server CPU → energy. Energy flows through (srv→gnb) backhaul edge.
#      GNN must learn to route this signal through the graph.
GNB_TO_SRV = {
    1: 0, 4: 0, 7: 0, 10: 0, 13: 0,  # site1_2 gNBs → Server_1 (srv idx 0)
    2: 1, 5: 1, 8: 1, 11: 1,          # site3   gNBs → Server_2 (srv idx 1)
    3: 2, 6: 2, 9: 2, 12: 2,          # site4   gNBs → Server_3 (srv idx 2)
}

N_GNB = 13
N_SRV = 3

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ─ FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_files(day_root: Path):
    """
    Walk one day's folder tree and return file maps.

    5G3E folder layout:
      day_N/
        RAN_level/
          site1_2/  gnb01_metrics_dayN.json  ue01_metrics_dayN.csv  ...
          site3/    gnb02_metrics_dayN.json  ue06_metrics_dayN.csv  ...
          site4/    gnb03_metrics_dayN.json  ue11_metrics_dayN.csv  ...
        physical_level/
          Server_1.csv  Server_2.csv  Server_3.csv  ...

    Returns:
      gnb_files  {gnb_id: Path}      e.g. {1: .../gnb01_metrics_day1.json}
      ue_files   {ue_id:  Path}      e.g. {1: .../ue01_metrics_day1.csv}
      srv_files  {srv_idx: Path}     e.g. {0: .../Server_1.csv}
      site_of_ue {ue_id: site_name}  folder name of each UE file
      site_of_gnb{gnb_id: site_name} folder name of each gNB file
    """
    gnb_files, ue_files, site_of_ue, site_of_gnb = {}, {}, {}, {}

    # Walk RAN_level sub-folders
    ran_root = day_root / 'RAN_level'
    for json_path in sorted(ran_root.rglob('gnb*_metrics_day*.json')):
        m = re.search(r'gnb(\d+)', json_path.stem.lower())
        if m:
            gid = int(m.group(1))
            gnb_files[gid] = json_path
            site_of_gnb[gid] = json_path.parent.name

    for csv_path in sorted(ran_root.rglob('ue*_metrics_day*.csv')):
        m = re.search(r'ue(\d+)', csv_path.stem.lower())
        if m:
            uid = int(m.group(1))
            ue_files[uid] = csv_path
            site_of_ue[uid] = csv_path.parent.name

    # Walk physical_level for server CSVs (we need servers 1, 2, 3)
    srv_files = {}
    phys_root = day_root / 'physical_level'
    for csv_path in sorted(phys_root.glob('Server_*.csv')):
        m = re.search(r'Server_(\d+)', csv_path.name)
        if m:
            sid = int(m.group(1))
            if sid in (1, 2, 3):               # only the 3 mapped servers
                srv_files[sid - 1] = csv_path  # convert to 0-indexed

    print(f'  Found {len(gnb_files)} gNBs, {len(ue_files)} UEs, {len(srv_files)} servers')
    return gnb_files, ue_files, srv_files, site_of_ue, site_of_gnb


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ─ STATIC EDGE CONSTRUCTION
# Edges that do NOT change between snapshots. Built once, reused everywhere.
# ══════════════════════════════════════════════════════════════════════════════

def build_static_edges(gnb_id_list, ue_id_list, site_of_ue, site_of_gnb):
    """
    Returns a dict of edge_index tensors for all 6 edge types.

    UE→gNB assignment (serving edges, E1)
    ───────────────────────────────────────
    In the 5G3E testbed all gNBs share PCI=1 (same PCI reuse across cells).
    We cannot use the pci column to identify serving gNBs.

    Instead we use the FOLDER CO-LOCATION principle:
      • UEs in the site1_2 folder are served by gNBs in the site1_2 folder.
      • Within a site, we assign UEs to gNBs in ROUND-ROBIN order by ID.

    Example for site1_2 (5 gNBs: 01,04,07,10,13 / 25 UEs: 01..05,16..20,...):
      ue01→gnb01, ue02→gnb04, ue03→gnb07, ue04→gnb10, ue05→gnb13,
      ue16→gnb01, ue17→gnb04, ...   (5 UEs per gNB, interspersed)

    gNB→gNB interference edges (E2, DGAT target)
    ──────────────────────────────────────────────
    In a real network: connect gNB_i → gNB_j iff |I_ij| ≥ -110 dBm.
    In 5G3E testbed: no inter-gNB RSRP measurements available.
    Proxy: connect ALL pairs within the SAME SITE (they share physical space
    and spectrum → maximum co-site interference).
    DGAT (GATv2Conv) will learn attention weights to distinguish strong
    vs weak interferers within this set.

    gNB→gNB handover edges (E3)
    ────────────────────────────
    Connect ALL pairs across DIFFERENT SITES.
    WHY: in CMOA (Block 4), sleeping a gNB redirects its UEs to a neighbor.
    Cross-site neighbors are the natural candidates for UE absorption.

    gNB→srv backhaul edges (E4)
    ────────────────────────────
    One directed edge per gNB using GNB_TO_SRV.
    WHY: this is the path through which energy information reaches gNB nodes.
    Without this edge, the model would never see the server's cpu_util.
    """
    gnb_idx = {g: i for i, g in enumerate(sorted(gnb_id_list))}
    ue_idx  = {u: i for i, u in enumerate(sorted(ue_id_list))}

    # ── E1: UE → gNB serving (round-robin within site) ──────────────────────
    ue_to_gnb = {}
    site_gnbs = defaultdict(list)  # site_name → sorted list of gnb_ids
    site_ues  = defaultdict(list)  # site_name → sorted list of ue_ids

    for gid in sorted(gnb_id_list):
        site_gnbs[site_of_gnb[gid]].append(gid)
    for uid in sorted(ue_id_list):
        site_ues[site_of_ue[uid]].append(uid)

    for site in sorted(site_gnbs):
        gs = sorted(site_gnbs[site])
        us = sorted(site_ues[site])
        for i, u in enumerate(us):
            ue_to_gnb[u] = gs[i % len(gs)]   # round-robin: ue_i → gNB[i mod |gNBs|]

    src_ue  = [ue_idx[u]         for u in sorted(ue_id_list)]
    dst_gnb = [gnb_idx[ue_to_gnb[u]] for u in sorted(ue_id_list)]
    e_serve     = torch.tensor([src_ue,  dst_gnb], dtype=torch.long)
    e_rev_serve = torch.tensor([dst_gnb, src_ue],  dtype=torch.long)

    # ── E2: gNB → gNB interference (same-site, all pairs) ───────────────────
    intra_src, intra_dst = [], []
    for site, gs in site_gnbs.items():
        for a in gs:
            for b in gs:
                if a != b:
                    intra_src.append(gnb_idx[a])
                    intra_dst.append(gnb_idx[b])
    e_interfere = torch.tensor([intra_src, intra_dst], dtype=torch.long)

    # ── E3: gNB → gNB handover (cross-site, all pairs) ──────────────────────
    inter_src, inter_dst = [], []
    gnb_sorted = sorted(gnb_id_list)
    for a in gnb_sorted:
        for b in gnb_sorted:
            if a != b and site_of_gnb[a] != site_of_gnb[b]:
                inter_src.append(gnb_idx[a])
                inter_dst.append(gnb_idx[b])
    e_handover = torch.tensor([inter_src, inter_dst], dtype=torch.long)

    # ── E4: gNB → srv backhaul (from GNB_TO_SRV mapping) ────────────────────
    bh_src, bh_dst = [], []
    for gid in gnb_sorted:
        bh_src.append(gnb_idx[gid])
        bh_dst.append(GNB_TO_SRV[gid])
    e_backhaul     = torch.tensor([bh_src, bh_dst], dtype=torch.long)
    e_rev_backhaul = torch.tensor([bh_dst, bh_src], dtype=torch.long)

    print(f'  Edges — E1 serving: {e_serve.shape[1]} | '
          f'E2 interfere: {e_interfere.shape[1]} | '
          f'E3 handover: {e_handover.shape[1]} | '
          f'E4 backhaul: {e_backhaul.shape[1]}')

    return {
        'e_serve':        e_serve,
        'e_rev_serve':    e_rev_serve,
        'e_interfere':    e_interfere,
        'e_handover':     e_handover,
        'e_backhaul':     e_backhaul,
        'e_rev_backhaul': e_rev_backhaul,
        'ue_to_gnb':      ue_to_gnb,
        'gnb_idx':        gnb_idx,
        'ue_idx':         ue_idx,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ─ RAW DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_gnb_records(gnb_files: dict, sample_every: int):
    """
    Read each gNB NDJSON file, keep every sample_every-th record.

    NDJSON format: each line is one JSON object (one 4-second snapshot).
    We subsample to reduce from ~21 000 records/day to ~7 000.

    Returns: {gnb_id: [record_0, record_1, ...]}
    """
    gnb_records = {}
    for gid, path in gnb_files.items():
        records = []
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % sample_every != 0:
                    continue
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        gnb_records[gid] = records

    n_min = min(len(v) for v in gnb_records.values())
    n_max = max(len(v) for v in gnb_records.values())
    print(f'  gNB records per file: min={n_min}, max={n_max}  → using {n_min} snapshots')
    return gnb_records, n_min


def load_ue_data(ue_files: dict, row_indices: list):
    """
    Read each UE CSV (semicolon-delimited), extract only the rows we need.

    TIME ALIGNMENT:
    The gNB JSON timestamp is Unix seconds; UE CSV 'time' is milliseconds
    from an internal start counter — different reference frames.
    Solution: we use INDEX-BASED alignment.
      If gNBs have N_snap records and UEs have ~4N_snap rows,
      gNB snapshot t → UE row index = round(t × UE_len / N_snap).
    The caller precomputes row_indices with np.linspace for this.

    Returns: {ue_id: DataFrame with len(row_indices) rows}
    """
    ue_data = {}
    for uid, path in ue_files.items():
        try:
            df = pd.read_csv(path, sep=';')
        except Exception:
            df = pd.read_csv(path)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Replace ±inf in dl_snr (can appear when signal is perfect)
        if 'dl_snr' in df.columns:
            df['dl_snr'] = df['dl_snr'].replace([np.inf, -np.inf], 70.0)

        # Clip row indices to valid range
        valid_idx = [min(i, len(df) - 1) for i in row_indices]
        ue_data[uid] = df.iloc[valid_idx].reset_index(drop=True)

    return ue_data


def load_server_cpu(srv_files: dict, n_snap: int):
    """
    Extract cpu_util from each server CSV using node_load1.

    cpu_util proxy: node_load1 / N_CORES   (clipped to [0, 1])
    WHY node_load1: it's the Linux 1-minute load average.
      load_avg = 32 on a 32-core server means 100% utilization.
      More accurate would be node_cpu_seconds_total deltas, but that
      requires two-pass processing of 3.6 GB files. node_load1 is
      a well-correlated proxy that fits in memory.

    IMPORTANT: cpu_util goes to the SERVER node feature vector (R^1).
    It does NOT appear in gNB features — avoiding the leakage present
    in the old code where load_norm appeared in gNB_feat[1].

    Returns: {srv_idx: np.array of shape (n_snap,)}
    """
    srv_cpu = {}
    for sidx, path in srv_files.items():
        loads = []
        try:
            for chunk in pd.read_csv(path, usecols=['node_load1'], chunksize=200_000):
                arr = pd.to_numeric(chunk['node_load1'], errors='coerce').values
                loads.append(arr)
        except ValueError:
            # Fallback: scan chunks for any column containing 'node_load1'
            for chunk in pd.read_csv(path, chunksize=200_000, nrows=200_000):
                col = next((c for c in chunk.columns if 'node_load1' in c.lower()), None)
                if col:
                    loads.append(pd.to_numeric(chunk[col], errors='coerce').values)
                break

        if not loads:
            print(f'  [WARN] Server {sidx}: node_load1 not found, using cpu_util=0.5')
            srv_cpu[sidx] = np.full(n_snap, 0.5, dtype=np.float32)
            continue

        full = np.concatenate(loads)
        full = full[~np.isnan(full)]

        # Downsample to n_snap by linear index interpolation
        idx = np.linspace(0, len(full) - 1, n_snap).astype(int)
        sampled = full[idx]

        # Convert load average → relative cpu_util fraction.
        # WHY NOT /N_CORES: srsRAN servers show load1 >> N_CORES (e.g. 48 on 32-core)
        # because load average counts I/O-bound threads too. Dividing by N_CORES would
        # always clip to 1.0 (no variation). Instead we normalize by the observed
        # maximum across this day's data, giving a relative utilization that captures
        # intra-day variation — the same approach as the baseline code.
        load_max = float(np.nanmax(sampled)) if np.nanmax(sampled) > 0 else 1.0
        cpu = (sampled / load_max).clip(0.0, 1.0).astype(np.float32)
        srv_cpu[sidx] = cpu

        print(f'  Server {sidx}: node_load1 mean={np.mean(sampled):.2f}  '
              f'max={load_max:.2f}  cpu_util mean={np.mean(cpu):.3f} std={np.std(cpu):.3f}')

    return srv_cpu


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ─ LABEL EXTRACTORS
# Per-gNB ground-truth values derived from raw data (no UE features used).
# ══════════════════════════════════════════════════════════════════════════════

def extract_gnb_labels(gnb_records: dict, gnb_id_list: list,
                       srv_cpu: dict, n_snap: int):
    """
    Build label arrays for all gNBs across all snapshots.

    Returns:
      labels[gnb_id] = np.array shape (n_snap, 4)
        col 0: energy_W     = P_IDLE + P_DELTA × cpu_util    [Watts]
        col 1: delay_ms     = cell_metrics.average_latency    [ms]
        col 2: jitter_ms    = rolling_std(delay, win=JITTER_WIN) [ms]
        col 3: packet_loss  = mean(dl_nof_nok/(ok+nok)) over ue_list [0..1]

    energy_W:
      3GPP TR 36.814 linear power model.
      130W idle, 600W full load. cpu_util from server, NOT from gNB features.
      The GNN discovers this relationship by routing srv→gnb messages.

    delay_ms:
      Direct read from gNB JSON. This is the one-way radio delay measured
      by srsRAN between the gNB and its UEs. Range: ~64–228 ms in 5G3E.

    jitter_ms:
      Standard deviation of delay_ms over the last JITTER_WIN snapshots.
      This captures VARIATION in delay, which is what jitter means in QoS.
      A gNB with stable 100ms delay has low jitter; one that fluctuates has high.

    packet_loss:
      dl_nof_nok = downlink HARQ retransmissions (failed blocks).
      dl_nof_ok  = successful blocks.
      PL = nok / (nok + ok) per UE in ue_list, then averaged across all UEs.
      Formula used by 3GPP for BLER. If no UEs: PL = 0.
    """
    # Rolling delay buffer for jitter: {gnb_id: deque of last W delays}
    from collections import deque
    delay_buffer = {gid: deque(maxlen=JITTER_WIN) for gid in gnb_id_list}

    labels = {gid: np.zeros((n_snap, 4), dtype=np.float32) for gid in gnb_id_list}

    for t in range(n_snap):
        for gid in gnb_id_list:
            rec = gnb_records[gid][t]

            # ── delay_ms ────────────────────────────────────────────────────
            delay = float(rec.get('cell_metrics', {}).get('average_latency', 0) or 0)

            # ── jitter_ms (rolling std) ─────────────────────────────────────
            delay_buffer[gid].append(delay)
            if len(delay_buffer[gid]) >= 2:
                jitter = float(np.std(list(delay_buffer[gid])))
            else:
                jitter = 0.0

            # ── packet_loss (from gNB ue_list DL block counters) ────────────
            ue_list = rec.get('ue_list', [])
            if ue_list:
                pl_vals = []
                for entry in ue_list:
                    uc = entry.get('ue_container', {})
                    nok = float(uc.get('dl_nof_nok', 0) or 0)
                    ok  = float(uc.get('dl_nof_ok',  0) or 0)
                    total = nok + ok
                    pl_vals.append(nok / total if total > 0 else 0.0)
                packet_loss = float(np.mean(pl_vals))
            else:
                packet_loss = 0.0

            # ── energy_W (3GPP formula with server cpu_util) ────────────────
            sidx   = GNB_TO_SRV[gid]
            cpu_t  = float(srv_cpu[sidx][t]) if sidx in srv_cpu else 0.5
            energy = P_IDLE + P_DELTA * cpu_t

            labels[gid][t] = [energy, delay, jitter, packet_loss]

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ─ SNAPSHOT BUILDER
# For each timestamp t: construct one HeteroData graph object.
# ══════════════════════════════════════════════════════════════════════════════

def build_snapshot(t, gnb_id_list, ue_id_list, gnb_records,
                   ue_data, srv_cpu, labels, topo):
    """
    Build one HeteroData snapshot for timestamp t.

    Node features:
      ue.x  [N_UE, 7]  : [rsrp, dl_snr, dl_brate, ul_brate, dl_mcs, ul_mcs, ul_buff]
      gnb.x [N_GNB, 5] : [nof_ue, site_0, site_1, site_2, mean_rsrp]
      srv.x [N_SRV, 1] : [cpu_util]

    WHY THESE UE FEATURES:
      rsrp      — signal strength from serving gNB (dBm). Core link-quality indicator.
      dl_snr    — downlink signal-to-noise ratio (dB). Higher → better encoding possible.
      dl_brate  — downlink data rate (bps). Current throughput demand.
      ul_brate  — uplink data rate (bps). Uplink traffic load.
      dl_mcs    — downlink modulation & coding scheme (0–28). Maps SNR→throughput.
      ul_mcs    — uplink MCS.
      ul_buff   — uplink buffer occupancy. Indicates pending UL backlog.

    WHY THESE gNB FEATURES:
      nof_ue         — number of active UEs. Captures current load.
      site_0/1/2     — one-hot site encoding. Distinguishes interference pattern.
      mean_rsrp      — average UE signal quality. High → UEs well-covered.
      *** NO average_latency, NO load_norm *** (both were leaky in old code)

    WHY THIS SERVER FEATURE:
      cpu_util — fraction of server CPU used (0..1). Directly determines energy.
                 Energy label = 130 + 470 × cpu_util.
                 Information must travel along the (srv→gnb) backhaul edge.

    Node label:
      gnb.y [N_GNB, 4] : [energy_W, delay_ms, jitter_ms, packet_loss]
      Predictions are PER GNB (13 outputs), not per UE (old code: 65 outputs).
      This matches the architecture: CMOA acts on gNBs, not individual UEs.
    """
    gnb_idx   = topo['gnb_idx']
    ue_idx    = topo['ue_idx']
    ue_to_gnb = topo['ue_to_gnb']

    N_UE_  = len(ue_id_list)
    N_GNB_ = len(gnb_id_list)

    # ── UE node features ────────────────────────────────────────────────────
    ue_feat = np.zeros((N_UE_, 7), dtype=np.float32)

    for uid in ue_id_list:
        ni  = ue_idx[uid]
        row = ue_data[uid].iloc[t]

        rsrp     = float(row.get('rsrp',     0) or 0)
        dl_snr   = float(row.get('dl_snr',   0) or 0)
        dl_brate = float(row.get('dl_brate', 0) or 0)
        ul_brate = float(row.get('ul_brate', 0) or 0)
        dl_mcs   = float(row.get('dl_mcs',   0) or 0)
        ul_mcs   = float(row.get('ul_mcs',   0) or 0)
        ul_buff  = float(row.get('ul_buff',  0) or 0)

        ue_feat[ni] = [rsrp, dl_snr, dl_brate, ul_brate, dl_mcs, ul_mcs, ul_buff]

    # ── gNB node features ───────────────────────────────────────────────────
    gnb_feat = np.zeros((N_GNB_, 5), dtype=np.float32)

    # Pre-aggregate mean rsrp per gNB from UE features we just built
    gnb_rsrp_sum   = np.zeros(N_GNB_, dtype=np.float32)
    gnb_rsrp_count = np.zeros(N_GNB_, dtype=np.float32)
    for uid in ue_id_list:
        gi = gnb_idx[ue_to_gnb[uid]]
        ui = ue_idx[uid]
        gnb_rsrp_sum[gi]   += ue_feat[ui, 0]  # rsrp is feature index 0
        gnb_rsrp_count[gi] += 1

    for gid in gnb_id_list:
        ni  = gnb_idx[gid]
        rec = gnb_records[gid][t]

        nof_ue   = float(len(rec.get('ue_list', [])))
        site_oh  = [0.0, 0.0, 0.0]
        site_oh[SITE_IDX[GNB_SITE_NAME[gid]]] = 1.0
        mean_rsrp = (gnb_rsrp_sum[ni] / gnb_rsrp_count[ni]
                     if gnb_rsrp_count[ni] > 0 else 0.0)

        gnb_feat[ni] = [nof_ue, site_oh[0], site_oh[1], site_oh[2], mean_rsrp]

    # ── Server node features ────────────────────────────────────────────────
    srv_feat = np.zeros((N_SRV, 1), dtype=np.float32)
    for sidx in range(N_SRV):
        srv_feat[sidx, 0] = float(srv_cpu[sidx][t]) if sidx in srv_cpu else 0.0

    # ── gNB labels ──────────────────────────────────────────────────────────
    gnb_labels = np.zeros((N_GNB_, 4), dtype=np.float32)
    for gid in gnb_id_list:
        gnb_labels[gnb_idx[gid]] = labels[gid][t]

    # ── Assemble HeteroData ─────────────────────────────────────────────────
    d = HeteroData()

    # Node features
    d['ue'].x  = torch.tensor(ue_feat,    dtype=torch.float32)
    d['gnb'].x = torch.tensor(gnb_feat,   dtype=torch.float32)
    d['srv'].x = torch.tensor(srv_feat,   dtype=torch.float32)

    # Label (only on gNB — that is what we predict)
    d['gnb'].y = torch.tensor(gnb_labels, dtype=torch.float32)

    # Edges (all static — same topology for every snapshot)
    d['ue',  'connects',    'gnb'].edge_index = topo['e_serve']
    d['gnb', 'rev_connects', 'ue'].edge_index = topo['e_rev_serve']
    d['gnb', 'interferes',  'gnb'].edge_index = topo['e_interfere']
    d['gnb', 'handover',    'gnb'].edge_index = topo['e_handover']
    d['gnb', 'backhaul',    'srv'].edge_index = topo['e_backhaul']
    d['srv', 'rev_backhaul','gnb'].edge_index = topo['e_rev_backhaul']

    return d


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 ─ MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_day(day_root: Path, split: str, snap_offset: int,
                split_assignment: dict, out_dir: Path):
    """
    Process one day of raw data → save snapshot .pt files.

    Args:
      day_root        Path to the raw day folder (e.g. .../raw/day_1)
      split           'train', 'val', or 'test'
      snap_offset     global snapshot counter before this day
      split_assignment dict to collect snapshot IDs per split
      out_dir         where to write snapshot_XXXXX.pt files

    Returns: new snap_offset after this day
    """
    SEP = '-' * 60
    print(f'\n{SEP}')
    print(f'  Day: {day_root.name}  ->  split = {split.upper()}')
    print(f'{SEP}')

    # ── 1. Discover files ────────────────────────────────────────────────────
    gnb_files, ue_files, srv_files, site_of_ue, site_of_gnb = discover_files(day_root)

    gnb_id_list = sorted(gnb_files)
    ue_id_list  = sorted(ue_files)

    # ── 2. Build static topology (once per day; same structure every snapshot) ──
    topo = build_static_edges(gnb_id_list, ue_id_list, site_of_ue, site_of_gnb)

    # ── 3. Load gNB NDJSON records (sampled) ─────────────────────────────────
    print('  Loading gNB NDJSON records...')
    gnb_records, n_snap = load_gnb_records(gnb_files, SAMPLE_EVERY)

    # ── 4. Load UE CSVs (index-aligned to gNB record count) ─────────────────
    print('  Loading UE CSVs...')
    sample_ue_path = list(ue_files.values())[0]
    ue_len = sum(1 for _ in open(sample_ue_path)) - 1   # minus header
    row_indices = np.linspace(0, ue_len - 1, n_snap).astype(int).tolist()
    print(f'  UE rows={ue_len} → mapped to {n_snap} snapshot indices')
    ue_data = load_ue_data(ue_files, row_indices)

    # ── 5. Load server cpu_util (chunked, index-downsampled) ─────────────────
    print('  Loading server CPU data...')
    srv_cpu = load_server_cpu(srv_files, n_snap)

    # ── 6. Compute all gNB labels in one pass ─────────────────────────────────
    print('  Computing gNB labels (energy, delay, jitter, PL)...')
    labels = extract_gnb_labels(gnb_records, gnb_id_list, srv_cpu, n_snap)

    # ── 7. Build and save snapshots ───────────────────────────────────────────
    print(f'  Building {n_snap} snapshots...')
    day_snap_ids = []
    for t in range(n_snap):
        snap = build_snapshot(t, gnb_id_list, ue_id_list, gnb_records,
                              ue_data, srv_cpu, labels, topo)
        snap_id = snap_offset + t
        torch.save(snap, out_dir / f'snapshot_{snap_id:05d}.pt')
        day_snap_ids.append(snap_id)

        if t % 500 == 0 or t == n_snap - 1:
            gnb_lbl = snap['gnb'].y.numpy()
            print(f'  t={t:4d}/{n_snap-1}  '
                  f'energy={gnb_lbl[:,0].mean():.1f}W  '
                  f'delay={gnb_lbl[:,1].mean():.1f}ms  '
                  f'jitter={gnb_lbl[:,2].mean():.2f}ms  '
                  f'PL={gnb_lbl[:,3].mean():.4f}')

    # ── 8. Assign to split ────────────────────────────────────────────────────
    if split == 'train':
        n_val = int(VAL_FRAC * len(day_snap_ids))
        split_assignment['train'] += day_snap_ids[:-n_val] if n_val > 0 else day_snap_ids
        split_assignment['val']   += day_snap_ids[-n_val:] if n_val > 0 else []
    else:
        split_assignment[split] += day_snap_ids

    print(f'  [OK] {n_snap} snapshots saved  (IDs {snap_offset}..{snap_offset+n_snap-1})')
    return snap_offset + n_snap


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 ─ NORMALIZATION STATS
# Compute mean/std from training snapshots only, then save for model use.
# ══════════════════════════════════════════════════════════════════════════════

def compute_norm_stats(train_ids: list, out_dir: Path):
    """
    Compute z-score normalization statistics from training snapshots.

    WHY Z-SCORE:
      Features (rsrp in dBm, dl_brate in bps, ul_buff in bytes) are on
      wildly different scales. Without normalization the gradient steps
      of the largest-scale feature dominate and the model ignores others.

    WHY TRAIN ONLY:
      If we included validation/test in the stats computation we would leak
      future distribution information into the model's scaling.

    Saved stats:
      ue_mean,  ue_std   — shape (7,)  for 7 UE input features
      gnb_mean, gnb_std  — shape (5,)  for 5 gNB input features
      srv_mean, srv_std  — shape (1,)  for 1 server input feature
      y_mean,   y_std    — shape (4,)  for 4 gNB label targets
    """
    print('\n  Computing normalization stats from training set...')

    all_ue, all_gnb, all_srv, all_y = [], [], [], []
    for sid in train_ids:
        d = torch.load(out_dir / f'snapshot_{sid:05d}.pt', weights_only=False)
        all_ue.append(d['ue'].x.numpy())
        all_gnb.append(d['gnb'].x.numpy())
        all_srv.append(d['srv'].x.numpy())
        all_y.append(d['gnb'].y.numpy())

    def stats(arr):
        m = arr.mean(0)
        s = arr.std(0)
        s = np.where(s > 0, s, 1.0)   # prevent division by zero for constant cols
        return m.astype(np.float32), s.astype(np.float32)

    ue_arr  = np.concatenate(all_ue,  axis=0)  # [N_train×N_UE,  7]
    gnb_arr = np.concatenate(all_gnb, axis=0)  # [N_train×N_GNB, 5]
    srv_arr = np.concatenate(all_srv, axis=0)  # [N_train×N_SRV, 1]
    y_arr   = np.concatenate(all_y,   axis=0)  # [N_train×N_GNB, 4]

    ue_mean,  ue_std  = stats(ue_arr)
    gnb_mean, gnb_std = stats(gnb_arr)
    srv_mean, srv_std = stats(srv_arr)
    y_mean,   y_std   = stats(y_arr)

    ue_cols  = ['rsrp', 'dl_snr', 'dl_brate', 'ul_brate', 'dl_mcs', 'ul_mcs', 'ul_buff']
    gnb_cols = ['nof_ue', 'site_0', 'site_1', 'site_2', 'mean_rsrp']
    srv_cols = ['cpu_util']
    y_cols   = ['energy_W', 'delay_ms', 'jitter_ms', 'packet_loss']

    print('  UE features:')
    for i, c in enumerate(ue_cols):
        print(f'    {c:<12} mean={ue_mean[i]:9.3f}  std={ue_std[i]:9.3f}')
    print('  gNB features:')
    for i, c in enumerate(gnb_cols):
        print(f'    {c:<12} mean={gnb_mean[i]:9.3f}  std={gnb_std[i]:9.3f}')
    print('  Server features:')
    print(f'    cpu_util     mean={srv_mean[0]:9.3f}  std={srv_std[0]:9.3f}')
    print('  Labels:')
    for i, c in enumerate(y_cols):
        print(f'    {c:<14} mean={y_mean[i]:9.3f}  std={y_std[i]:9.3f}')

    stats_dict = {
        'ue_mean': ue_mean, 'ue_std': ue_std,
        'gnb_mean': gnb_mean, 'gnb_std': gnb_std,
        'srv_mean': srv_mean, 'srv_std': srv_std,
        'y_mean': y_mean, 'y_std': y_std,
        'ue_cols': ue_cols, 'gnb_cols': gnb_cols,
        'srv_cols': srv_cols, 'y_cols': y_cols,
    }
    torch.save(stats_dict, out_dir / 'norm_stats.pt')
    print(f'  Saved → {out_dir / "norm_stats.pt"}')
    return stats_dict


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WirelessDT-6G v3 — C1 Graph Construction')
    parser.add_argument('--days', nargs='+', type=int, default=[1],
                        help='Day numbers to process as TRAIN. Last 15%% → val.')
    parser.add_argument('--test_days', nargs='+', type=int, default=[],
                        help='Day numbers to use as TEST set (held-out).')
    parser.add_argument('--sample_every', type=int, default=SAMPLE_EVERY)
    args = parser.parse_args()

    SAMPLE_EVERY = args.sample_every
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('═' * 60)
    print('  WirelessDT-6G v3 — C1: Graph Construction')
    print('═' * 60)
    print(f'  Train days : {args.days}')
    print(f'  Test  days : {args.test_days}')
    print(f'  sample_every = {SAMPLE_EVERY}  (1 of every {SAMPLE_EVERY} gNB records)')
    print(f'  Output dir : {OUT_DIR}')

    split_assignment = {'train': [], 'val': [], 'test': []}
    snap_counter = 0

    # Process train days
    for day_num in args.days:
        day_root = BASE_RAW / f'day_{day_num}'
        snap_counter = process_day(day_root, 'train', snap_counter,
                                   split_assignment, OUT_DIR)

    # Process test days
    for day_num in args.test_days:
        day_root = BASE_RAW / f'day_{day_num}'
        snap_counter = process_day(day_root, 'test', snap_counter,
                                   split_assignment, OUT_DIR)

    # Save split index
    torch.save(split_assignment, OUT_DIR / 'split.pt')

    print('\n' + '═' * 60)
    print('  SPLIT SUMMARY')
    print('═' * 60)
    for k, v in split_assignment.items():
        print(f'  {k:<6} : {len(v)} snapshots')

    # Compute normalization stats from training set
    compute_norm_stats(split_assignment['train'], OUT_DIR)

    total = snap_counter
    print('\n' + '═' * 60)
    print(f'  DONE  |  {total} snapshots total  →  {OUT_DIR}')
    print('═' * 60)
