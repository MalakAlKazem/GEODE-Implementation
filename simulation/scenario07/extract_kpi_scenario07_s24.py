"""
extract_kpi_scenario07.py

Extracts UL delay, jitter, and packet_loss per (ue_index, snapshot_id)
for scenario07's 3 runs (p37/p43/p46), producing kpi_targets.csv in
each run's extracted_s24_p{37,43,46} folder -- same schema as scenario05's
kpi_targets.csv (snapshot_id, ue_index, delay_ms, jitter_ms, packet_loss).

Everything below is derived and confirmed live against the actual
.ini/.vec/.sca files this session, not assumed:

- Traffic is generated as 40 separate UdpBasicApp instances per UE
  (one per 3s phase), each talking to a DEDICATED server.app[N] socket.
  Confirmed: 30 UEs x 40 phases = 1200 total server.app instances.
- N -> (ue_index, phase_index) mapping: confirmed via destPort at both
  range boundaries (ue[0].app[39]->4039, ue[1].app[0]->4040,
  ue[1].app[39]->4079), giving N = ue_index*40 + phase_index.
- phase_index -> absolute time window: confirmed via .ini startTime/
  stopTime (app[0]: [0,3)s, app[1]: [3,6)s), giving window
  [3*phase_index, 3*phase_index+3).
- Delay: server.app[N]'s endToEndDelay:vector (app-layer, UE->server
  end-to-end latency -- NOT rlcDelayUl, which was the original plan
  before we found this is a better, more direct signal).
- Jitter: std of the INDIVIDUAL packet delays inside that same vector
  (not a separate extraction -- same vector, just std instead of mean).
  This is the finest granularity actually available: a per-100ms
  rolling std (scenario05's original approach) isn't possible here,
  since delay is only recorded once per (ue, phase), not per snapshot.
- Packet loss: NOT rlcPacketLossUl -- confirmed empty (count=0/mean=nan
  on every bearer type, TM/UM/AM alike; the vector name exists but
  nothing is ever recorded on it in this run). Computed instead as
  1 - received/sent using two independently-confirmed real counts:
    sent:     ue[ue_index].app[phase_index]'s packetSent:count (scalar)
    received: server.app[N]'s packetReceived:count (scalar)
  This is arguably a more honest metric anyway -- true end-to-end loss
  at the same measurement point as delay, not an RLC-internal stat.
- snapshot_id -> absolute time: confirmed the CURRENT extracted CSVs
  were re-indexed after trimming (TRIM_BEFORE_SNAPSHOT=190, raw
  snapshot_id -= 190), so:
      absolute_time = 19.0 + snapshot_id * 0.1
  (19.0s = 190 * 0.1s, the exact trim cutoff, confirmed from the trim
  script itself, not re-derived/assumed.)

Design choice: broadcast, not interpolate
------------------------------------------
Each (ue, phase) has exactly ONE delay/jitter/packet_loss value (the
phase is the finest granularity the traffic generator supports). Every
snapshot_id whose absolute_time falls inside that phase's [3k,3k+3)
window gets that same value. This is the same "broadcast a coarser
label across finer snapshots" pattern already implicit in how
gnb_inputs.csv/energy_targets.csv relate to the 40-phase L/M/H design
-- we are not inventing new smoothing/interpolation behavior here.
"""

import re
import subprocess
import sqlite3
import tempfile
import os
import pandas as pd
import numpy as np

RUNS = ["p37", "p43", "p46"]
RESULTS_DIR = "results"
EXTRACTED_DIR = "results"  # extracted_s24_p{run} lives alongside raw results/
TRIM_OFFSET_S = 0.0       # 190 * 0.1s, confirmed exact trim cutoff
WINDOW_S = 0.1
PHASE_LEN_S = 3.0
N_PHASES = 40
N_UE = 30


def run_opp_scavetool(args: list) -> str:
    """Run opp_scavetool and return stdout as text. Raises on failure
    so a silent bad run doesn't produce a silently-wrong CSV."""
    result = subprocess.run(["opp_scavetool"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"opp_scavetool failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout


def extract_delay_and_jitter(vec_file: str) -> pd.DataFrame:
    """
    Exports endToEndDelay:vector to a temporary SQLite file, reads the
    individual per-packet delay VALUES (not just the summary stats
    'query' gives us), and computes mean (delay_ms) + std (jitter_ms)
    per server.app[N].
    """
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        sqlite_path = tmp.name

    try:
        run_opp_scavetool([
            "export", "-F", "SqliteVectorFile", "-o", sqlite_path,
            "-f", "name =~ endToEndDelay:vector",
            vec_file,
        ])

        conn = sqlite3.connect(sqlite_path)
        df = pd.read_sql(
            """
            SELECT vector.moduleName AS moduleName, vectorData.value AS value
            FROM vectorData
            JOIN vector ON vectorData.vectorId = vector.vectorId
            """,
            conn,
        )
        conn.close()
    finally:
        os.remove(sqlite_path)

    # moduleName looks like "FourCell_Standalone.server.app[41]"
    # Filter out non-UE-app vectors that happen to share the same stat
    # name (confirmed live: 12 SCTP association vectors on the gNB
    # X2/S1 backhaul also report endToEndDelay:vector -- unrelated
    # control-plane signaling delay, not UE traffic. Drop before
    # N-extraction, or astype(int) crashes on the resulting NaNs.
    df = df[df["moduleName"].str.contains(r"server\.app\[\d+\]", regex=True)].copy()
    df["N"] = df["moduleName"].str.extract(r"server\.app\[(\d+)\]").astype(int)

    agg = df.groupby("N")["value"].agg(
        delay_ms=lambda v: v.mean() * 1000.0,
        jitter_ms=lambda v: v.std(ddof=0) * 1000.0 if len(v) > 1 else 0.0,
    ).reset_index()

    n_unique = agg["N"].nunique()
    if n_unique != N_UE * N_PHASES:
        print(f"  WARNING: expected {N_UE*N_PHASES} server.app entries with "
              f"delay data, found {n_unique} -- some (ue,phase) combos may "
              f"have sent zero packets (no delay vector recorded at all). "
              f"These will be dropped, not zero-filled -- check the final "
              f"row count against 30*40 afterward.")

    agg["ue_index"] = agg["N"] // N_PHASES
    agg["phase_index"] = agg["N"] % N_PHASES
    return agg[["ue_index", "phase_index", "delay_ms", "jitter_ms"]]


def extract_scalar_counts(sca_file: str, stat_name: str, module_regex: str) -> dict:
    """
    Runs opp_scavetool query filtered to ONE exact scalar name (avoids
    the ambiguous human-readable duplicate like 'packets sent' which
    has a space and would break naive whitespace parsing), parses the
    text output, and returns {key: value} where key is whatever
    module_regex's single capture group extracts (either N, for
    server.app[N], or (ue_index,phase_index) for ue[i].app[k]).
    """
    out = run_opp_scavetool([
        "query", "-l", "-f", f"name =~ {stat_name}", sca_file,
    ])

    result = {}
    pattern = re.compile(module_regex)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "scalar":
            continue
        module_name, stat, value = parts[1], parts[2], parts[3]
        if stat != stat_name:
            continue
        m = pattern.search(module_name)
        if not m:
            continue
        key = tuple(int(g) for g in m.groups()) if len(m.groups()) > 1 else int(m.group(1))
        result[key] = float(value)
    return result


def build_phase_level_kpi(run: str) -> pd.DataFrame:
    vec_file = f"{RESULTS_DIR}/Scenario07s24_{run}/0.vec"
    sca_file = f"{RESULTS_DIR}/Scenario07s24_{run}/0.sca"

    print(f"[{run}] Extracting delay/jitter from {vec_file} ...")
    delay_df = extract_delay_and_jitter(vec_file)

    print(f"[{run}] Extracting sent/received counts from {sca_file} ...")
    sent = extract_scalar_counts(
        sca_file, "packetSent:count", r"ue\[(\d+)\]\.app\[(\d+)\]"
    )
    received = extract_scalar_counts(
        sca_file, "packetReceived:count", r"server\.app\[(\d+)\]"
    )

    if len(sent) != N_UE * N_PHASES:
        print(f"  WARNING: expected {N_UE*N_PHASES} sent-count entries, "
              f"got {len(sent)}.")
    if len(received) != N_UE * N_PHASES:
        print(f"  WARNING: expected {N_UE*N_PHASES} received-count entries, "
              f"got {len(received)}.")

    rows = []
    for _, r in delay_df.iterrows():
        ue, ph = int(r.ue_index), int(r.phase_index)
        N = ue * N_PHASES + ph
        sent_count = sent.get((ue, ph))
        recv_count = received.get(N)

        if sent_count is None or recv_count is None:
            print(f"  WARNING: missing sent/received count for "
                  f"ue={ue} phase={ph} -- skipping this (ue,phase).")
            continue
        if sent_count == 0:
            print(f"  WARNING: sent_count=0 for ue={ue} phase={ph} "
                  f"-- packet_loss undefined, setting to NaN.")
            loss = np.nan
        else:
            loss = 1.0 - (recv_count / sent_count)

        rows.append({
            "ue_index": ue, "phase_index": ph,
            "delay_ms": r.delay_ms, "jitter_ms": r.jitter_ms,
            "packet_loss": loss,
        })

    return pd.DataFrame(rows)


def broadcast_to_snapshots(phase_kpi: pd.DataFrame, snapshot_ids: list) -> pd.DataFrame:
    """Expands one row per (ue,phase) into one row per (ue,snapshot_id),
    using absolute_time -> phase_index to pick the right value."""
    rows = []
    lookup = {(r.ue_index, r.phase_index): r for _, r in phase_kpi.iterrows()}

    for sid in snapshot_ids:
        abs_time = TRIM_OFFSET_S + sid * WINDOW_S
        phase_idx = int(abs_time // PHASE_LEN_S)
        for ue in range(N_UE):
            key = (ue, phase_idx)
            if key not in lookup:
                continue  # already warned about above, during extraction
            r = lookup[key]
            rows.append({
                "snapshot_id": sid, "ue_index": ue,
                "delay_ms": r.delay_ms, "jitter_ms": r.jitter_ms,
                "packet_loss": r.packet_loss,
            })

    return pd.DataFrame(rows)


def main():
    for run in RUNS:
        print(f"\n{'='*60}\nRun: {run}\n{'='*60}")

        gnb = pd.read_csv(f"{EXTRACTED_DIR}/extracted_s24_{run}/gnb_inputs.csv")
        snapshot_ids = sorted(gnb.snapshot_id.unique())
        print(f"[{run}] snapshot_id range in extracted CSVs: "
              f"{snapshot_ids[0]}-{snapshot_ids[-1]} ({len(snapshot_ids)} total)")

        phase_kpi = build_phase_level_kpi(run)
        print(f"[{run}] Built phase-level KPI table: {len(phase_kpi)} rows "
              f"(expect up to {N_UE*N_PHASES})")

        kpi_targets = broadcast_to_snapshots(phase_kpi, snapshot_ids)
        expected_rows = len(snapshot_ids) * N_UE
        print(f"[{run}] Broadcast to snapshot level: {len(kpi_targets)} rows "
              f"(expect {expected_rows} = {len(snapshot_ids)} snapshots x {N_UE} UEs)")
        if len(kpi_targets) != expected_rows:
            print(f"  WARNING: row count mismatch -- some (ue,phase) combos "
                  f"were missing data and got dropped. Check warnings above.")

        out_path = f"{EXTRACTED_DIR}/extracted_s24_{run}/kpi_targets.csv"
        kpi_targets.to_csv(out_path, index=False)
        print(f"[{run}] Wrote {out_path}")

        print(f"[{run}] Sanity check -- delay/jitter/loss summary:")
        print(kpi_targets[["delay_ms", "jitter_ms", "packet_loss"]].describe())


if __name__ == "__main__":
    main()
