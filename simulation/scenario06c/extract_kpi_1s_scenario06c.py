"""
extract_kpi_1s_scenario06c.py -- delay and jitter at 1 s resolution.

    SEED=61 python3 extract_kpi_1s_scenario06c.py

Writes kpi_targets_1s.csv alongside the existing kpi_targets.csv rather
than replacing it, so the two label resolutions can be compared directly
and nothing already validated is lost.

Why
---
Delay is currently measured once per 3 s traffic phase, because each UE
runs one UdpBasicApp per phase and the extractor averages that app's
whole packet stream. Network state is sampled every 100 ms, so one label
covers 30 snapshots. Training on one snapshot per phase to avoid the
duplication leaves only ~544 training graphs, which is the binding
constraint on delay accuracy.

The per-packet delays carry timestamps, so the same stream can be cut
into 1 s windows instead: 3 labels per phase rather than 1, 7200 per run
instead of 2400, and a stride of 10 instead of 30 -- roughly 1632
training graphs, a threefold increase with no duplication. The label also
describes the same second the input snapshot sits in, rather than a 3 s
average the snapshot only partly belongs to.

What this does NOT fix: packet loss
-----------------------------------
Packet loss is computed from packetSent:count and packetReceived:count,
which are SCALARS recorded once per app -- that is, once per (UE, phase).
There is no per-second breakdown anywhere in the .sca file, so loss
cannot be rebinned. It stays at 3 s resolution and is repeated across the
three 1 s windows of its phase.

The result is mixed resolution: delay and jitter at 1 s, packet loss at
3 s. That is acceptable here because the loss head is a separate
two-stage classifier and is the one target where the graph does not help,
but it must be stated rather than left to be discovered.

Sample-count caveat
-------------------
A Low-traffic phase sends roughly one packet per 150 ms, so a 1 s window
holds about 7 packets. Mean delay over 7 samples is reasonable; a
standard deviation over 7 samples is noisy. Every row therefore carries
n_packets, so thin windows can be filtered or down-weighted rather than
trusted blindly. Windows with a single packet get jitter 0.0, which is
the same convention the 3 s extractor uses.
"""

import os
import re
import sqlite3
import subprocess
import tempfile

import numpy as np
import pandas as pd

SEED = int(os.environ.get("SEED", "61"))
BIN_S = float(os.environ.get("BIN_S", "1.0"))     # window length in seconds

N_UE = 60
N_PHASES = 40
PHASE_LEN_S = 3.0
WINDOW_S = 0.1                # snapshot spacing
TRIM_OFFSET_S = 0.0           # scenario06c needs no settling trim
RESULTS_DIR = "results"
EXTRACTED_DIR = "results"

BINS_PER_PHASE = int(round(PHASE_LEN_S / BIN_S))
N_WINDOWS = N_PHASES * BINS_PER_PHASE

REPS = {
    (61, "p37"): 0, (61, "p43"): 0, (61, "p46"): 0,
    (61, "p43_sleep0"): 0, (61, "p43_sleep1"): 0, (61, "p43_sleep2"): 0,
    (61, "p43_sleep3"): 0, (61, "p43_sleep4"): 0, (61, "p43_sleep5"): 0,
    (62, "p37"): 0, (62, "p43"): 0, (62, "p46"): 0,
    (62, "p43_sleep0"): 0, (62, "p43_sleep1"): 0, (62, "p43_sleep2"): 0,
    (62, "p43_sleep5"): 0,
    (63, "p37"): 0, (63, "p43"): 0, (63, "p46"): 0,
    (63, "p43_sleep0"): 0, (63, "p43_sleep1"): 0, (63, "p43_sleep2"): 0,
    (63, "p43_sleep3"): 0, (63, "p43_sleep4"): 0, (63, "p43_sleep5"): 0,
}
RUNS = [r for (s, r) in REPS if s == SEED]


def run_opp_scavetool(args):
    r = subprocess.run(["opp_scavetool"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"opp_scavetool failed: {' '.join(args)}\n{r.stderr}")
    return r.stdout


def _simtime_scale(conn):
    """Seconds per raw simtime unit.

    OMNeT++ stores vector timestamps as integers plus an exponent. The
    exponent lives in the run table, but the column name has moved
    between versions, so this probes rather than assumes, and reports
    what it found. Guessing wrong here would silently shift every packet
    into the wrong window.
    """
    for q in ("SELECT simtimeExp FROM run LIMIT 1",
              "SELECT attrValue FROM runAttr WHERE attrName='simtimeExp' LIMIT 1"):
        try:
            v = conn.execute(q).fetchone()
            if v is not None and v[0] is not None:
                exp = int(v[0])
                print(f"    simtimeExp = {exp} (from: {q.split(' FROM ')[1].split()[0]})")
                return 10.0 ** exp
        except sqlite3.Error:
            continue
    print("    WARNING: simtimeExp not found, assuming -12 (picoseconds). "
          "Verify the reported time range looks like 0-120 s.")
    return 1e-12


def extract_delay_jitter_binned(vec_file):
    """One row per (ue_index, window_index) with mean delay, jitter and
    the packet count the estimate rests on."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        sqlite_path = tmp.name
    try:
        run_opp_scavetool([
            "export", "-F", "SqliteVectorFile", "-o", sqlite_path,
            "-f", "name =~ endToEndDelay:vector", vec_file,
        ])
        conn = sqlite3.connect(sqlite_path)
        scale = _simtime_scale(conn)
        df = pd.read_sql(
            """
            SELECT vector.moduleName AS moduleName,
                   vectorData.simtimeRaw AS traw,
                   vectorData.value      AS value
            FROM vectorData
            JOIN vector ON vectorData.vectorId = vector.vectorId
            """, conn)
        conn.close()
    finally:
        os.remove(sqlite_path)

    # Drop the SCTP X2/S1 association vectors, which share the stat name
    # but are control-plane signalling rather than UE traffic.
    df = df[df.moduleName.str.contains(r"server\.app\[\d+\]", regex=True)].copy()
    if df.empty:
        raise RuntimeError("no server.app endToEndDelay vectors found")

    df["N"] = df.moduleName.str.extract(r"server\.app\[(\d+)\]").astype(int)
    df["t"] = df.traw * scale
    print(f"    packet timestamps span {df.t.min():.2f}-{df.t.max():.2f} s "
          f"({len(df)} packets)")
    if df.t.max() > 200 or df.t.max() < 10:
        print("    WARNING: time range implausible for a 120 s run -- "
              "the simtime scale is probably wrong. Do not trust this output.")

    df["ue_index"] = df.N // N_PHASES
    df["phase_index"] = df.N % N_PHASES

    # Window index from absolute time. Using absolute time rather than
    # (phase start + offset) means a packet delivered slightly after its
    # phase boundary lands in the window it actually arrived in.
    df["window_index"] = np.floor(df.t / BIN_S).astype(int)
    df = df[(df.window_index >= 0) & (df.window_index < N_WINDOWS)]

    agg = df.groupby(["ue_index", "window_index"])["value"].agg(
        delay_ms=lambda v: v.mean() * 1000.0,
        jitter_ms=lambda v: v.std(ddof=0) * 1000.0 if len(v) > 1 else 0.0,
        n_packets="size",
    ).reset_index()

    thin = int((agg.n_packets < 3).sum())
    if thin:
        print(f"    {thin} of {len(agg)} windows have fewer than 3 packets "
              f"({thin/len(agg)*100:.1f}%) -- jitter unreliable there, "
              f"see n_packets column")
    return agg


def extract_scalar_counts(sca_file, stat_name, module_regex):
    out = run_opp_scavetool(["query", "-l", "-f", f"name =~ {stat_name}", sca_file])
    result, pattern = {}, re.compile(module_regex)
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


def phase_level_loss(sca_file):
    """{(ue, phase): loss}. Stays at phase resolution -- see the module
    docstring; the underlying counters are per-app scalars."""
    sent = extract_scalar_counts(sca_file, "packetSent:count",
                                 r"ue\[(\d+)\]\.app\[(\d+)\]")
    recv = extract_scalar_counts(sca_file, "packetReceived:count",
                                 r"server\.app\[(\d+)\]")
    loss = {}
    for (ue, ph), s in sent.items():
        r = recv.get(ue * N_PHASES + ph)
        if r is None or s == 0:
            continue
        loss[(ue, ph)] = 1.0 - (r / s)
    return loss


def build_window_kpi(run):
    rep = REPS[(SEED, run)]
    vec = f"{RESULTS_DIR}/Scenario06Cs{SEED}_{run}/{rep}.vec"
    sca = f"{RESULTS_DIR}/Scenario06Cs{SEED}_{run}/{rep}.sca"

    print(f"[{run}] delay/jitter in {BIN_S:g} s windows from {vec}")
    agg = extract_delay_jitter_binned(vec)

    print(f"[{run}] packet loss (phase resolution) from {sca}")
    loss = phase_level_loss(sca)

    agg["phase_index"] = (agg.window_index * BIN_S // PHASE_LEN_S).astype(int)
    agg["packet_loss"] = [loss.get((int(u), int(p)), np.nan)
                          for u, p in zip(agg.ue_index, agg.phase_index)]
    miss = int(agg.packet_loss.isna().sum())
    if miss:
        print(f"    {miss} windows have no packet-loss count "
              f"(their phase sent nothing) -- left as NaN for the repair step")
    return agg


def broadcast(window_kpi, snapshot_ids):
    lookup = {(int(r.ue_index), int(r.window_index)): r
              for r in window_kpi.itertuples()}
    rows = []
    for sid in snapshot_ids:
        t = TRIM_OFFSET_S + sid * WINDOW_S
        w = int(t // BIN_S)
        for ue in range(N_UE):
            r = lookup.get((ue, w))
            if r is None:
                continue
            rows.append({"snapshot_id": sid, "ue_index": ue,
                         "kpi_window_id": w,
                         "delay_ms": r.delay_ms, "jitter_ms": r.jitter_ms,
                         "packet_loss": r.packet_loss,
                         "n_packets": r.n_packets})
    return pd.DataFrame(rows)


def main():
    print(f"seed {SEED}, {BIN_S:g} s windows "
          f"({BINS_PER_PHASE} per phase, {N_WINDOWS} per run)\n")
    for run in RUNS:
        print(f"{'='*60}\nRun: {run}\n{'='*60}")
        d = f"{EXTRACTED_DIR}/extracted_s06c{SEED}_{run}"
        gnb = pd.read_csv(f"{d}/gnb_inputs.csv")
        snaps = sorted(gnb.snapshot_id.unique())

        wk = build_window_kpi(run)
        print(f"[{run}] window-level table: {len(wk)} rows "
              f"(expect up to {N_UE*N_WINDOWS})")

        out = broadcast(wk, snaps)
        expect = len(snaps) * N_UE
        print(f"[{run}] broadcast: {len(out)} rows (expect {expect})")
        if len(out) != expect:
            print(f"  WARNING: {expect-len(out)} rows missing -- windows with "
                  f"no packets at all. repair_kpi_grid will fill and flag them.")

        path = f"{d}/kpi_targets_1s.csv"
        out.to_csv(path, index=False)
        print(f"[{run}] wrote {path}")
        print(f"    delay mean {out.delay_ms.mean():.1f}ms  "
              f"median {out.delay_ms.median():.1f}ms  max {out.delay_ms.max():.1f}ms")
        print(f"    unique (ue,window) labels: "
              f"{out.groupby(['ue_index','kpi_window_id']).ngroups} "
              f"(3 s version had {N_UE*N_PHASES})\n")


if __name__ == "__main__":
    main()
