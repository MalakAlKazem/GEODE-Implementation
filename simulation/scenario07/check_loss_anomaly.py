"""
check_loss_anomaly.py

At p37, sleeping gnb0 REDUCED mean packet loss (0.72% -> 0.31%). That
should not happen when a cell's traffic is pushed onto neighbours.

Hypothesis: loss is computed as 1 - received/sent from .sca scalar
counts. If UEs that lost their serving cell also SENT fewer packets --
because they spent time unattached, or because the app's send queue
stalled -- the ratio can improve while service actually got worse. The
metric would then be unsafe to use as a QoS signal under sleep, since
"less loss" would sometimes mean "less traffic got out at all".

This checks the hypothesis directly by comparing total sent counts
between each pair. If sent drops materially in the sleep runs, the
loss metric is confounded and CMOA must not use it as-is.

    SEED=22 python3 check_loss_anomaly.py
"""

import os
import sqlite3
import subprocess
import tempfile

import pandas as pd

SEED = int(os.environ.get("SEED", "22"))
PAIRS = [(37, 0), (37, 1), (37, 3), (43, 0), (43, 1), (46, 1)]
NUM_UES = 30
N_PHASES = 40


def sca_counts(cfg):
    """Total packetSent and packetReceived across all app instances."""
    sca = f"results/{cfg}/0.sca"
    if not os.path.exists(sca):
        return None
    tmp = tempfile.mktemp(suffix=".sqlite")
    try:
        subprocess.run(["opp_scavetool", "export", "-T", "s",
                        "-F", "SqliteScalarFile", "-o", tmp, sca],
                       capture_output=True)
        conn = sqlite3.connect(tmp)
        df = pd.read_sql(
            "SELECT s.scalarName, s.moduleName, s.scalarValue "
            "FROM scalar s", conn)
        conn.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    sent = df[(df.scalarName == "packetSent:count")
              & df.moduleName.str.contains(".ue[", regex=False)]
    recv = df[(df.scalarName == "packetReceived:count")
              & df.moduleName.str.contains("server", regex=False)]
    return sent.scalarValue.sum(), recv.scalarValue.sum()


print(f"seed {SEED}: is the loss drop real, or just less traffic sent?\n")
print(f"{'pair':<18} {'sent':>12} {'recv':>12} {'loss%':>7}   {'vs base':>22}")
print("-" * 78)

for power, k in PAIRS:
    base = sca_counts(f"Scenario07s{SEED}_p{power}")
    slp = sca_counts(f"Scenario07s{SEED}_p{power}_sleep{k}")
    if base is None or slp is None:
        print(f"p{power}_sleep{k}: missing .sca, skipped")
        continue

    bs, br = base
    ss, sr = slp
    bl = (1 - br / bs) * 100 if bs else float("nan")
    sl = (1 - sr / ss) * 100 if ss else float("nan")
    sent_change = (ss - bs) / bs * 100
    recv_change = (sr - br) / br * 100

    print(f"p{power} baseline    {bs:>12,.0f} {br:>12,.0f} {bl:>7.2f}")
    print(f"p{power} sleep{k}      {ss:>12,.0f} {sr:>12,.0f} {sl:>7.2f}   "
          f"sent {sent_change:+.1f}%  recv {recv_change:+.1f}%")

    if sent_change < -2:
        print(f"    ^ CONFOUNDED: {abs(sent_change):.1f}% fewer packets SENT. "
              f"A lower loss ratio here does not mean better service.")
    elif sl < bl:
        print(f"    ^ loss genuinely lower with similar traffic volume "
              f"-- needs another explanation")
    print()

print("If sent counts drop in the sleep runs, packet_loss as currently")
print("defined (1 - received/sent) is not a valid QoS signal for CMOA:")
print("it can improve precisely when service degrades. Delivered-packet")
print("count, or loss measured against OFFERED load, would be the fix.")
