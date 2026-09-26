"""
extend_trim_scenario07.py

Second-stage trim for scenario07, extending the original handover-
corruption fix (TRIM_BEFORE_SNAPSHOT=190, based on aggregate gNB
connected-UE count stabilizing by t=19s).

Why a SECOND trim is needed, on top of the first
------------------------------------------------------
This session found (via kpi_targets.csv, generated after the first
trim already happened) that mean delay/jitter/packet_loss are still
elevated for roughly the first 6 phases of the ALREADY-TRIMMED data
(local snapshot 0-179, i.e. absolute t=19-37s):
    phases 6-11  (first 6 after original trim): mean delay 35.9ms
    phases 12-25:                                mean delay 25.9ms
    phases 26-39:                                mean delay 22.8ms
Per-UE HANDOVER EVENT COUNT (from serving_edges.csv) shows the same
sharp concentration: ~7 handovers/phase in phases 6-11, dropping to
~1.4-1.8/phase afterward -- consistent with individual UEs still
actively correcting the original mis-assigned initial handover
(confirmed at the very start of this pipeline's development: UE0
started correctly assigned to gNB2, but enableHandover immediately
switched it to gNB0 within 50ms, before real channel measurements
stabilized) for a while AFTER the aggregate gNB connected-UE count
already looked balanced. The count stabilizing quickly doesn't mean
individual UEs stopped churning -- one UE leaving a cell can be
offset by another arriving, keeping the count flat while individual
handover events keep happening.

Confirmed across ALL 3 tx-power runs this session (not just one):
    p37: phases 6-11 mean delay 32.7ms vs phases 26-39 mean 22.6ms
    p43: phases 6-11 mean delay 36.6ms vs phases 26-39 mean 24.2ms
    p46: phases 6-11 mean delay 35.9ms vs phases 26-39 mean 22.8ms
Consistent ~45-57% elevation in all three, as expected since mobility
params (positions/speeds/headings) are defined once and shared across
all 3 power configs -- confirmed directly against omnetpp.ini
(mobility.speed appears exactly 30 times total, once per UE, not once
per power level).

Ruled OUT as an explanation: a designed L/M/H traffic ramp. Checked
directly against omnetpp.ini's sendInterval per phase: phases 6-25
are actually LESS busy by traffic-rate (55% H+M) than phases 26-39
(85.7% H+M) -- the opposite of what a traffic-driven ramp would
predict. The elevated delay/loss tracks handover churn, not traffic
load.

What this script does
--------------------------
Drops an ADDITIONAL 180 snapshots (6 phases x 30 snapshots) from the
front of all 6 files (5 original + kpi_targets.csv), for all 3 runs,
then re-indexes snapshot_id to start at 0 again -- same pattern as
the original TRIM_BEFORE_SNAPSHOT=190 script, just a second pass with
a new offset. Run this AFTER kpi_targets.csv already exists (i.e.
after extract_kpi_scenario07.py has already been run once) so the
same trim applies consistently to energy/gNB/UE data AND the KPI
labels together.
"""

import pandas as pd

SECOND_TRIM_SNAPSHOTS = 360  # 6 phases x 30 snapshots (phases 6-11 of the ALREADY-trimmed data)
RUNS = ["p37", "p43", "p46"]
FILES = ["gnb_inputs.csv", "ue_inputs.csv", "serving_edges.csv",
         "energy_targets.csv", "energy_diagnostics.csv", "kpi_targets.csv"]


def main():
    for run in RUNS:
        folder = f"results/extracted_s23_{run}"
        print(f"\n=== {run} ===")
        for fname in FILES:
            path = f"{folder}/{fname}"
            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                print(f"  {fname}: not found, skipping")
                continue

            before = len(df)
            df = df[df.snapshot_id >= SECOND_TRIM_SNAPSHOTS].copy()
            df["snapshot_id"] = df["snapshot_id"] - SECOND_TRIM_SNAPSHOTS
            df.to_csv(path, index=False)
            print(f"  {fname}: {before} -> {len(df)} rows")


if __name__ == "__main__":
    main()
