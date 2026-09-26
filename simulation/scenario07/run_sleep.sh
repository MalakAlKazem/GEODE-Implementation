#!/bin/bash
# run_sleep.sh -- runs the baseline + 4 sleep variants for one tx power.
#
#   ./run_sleep.sh 22 43
#
# Retries repetitions on crash, same as run_s08.sh: the NrMacGnb
# handover race is RNG-dependent. At 4 gNB it is rare (1 failure in 21
# runs), but sleeping a cell concentrates handovers onto the remaining
# three, so the rate may be higher here than in ordinary scenario07
# runs. Logged to sleep_successful_runs.txt.
SEED=$1
POWER=$2
[ -z "$POWER" ] && { echo "usage: ./run_sleep.sh SEED POWER"; exit 1; }

run_cfg () {
  CFG=$1
  for r in 0 1 2; do
    echo "=== $CFG  -r $r ==="
    if simu5g -f omnetpp_s${SEED}.ini -c $CFG -u Cmdenv -r $r 2>&1 | tail -3 | grep -q "finish()"; then
      echo "OK: $CFG at repetition $r"
      echo "$CFG r=$r" >> sleep_successful_runs.txt
      return 0
    fi
    echo "crashed at -r $r, retrying"
  done
  echo "FAILED: $CFG"
  return 1
}

run_cfg "Scenario07s${SEED}_p${POWER}"
for k in 0 1 2 3; do
  run_cfg "Scenario07s${SEED}_p${POWER}_sleep${k}"
done

echo
echo "done. Next: python3 verify_sleep.py  (confirms each sleeping cell"
echo "really lost its UEs -- do NOT extract before this passes)"
