#!/bin/bash
# Runs one scenario08 config, retrying with successive repetition indices
# until one completes. The repetition index only changes OMNeT++'s RNG
# stream -- network, mobility and traffic schedule are identical -- so a
# run recovered at -r 1 is the same scenario, just a different random
# realization. Needed because the NrMacGnb handover race is RNG-dependent
# and fires in most repetitions at 8 gNB / 80 UE.
SEED=$1
POWER=$2
CFG="Scenario06Bs${SEED}_p${POWER}"
for r in 0 1 2; do
  echo "=== $CFG  -r $r ==="
  if simu5g -f omnetpp_s${SEED}.ini -c $CFG -u Cmdenv -r $r 2>&1 | tail -3 | grep -q "finish()"; then
    echo "OK: $CFG completed at repetition $r  -> results/$CFG/${r}.vec"
    echo "$CFG r=$r" >> s06b_successful_runs.txt
    exit 0
  fi
  echo "crashed at -r $r, retrying"
done
echo "FAILED: $CFG did not complete in 3 repetitions"
exit 1
