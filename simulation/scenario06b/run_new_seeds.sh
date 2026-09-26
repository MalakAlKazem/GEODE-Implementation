#!/bin/bash
# One completed run per (seed, power) for the new topologies.
# Repetitions are tried only as crash recovery, not for extra data --
# they share the same topology, so a second completed repetition adds
# rows without adding topological information.
NED="${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/simulations:${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/emulation:${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/src:${SIMU5G_WS:-$HOME/simu5g-workspace}/inet-4.6.0/src"
LIB="${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/src/simu5g"
LOG="new_seeds_results.txt"
: > "$LOG"
for SEED in 54 55 56; do
  for POWER in 37 43 46; do
    CFG="Scenario06Bs${SEED}_p${POWER}"
    DONE=0
    for r in 0 1 2 3 4; do
      printf "%s r=%s ... " "$CFG" "$r" | tee -a "$LOG"
      if opp_run -l "$LIB" -n "$NED" -f omnetpp_s${SEED}.ini -c "$CFG" -u Cmdenv -r $r 2>&1 | tail -3 | grep -q "finish()"; then
        echo "OK" | tee -a "$LOG"
        echo "$CFG r=$r" >> s06b_successful_runs.txt
        DONE=1
        break
      fi
      echo "CRASH" | tee -a "$LOG"
    done
    [ $DONE -eq 0 ] && echo "FAILED: $CFG exhausted all 5 repetitions" | tee -a "$LOG"
  done
done
echo "=== SUMMARY ===" | tee -a "$LOG"
echo "completed: $(grep -c ' OK$' "$LOG")/9" | tee -a "$LOG"
grep '^FAILED' "$LOG"
