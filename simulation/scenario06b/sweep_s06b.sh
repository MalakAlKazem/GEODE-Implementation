#!/bin/bash
# Sweep every (seed, power, repetition) combination for scenario06b.
# Records outcome per attempt so we know exactly what completed.
NED="${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/simulations:${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/emulation:${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/src:${SIMU5G_WS:-$HOME/simu5g-workspace}/inet-4.6.0/src"
LIB="${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/src/simu5g"
LOG="sweep_results.txt"
: > "$LOG"
for SEED in 51 52 53; do
  for POWER in 37 43 46; do
    CFG="Scenario06Bs${SEED}_p${POWER}"
    for r in 0 1 2 3 4; do
      if [ -d "results/${CFG}" ] && [ -f "results/${CFG}/${r}.vec" ]; then
        echo "SKIP  $CFG r=$r (already exists)" | tee -a "$LOG"
        continue
      fi
      printf "RUN   %s r=%s ... " "$CFG" "$r" | tee -a "$LOG"
      if opp_run -l "$LIB" -n "$NED" -f omnetpp_s${SEED}.ini -c "$CFG" -u Cmdenv -r $r 2>&1 | tail -3 | grep -q "finish()"; then
        echo "OK" | tee -a "$LOG"
      else
        echo "CRASH" | tee -a "$LOG"
      fi
    done
  done
done
echo "=== SUMMARY ===" | tee -a "$LOG"
echo "OK:    $(grep -c ' OK$' "$LOG")" | tee -a "$LOG"
echo "CRASH: $(grep -c ' CRASH$' "$LOG")" | tee -a "$LOG"
echo "SKIP:  $(grep -c '^SKIP' "$LOG")" | tee -a "$LOG"
