#!/bin/bash
# Baselines (3 powers) + sleep configs (6 cells, p43 only) per seed.
# Repetitions are crash recovery only -- they share a topology, so a
# second completed repetition adds rows without adding information.
NED="${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/simulations:${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/emulation:${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/src:${SIMU5G_WS:-$HOME/simu5g-workspace}/inet-4.6.0/src"
LIB="${SIMU5G_WS:-$HOME/simu5g-workspace}/simu5g-1.4.4/src/simu5g"
LOG="s06c_runs.txt"
for SEED in "$@"; do
  CFGS="Scenario06Cs${SEED}_p37 Scenario06Cs${SEED}_p43 Scenario06Cs${SEED}_p46"
  for k in 0 1 2 3 4 5; do CFGS="$CFGS Scenario06Cs${SEED}_p43_sleep${k}"; done
  for CFG in $CFGS; do
    DONE=0
    for r in 0 1 2 3 4; do
      printf "%-34s r=%s ... " "$CFG" "$r"
      if opp_run -l "$LIB" -n "$NED" -f omnetpp_s${SEED}.ini -c "$CFG" -u Cmdenv -r $r 2>&1 | tail -3 | grep -q "finish()"; then
        echo "OK"; echo "$CFG r=$r" >> "$LOG"; DONE=1; break
      fi
      echo "CRASH"
    done
    [ $DONE -eq 0 ] && echo "FAILED: $CFG" | tee -a "$LOG"
  done
done
echo "=== done ==="
