"""
fix_positions.py -- corrects pos_x / pos_y in every extracted dataset.

    python3 fix_positions.py --check      # report only, writes nothing
    python3 fix_positions.py --apply      # rewrite ue_inputs.csv in place

THE BUG
-------
INET's MobilityBase defaults `initFromDisplayString = true`, which takes
a module's starting position from the NED `@display("p=...")` string and
IGNORES `initialX` / `initialY` from the .ini. FourCell_Standalone
declares `ue[numUe]: NrUe { @display("p=1000,1000") }`, so ALL 30 UEs
started at the map centre, not at the random positions the .ini
specified.

Speed and heading were NOT overridden -- those come from the .ini and
were applied. So each UE's trajectory has the correct velocity vector
from the wrong origin.

Measured on seed 22 p43, serving-cell agreement with a nearest-cell
reconstruction over t=40-60s:
    origin = .ini values      35.9%   <- what the extractors assumed
    origin = (1000,1000)      95.0%   <- what actually happened
A control run with `initFromDisplayString = false` gave 91.9% using the
.ini origin, confirming the flag is the cause rather than some other
difference.

SCOPE
-----
Only pos_x and pos_y are affected. Everything else is either read from
simulation output (energy, delay, jitter, packet loss, SINR, serving
edges) or replayed correctly from the generator's RNG (speed_mps,
qos_class, home_gnb_index). No re-simulation is needed: the true
trajectories are fully determined by (1000,1000) plus each UE's correct
speed and heading, both of which are known.

home_gnb_index is left as-is deliberately. It records which gNB's
traffic schedule the UE follows -- assigned by the generator from the
INTENDED start position -- and that assignment really did drive the
traffic, so it is correct as a description of the traffic pattern even
though it no longer describes geography.

WHY NOT RE-SIMULATE
-------------------
The analysis discards the first 30-51 seconds of every run as the
handover-settling window. At 3-8 m/s UEs disperse 150-400 m within that
window and are spread across the 2000 m map before the first retained
snapshot. The common origin therefore affects only data that is thrown
away. Re-simulating all 68 runs would cost a day, rebuild the sleep-pair
coverage from scratch, and risk the NrMacGnb crashes again, to change
positions in a discarded window.

Future scenarios should set `**.mobility.initFromDisplayString = false`
so the .ini positions are honoured. scenario09 (dense) already does.
"""

import argparse
import glob
import math
import os
import random
import re

import pandas as pd

# The origin is each NETWORK's own UE @display("p=...") point, not a
# constant. FourCell_Standalone declares p=1000,1000 (centre of its
# 2000x2000 area); SixCell_Standalone declares p=1500,1000 (centre of
# 3000x2000). Verified against serving-cell ground truth:
#   scenario07 seed 22: (1000,1000) 95.0%, and
#   scenario06b seed 51: (1500,1000) 95.9% vs (1000,1000) 49.9%.
# Using one constant for both would leave scenario06b wrong in a new way.
ORIGIN_4G = (1000.0, 1000.0)     # FourCell_Standalone
ORIGIN_6G = (1500.0, 1000.0)     # SixCell_Standalone
MAP_SIZE_4G = 2000.0       # scenario07 and the sleep pairs: 2000x2000
MAP_X_6G, MAP_Y_6G = 3000.0, 2000.0   # scenario06b: non-square

QOS_WEIGHTS = {"eMBB": 0.4, "URLLC": 0.3, "mMTC": 0.3}


def replay(seed, num_ues, x_range):
    """Reproduce the generator's per-UE draws IN ORDER: x, y, speed, heading.

    x and y are drawn and then ignored by INET (the display string wins),
    but they MUST still be drawn here or the RNG stream desynchronises and
    every subsequent speed and heading would be wrong. Both scenario07 and
    scenario06b use y in (200, 1800); only the x range differs.
    """
    random.seed(seed)
    mob = {}
    for u in range(num_ues):
        random.uniform(*x_range)          # x: drawn, overridden by INET
        random.uniform(200, 1800)         # y: drawn, overridden by INET
        sp = random.uniform(3, 8)
        hd = random.uniform(0, 360)
        mob[u] = {"speed": sp, "heading": hd}
    return mob


def reflect(pos, size):
    period = 2 * size
    m = pos % period
    return period - m if m > size else m


def true_pos(m, t, map_x, map_y, origin):
    r = math.radians(m["heading"])
    return (reflect(origin[0] + m["speed"] * math.cos(r) * t, map_x),
            reflect(origin[1] + m["speed"] * math.sin(r) * t, map_y))


def folder_spec(name):
    """(seed, num_ues, map_x, map_y, trim, x_range, origin) or None."""
    # scenario06b: 6 gNB / 60 UE, non-square 3000x2000
    m = re.match(r"extracted_s06b(\d+)_p\d+$", name)
    if m:
        seed = int(m.group(1))
        trim = {51: 180, 52: 180, 53: 240}[seed]
        return seed, 60, MAP_X_6G, MAP_Y_6G, trim, (200, 2800), ORIGIN_6G
    # forced-sleep pairs: seed from the folder, 4 gNB / 30 UE, untrimmed
    m = re.match(r"extracted_sleep_s(\d+)_p\d+(_sleep\d+)?$", name)
    if m:
        return int(m.group(1)), 30, MAP_SIZE_4G, MAP_SIZE_4G, 0, (200, 1800), ORIGIN_4G
    # scenario07 seeds 22-27
    m = re.match(r"extracted_s(2[2-7])_p\d+$", name)
    if m:
        seed = int(m.group(1))
        trim = {22: 360, 23: 360, 24: 360, 25: 510, 26: 420, 27: 300}[seed]
        return seed, 30, MAP_SIZE_4G, MAP_SIZE_4G, trim, (200, 1800), ORIGIN_4G
    # scenario07 seed 21 (unprefixed folders), two-stage trim totalling 370
    m = re.match(r"extracted_p\d+$", name)
    if m:
        return 21, 30, MAP_SIZE_4G, MAP_SIZE_4G, 370, (200, 1800), ORIGIN_4G
    return None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default="results")
    args = ap.parse_args()

    folders = sorted(glob.glob(os.path.join(args.root, "extracted_*")))
    if not folders:
        raise SystemExit(f"no extracted_* folders under {args.root}")

    done = skipped = 0
    for f in folders:
        name = os.path.basename(f)
        spec = folder_spec(name)
        if spec is None:
            print(f"{name:<34} UNRECOGNISED naming, skipped")
            skipped += 1
            continue
        seed, num_ues, map_x, map_y, trim, x_range, origin = spec

        path = os.path.join(f, "ue_inputs.csv")
        if not os.path.exists(path):
            print(f"{name:<34} no ue_inputs.csv, skipped")
            skipped += 1
            continue

        ue = pd.read_csv(path)
        mob = replay(seed, num_ues, x_range)

        # snapshot_id 0 corresponds to absolute time trim*0.1s; positions
        # are evaluated at the window midpoint, as the extractors did.
        t = (ue.snapshot_id + trim) * 0.1 + 0.05
        new_x, new_y = [], []
        for idx, u in zip(t.values, ue.ue_index.values):
            x, y = true_pos(mob[int(u)], float(idx), map_x, map_y, origin)
            new_x.append(x)
            new_y.append(y)

        dx = (pd.Series(new_x) - ue.pos_x).abs().mean()
        dy = (pd.Series(new_y) - ue.pos_y).abs().mean()

        if args.apply:
            ue["pos_x"] = new_x
            ue["pos_y"] = new_y
            ue.to_csv(path, index=False)

        print(f"{name:<34} seed {seed:<3} trim {trim:<4} "
              f"origin ({origin[0]:.0f},{origin[1]:.0f})  "
              f"mean |Δx| {dx:6.1f}m  |Δy| {dy:6.1f}m"
              f"{'  WRITTEN' if args.apply else ''}")
        done += 1

    print()
    print(f"{done} folders processed, {skipped} skipped")
    if args.check:
        print("check only -- nothing written. Re-run with --apply to fix.")
    else:
        print("pos_x / pos_y rewritten. Re-run the experiments: the feature")
        print("vector has changed, so all previously recorded numbers are stale.")


if __name__ == "__main__":
    main()
