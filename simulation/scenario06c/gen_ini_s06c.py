"""
gen_ini_s06c.py -- scenario06c generator: 6 gNB / 60 UE, mobility +
handover + QoS bursty UL, WITH paired sleep configs.

    SEED=61 python3 gen_ini_s06c.py

What changed from scenario06b, and why
--------------------------------------
1. initFromDisplayString = false.  THE reason 06c exists.  INET's
   MobilityBase defaults this to true, which takes each UE's start
   position from the NED @display("p=...") string and IGNORES
   initialX/initialY from the ini.  SixCell_Standalone declares
   p=1500,1000, so in 06b all 60 UEs started within ~130 m of the map
   centre and dispersed outward for the whole run (measured: pos_x std
   67 m at t=0, still only 446 m at t=120 s, 75% of UEs inside the
   central third at the END).  gnb1 (1500,500) and gnb4 (1500,1500) sit
   either side of that point and absorbed ~90% of all UE-snapshots,
   giving 16-27 UEs on two cells while gnb2 saw 250 samples in a whole
   run.  That, not the traffic schedule, is what produced 06b's
   16-second delay tail.  fix_positions.py repairs the RECORDED
   positions but cannot undo what the simulator actually did.

2. Genuine 6-cell traffic schedule.  06b built its phases by tiling a
   4-gNB schedule with `state[g] = state4[g % 4]`, so gnb4 always
   mirrored gnb0 and gnb5 always mirrored gnb1.  With positions fixed
   that becomes a real artefact: two pairs of cells peak in lockstep
   forever, an artificial correlation a GNN can learn instead of
   genuine relational structure.  The schedule below gives all six
   cells independent states while keeping the properties that made
   delay learnable: all-H congestion phases (12, 20, 38), uniform
   all-L and all-M calibration phases, single-cell-H rotations through
   every cell, and pair/column/half-half combinations.

3. Sleep configs built in.  scenario07 needed a separate
   gen_sleep_configs.py run afterwards; here the paired sleep configs
   are emitted with everything else so a seed is never half-built.
   Six sleep variants (one per cell) at p43 only.  KPI labels do not
   depend on eNodeBTxPower -- delay/jitter/loss are identical across
   p37/p43/p46 -- so sleeping every cell at all three powers would
   triple the simulation cost for no new KPI information.  Baselines
   still run at all three powers, which is where power genuinely
   matters (the energy head).

   1 dBm rather than 0: keeps the cell nominally alive (no division by
   zero in path-loss maths) while making it unusable to any UE.

UNCHANGED ON PURPOSE
--------------------
RNG draw order is exactly 06b's: per UE x, y, speed, heading for all
NUM_UE, THEN NUM_UE qos draws.  extract_scenario06b.py replays this
sequence to reconstruct positions and QoS labels; any deviation shifts
the whole stream and every value comes out wrong with no error raised.
Energy stays derivable from RB occupancy alone -- active_state is NOT
written as a gNB input anywhere, preserving the non-circularity
principle (the sleep flag stays diagnostic-only, so energy R^2 reflects
a learnable physical relationship and not leakage).
"""
import random, math, os
from collections import Counter

SEED = int(os.environ.get("SEED", "61"))
random.seed(SEED)

NUM_GNB = 6
NUM_UE = 60
SLEEP_POWER = 43
SLEEP_DBM = 1
# 3GPP TR 38.901 Urban Macro reference ISD is 500 m. scenario06b/06c-v1
# used 1000 m, which halves the rate at which UEs cross cell boundaries.
# 3x2 grid at 500 m spacing over a 1500x1000 m area.
GNB_POSITIONS = {0:(250,250),1:(750,250),2:(1250,250),
                 3:(250,750),4:(750,750),5:(1250,750)}

def closest_gnb(x, y):
    return min(GNB_POSITIONS, key=lambda g: math.hypot(x-GNB_POSITIONS[g][0], y-GNB_POSITIONS[g][1]))

# 40 phases x 3 s = 120 s.  Six independent per-cell states, no tiling.
phases = [
    (0,3,    'L','L','L','L','L','L'),
    (3,6,    'H','L','L','L','L','L'),
    (6,9,    'L','H','L','L','L','L'),
    (9,12,   'L','L','H','L','L','L'),
    (12,15,  'L','L','L','H','L','L'),
    (15,18,  'L','L','L','L','H','L'),
    (18,21,  'L','L','L','L','L','H'),
    (21,24,  'L','L','L','L','L','L'),
    (24,27,  'H','H','L','L','L','L'),
    (27,30,  'L','L','H','H','L','L'),
    (30,33,  'L','L','L','L','H','H'),
    (33,36,  'H','H','H','H','H','H'),
    (36,39,  'L','L','L','L','L','L'),
    (39,42,  'H','L','H','L','H','L'),
    (42,45,  'L','H','L','H','L','H'),
    (45,48,  'M','M','M','M','M','M'),
    (48,51,  'H','L','L','H','L','L'),
    (51,54,  'L','H','L','L','H','L'),
    (54,57,  'L','L','H','L','L','H'),
    (57,60,  'H','H','H','H','H','H'),
    (60,63,  'L','L','L','L','L','L'),
    (63,66,  'M','M','M','M','M','M'),
    (66,69,  'H','M','M','M','M','M'),
    (69,72,  'M','H','M','M','M','M'),
    (72,75,  'M','M','H','M','M','M'),
    (75,78,  'M','M','M','H','M','M'),
    (78,81,  'M','M','M','M','H','M'),
    (81,84,  'M','M','M','M','M','H'),
    (84,87,  'H','H','M','M','L','L'),
    (87,90,  'L','L','H','H','M','M'),
    (90,93,  'M','M','L','L','H','H'),
    (93,96,  'H','H','H','L','L','L'),
    (96,99,  'L','L','L','H','H','H'),
    (99,102, 'M','M','M','M','M','M'),
    (102,105,'H','M','L','H','M','L'),
    (105,108,'L','H','M','L','H','M'),
    (108,111,'M','L','H','M','L','H'),
    (111,114,'H','H','H','H','H','H'),
    (114,117,'L','L','L','L','L','L'),
    (117,120,'M','M','M','M','M','M'),
]

MEAN_RATE = {"H": 0.0015, "M": 0.0075, "L": 0.15}
GAPS = {"H": 1.5, "M": 0.7, "L": 0.0}
NUM_PHASES = len(phases)
QOS_PROFILES = {"eMBB": (1000,1450), "URLLC": (100,200), "mMTC": (60,140)}
QOS_WEIGHTS = {"eMBB": 0.4, "URLLC": 0.3, "mMTC": 0.3}

def assign_qos():
    r = random.random(); cum = 0.0
    for cls, w in QOS_WEIGHTS.items():
        cum += w
        if r < cum: return cls
    return "eMBB"

CFG = f"Scenario06Cs{SEED}"
L = []
L.append('[General]')
L.append('sim-time-limit = 120s')
L.append('**.routingRecorder.enabled = false')
L.append('output-scalar-file = ${resultdir}/${configname}/${repetition}.sca')
L.append('output-vector-file = ${resultdir}/${configname}/${repetition}.vec')
L.append('seed-set = ${repetition}')
L.append('**.vector-recording = true')
# 25 RBs, not 50. At ISD 500 m every UE sits close to its serving cell,
# so SINR and spectral efficiency rise and 50 RBs carried the offered
# load with almost no contention: delay std collapsed 83->11 ms and
# packet loss fell ~23x, leaving the loss head nothing to learn and
# CMOA no gradient (every sleep would look safe). Halving the band
# count restores contention at the 3GPP-aligned spacing.
L.append('**.numBands = 25')
L.append('# See docstring note 1: without this, every UE starts at the')
L.append('# NED display-string point (1500,1000) and initialX/Y is ignored.')
L.append('**.mobility.initFromDisplayString = false')
L.append('**.mobility.constraintAreaMaxX = 1500m')
L.append('**.mobility.constraintAreaMaxY = 1000m')
L.append('**.mobility.constraintAreaMinX = 0m')
L.append('**.mobility.constraintAreaMinY = 0m')
L.append('**.mobility.constraintAreaMinZ = 0m')
L.append('**.mobility.constraintAreaMaxZ = 0m')
L.append('*.configurator.config = xmldoc("./demo.xml")')
L.append('')
L.append(f'[Config {CFG}]')
L.append('network = simu5g.simulations.nr.networks.SixCell_Standalone')
for g,(x,y) in GNB_POSITIONS.items():
    L.append(f'*.gnb{g}.mobility.initialX = {x}m')
    L.append(f'*.gnb{g}.mobility.initialY = {y}m')
L.append('**.eNodeBTxPower = 43dBm')
L.append('**.ueTxPower = 26dBm')
L.append('**.targetBler = 0.01')
L.append('**.blerShift = 5')
L.append('**.enableHandover = true')
L.append('')
for g in range(NUM_GNB):
    L.append(f'*.gnb{g}.numX2Apps = {NUM_GNB-1}')
    L.append(f'*.gnb{g}.x2App[*].server.localPort = {5000+g*100} + ancestorIndex(1)')
for g in range(NUM_GNB):
    peers = [p for p in range(NUM_GNB) if p != g]
    for k, p in enumerate(peers):
        L.append(f'*.gnb{g}.x2App[{k}].client.connectAddress = "gnb{p}%x2ppp0"')
L.append('**.schedulingDisciplineDl = "PF"')
L.append('**.schedulingDisciplineUl = "PF"')
L.append(f'*.numUe = {NUM_UE}')
L.append('')

ue_home_gnb = {}
for u in range(NUM_UE):
    x = random.uniform(100, 1400)
    y = random.uniform(100, 900)
    # Heterogeneous mobility. A single uniform(3,8) draw put every UE at
    # 11-29 km/h -- too fast for a pedestrian, too slow for a vehicle, so
    # nobody in the simulation moved like anything real. 3GPP TR 38.901
    # evaluates UMa at 3 km/h (pedestrian) and 30 km/h (vehicular); a real
    # cell serves a mix. Two draws per UE (class, then speed) instead of
    # one -- extract_scenario06c.py replays this EXACT sequence, so both
    # files must change together or every heading and QoS label after this
    # point comes out wrong with no error raised.
    mc = random.random()
    if mc < 0.60:
        speed = random.uniform(1.0, 2.0)      # pedestrian, 3.6-7.2 km/h
    elif mc < 0.90:
        speed = random.uniform(8.0, 15.0)     # urban vehicular, 29-54 km/h
    else:
        speed = random.uniform(15.0, 25.0)    # fast vehicular, 54-90 km/h
    heading = random.uniform(0, 360)
    ue_home_gnb[u] = closest_gnb(x, y)
    L.append(f'*.ue[{u}].mobility.typename = "LinearMobility"')
    L.append(f'*.ue[{u}].mobility.initialX = {x:.1f}m')
    L.append(f'*.ue[{u}].mobility.initialY = {y:.1f}m')
    L.append(f'*.ue[{u}].mobility.speed = {speed:.2f}mps')
    L.append(f'*.ue[{u}].mobility.initialMovementHeading = {heading:.1f}deg')
    L.append(f'*.ue[{u}].mobility.borderPolicy = "reflect"')
    L.append(f'*.ue[{u}].nrServingNodeId = {ue_home_gnb[u]+1}')
L.append('*.ue[*].servingNodeId = 0')
L.append('')
L.append(f'*.server.numApps = {NUM_UE * NUM_PHASES}')
L.append(f'*.ue[*].numApps = {NUM_PHASES}')
L.append('')

qos_assignments = {u: assign_qos() for u in range(NUM_UE)}

for u in range(NUM_UE):
    home_g = ue_home_gnb[u]
    min_len, max_len = QOS_PROFILES[qos_assignments[u]]
    for pi, phase in enumerate(phases):
        t_start, t_end = phase[0], phase[1]
        state = phase[2 + home_g]
        gap = GAPS[state]
        stop = max(t_end - gap, t_start + 0.1)
        sai = u * NUM_PHASES + pi
        L.append(f'*.ue[{u}].app[{pi}].typename = "UdpBasicApp"')
        L.append(f'*.ue[{u}].app[{pi}].destAddresses = "server"')
        L.append(f'*.ue[{u}].app[{pi}].destPort = {10000+sai}')
        L.append(f'*.ue[{u}].app[{pi}].localPort = {30000+sai}')
        L.append(f'*.ue[{u}].app[{pi}].messageLength = int(uniform({min_len},{max_len})) * 1B')
        L.append(f'*.ue[{u}].app[{pi}].sendInterval = exponential({MEAN_RATE[state]}s)')
        L.append(f'*.ue[{u}].app[{pi}].startTime = {t_start}s')
        L.append(f'*.ue[{u}].app[{pi}].stopTime = {stop}s')
        L.append(f'*.server.app[{sai}].typename = "UdpSink"')
        L.append(f'*.server.app[{sai}].localPort = {10000+sai}')

L.append('')
for p in (37, 43, 46):
    L.append(f'[Config {CFG}_p{p}]')
    L.append(f'extends = {CFG}')
    L.append(f'**.eNodeBTxPower = {p}dBm')
    L.append('')

L.append(f'# --- paired sleep configs, p{SLEEP_POWER} only (see docstring note 3) ---')
L.append('')
for k in range(NUM_GNB):
    L.append(f'[Config {CFG}_p{SLEEP_POWER}_sleep{k}]')
    L.append(f'extends = {CFG}_p{SLEEP_POWER}')
    L.append(f'*.gnb{k}.cellularNic.phy.eNodeBTxPower = {SLEEP_DBM}dBm')
    L.append('')

fname = f'omnetpp_s{SEED}.ini'
with open(fname, 'w') as f:
    f.write('\n'.join(L))
with open(f'ue_assignment_s{SEED}.txt', 'w') as f:
    f.write(f"seed = {SEED}  (scenario06c: {NUM_GNB} gNB / {NUM_UE} UE)\n")
    for u in range(NUM_UE):
        f.write(f"UE {u}: home_gnb={ue_home_gnb[u]}, qos={qos_assignments[u]}\n")

print(f"Written {fname}, {len(L)} lines")
print("home gNB distribution:", dict(sorted(Counter(ue_home_gnb.values()).items())))
print("qos distribution:", dict(Counter(qos_assignments.values())))
print(f"configs: {CFG}_p{{37,43,46}} + {CFG}_p{SLEEP_POWER}_sleep{{0..{NUM_GNB-1}}}")
print()
print("VERIFY BEFORE TRUSTING ANY SLEEP RUN: confirm the slept cell")
print("actually lost its UEs (serving_gnb_index k absent after settling).")
print("OMNeT++ accepts a module path matching nothing without complaint,")
print("which would leave the cell at full power and produce believable")
print("but meaningless data.")
