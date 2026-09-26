"""
gen_ini_s08.py -- scenario08 generator: 8 gNB / 80 UE, mobility +
handover + QoS bursty UL. SEED=41 python3 gen_ini_s08.py

Design decisions, stated so they are reviewable:
- 2x4 grid over 4000x2000m, inter-site distance 1000m: same density as
  scenario07, so the tested variable is SCALE, not congestion.
- Traffic schedule: scenario07's 40-phase schedule TILED to 8 gNBs
  (state[g] = scenario07_state[g % 4]). This preserves the all-H
  congestion phases (10, 19, 24, 25) exactly -- the samples that made
  delay learnable -- and keeps uniform calibration phases uniform.
- Draw order (30->80 mobility draws, then QoS draws) matches the
  extraction replay convention: position/speed/heading per UE first,
  then qos per UE. Extraction scripts must replay with the same NUM_UES.
- X2 wiring mirrors the working 4-cell pattern generalized: gnb_i has
  numX2Apps=7, its app k connects to the k-th OTHER gNB in ascending
  index order; server ports 5000 + i*100 + appIndex.
"""
import random, math, os
from collections import Counter

SEED = int(os.environ.get("SEED", "41"))
# Slow-UE variant: 1-3 m/s instead of 3-8. The NrMacGnb crash is a
# handover race, and handover frequency scales with UE speed, so this
# is the cheapest test of whether reducing handover pressure makes
# 8 gNB / 80 UE runnable at all. Cost: not directly comparable with
# the 4-gNB runs, which use 3-8 m/s.
SPEED_MIN, SPEED_MAX = 1.0, 3.0
random.seed(SEED)

NUM_GNB = 8
NUM_UE = 80
GNB_POSITIONS = {0:(500,500),1:(1500,500),2:(2500,500),3:(3500,500),
                 4:(500,1500),5:(1500,1500),6:(2500,1500),7:(3500,1500)}

def closest_gnb(x, y):
    return min(GNB_POSITIONS, key=lambda g: math.hypot(x-GNB_POSITIONS[g][0], y-GNB_POSITIONS[g][1]))

phases4 = [
    (0,3,'L','L','L','L'),(3,6,'H','L','L','L'),(6,9,'L','H','L','L'),
    (9,12,'L','L','H','L'),(12,15,'L','L','L','H'),(15,18,'H','H','L','L'),
    (18,21,'L','L','H','H'),(21,24,'H','L','H','L'),(24,27,'L','H','L','H'),
    (27,30,'L','L','L','L'),(30,33,'H','H','H','H'),(33,36,'H','L','L','L'),
    (36,39,'L','H','L','L'),(39,42,'L','L','H','L'),(42,45,'L','L','L','H'),
    (45,48,'H','L','L','H'),(48,51,'L','H','H','L'),(51,54,'M','M','M','M'),
    (54,57,'L','L','L','L'),(57,60,'H','H','H','H'),(60,63,'L','L','L','L'),
    (63,66,'L','L','L','L'),(66,69,'M','M','M','M'),(69,72,'M','M','M','M'),
    (72,75,'H','H','H','H'),(75,78,'H','H','H','H'),(78,81,'H','M','M','M'),
    (81,84,'M','H','M','M'),(84,87,'M','M','H','M'),(87,90,'M','M','M','H'),
    (90,93,'H','H','M','M'),(93,96,'M','M','H','H'),(96,99,'H','M','H','M'),
    (99,102,'M','H','M','H'),(102,105,'H','M','M','L'),(105,108,'L','H','M','M'),
    (108,111,'M','L','H','M'),(111,114,'M','M','L','H'),(114,117,'H','H','L','L'),
    (117,120,'L','L','H','H'),
]
# tile 4-gNB states to 8
phases = [(p[0], p[1]) + tuple(p[2 + (g % 4)] for g in range(NUM_GNB)) for p in phases4]

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

CFG = f"Scenario08slow{SEED}"
L = []
L.append('[General]')
L.append('sim-time-limit = 120s')
L.append('**.routingRecorder.enabled = false')
L.append('output-scalar-file = ${resultdir}/${configname}/${repetition}.sca')
L.append('output-vector-file = ${resultdir}/${configname}/${repetition}.vec')
L.append('seed-set = ${repetition}')
L.append('**.vector-recording = true')
L.append('**.numBands = 50')
L.append('**.mobility.constraintAreaMaxX = 4000m')
L.append('**.mobility.constraintAreaMaxY = 2000m')
L.append('**.mobility.constraintAreaMinX = 0m')
L.append('**.mobility.constraintAreaMinY = 0m')
L.append('**.mobility.constraintAreaMinZ = 0m')
L.append('**.mobility.constraintAreaMaxZ = 0m')
L.append('*.configurator.config = xmldoc("./demo.xml")')
L.append('')
L.append(f'[Config {CFG}]')
L.append('network = simu5g.simulations.nr.networks.EightCell_Standalone')
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
    x = random.uniform(200, 3800)
    y = random.uniform(200, 1800)
    speed = random.uniform(SPEED_MIN, SPEED_MAX)
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

fname = f'omnetpp_slow{SEED}.ini'
with open(fname, 'w') as f:
    f.write('\n'.join(L))
with open(f'ue_assignment_slow{SEED}.txt', 'w') as f:
    f.write(f"seed = {SEED}  (scenario08: 8 gNB / 80 UE)\n")
    for u in range(NUM_UE):
        f.write(f"UE {u}: home_gnb={ue_home_gnb[u]}, qos={qos_assignments[u]}\n")

print(f"Written {fname}, {len(L)} lines")
print("home gNB distribution:", dict(sorted(Counter(ue_home_gnb.values()).items())))
print("qos distribution:", dict(Counter(qos_assignments.values())))
