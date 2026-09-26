import random, math, os

SEED = int(os.environ.get("SEED", "22"))
random.seed(SEED)

GNB_POSITIONS = {0: (500,500), 1: (1500,500), 2: (500,1500), 3: (1500,1500)}

def closest_gnb(x, y):
    best_g, best_d = None, float('inf')
    for g, (gx, gy) in GNB_POSITIONS.items():
        d = math.hypot(x-gx, y-gy)
        if d < best_d:
            best_d, best_g = d, g
    return best_g

phases = [
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

MEAN_RATE = {"H": 0.0015, "M": 0.0075, "L": 0.15}
GAPS = {"H": 1.5, "M": 0.7, "L": 0.0}
NUM_PHASES = len(phases)
QOS_PROFILES = {"eMBB": (1000,1450), "URLLC": (100,200), "mMTC": (60,140)}
QOS_WEIGHTS = {"eMBB": 0.4, "URLLC": 0.3, "mMTC": 0.3}

def assign_qos():
    r = random.random()
    cum = 0.0
    for cls, w in QOS_WEIGHTS.items():
        cum += w
        if r < cum:
            return cls
    return "eMBB"

CFG = f"Scenario07s{SEED}"
L = []
L.append('[General]')
L.append('sim-time-limit = 120s')
L.append('**.routingRecorder.enabled = false')
L.append('output-scalar-file = ${resultdir}/${configname}/${repetition}.sca')
L.append('output-vector-file = ${resultdir}/${configname}/${repetition}.vec')
L.append('seed-set = ${repetition}')
L.append('**.vector-recording = true')
L.append('**.numBands = 50')
L.append('**.mobility.constraintAreaMaxX = 2000m')
L.append('**.mobility.constraintAreaMaxY = 2000m')
L.append('**.mobility.constraintAreaMinX = 0m')
L.append('**.mobility.constraintAreaMinY = 0m')
L.append('**.mobility.constraintAreaMinZ = 0m')
L.append('**.mobility.constraintAreaMaxZ = 0m')
L.append('*.configurator.config = xmldoc("./demo.xml")')
L.append('')
L.append(f'[Config {CFG}]')
L.append('network = simu5g.simulations.nr.networks.FourCell_Standalone')
L.append('*.gnb0.mobility.initialX = 500m')
L.append('*.gnb0.mobility.initialY = 500m')
L.append('*.gnb1.mobility.initialX = 1500m')
L.append('*.gnb1.mobility.initialY = 500m')
L.append('*.gnb2.mobility.initialX = 500m')
L.append('*.gnb2.mobility.initialY = 1500m')
L.append('*.gnb3.mobility.initialX = 1500m')
L.append('*.gnb3.mobility.initialY = 1500m')
L.append('**.eNodeBTxPower = 43dBm')
L.append('**.ueTxPower = 26dBm')
L.append('**.targetBler = 0.01')
L.append('**.blerShift = 5')
L.append('**.enableHandover = true')
L.append('*.gnb0.numX2Apps = 3')
L.append('*.gnb1.numX2Apps = 3')
L.append('*.gnb2.numX2Apps = 3')
L.append('*.gnb3.numX2Apps = 3')
L.append('*.gnb0.x2App[*].server.localPort = 5000 + ancestorIndex(1)')
L.append('*.gnb1.x2App[*].server.localPort = 5100 + ancestorIndex(1)')
L.append('*.gnb2.x2App[*].server.localPort = 5200 + ancestorIndex(1)')
L.append('*.gnb3.x2App[*].server.localPort = 5300 + ancestorIndex(1)')
L.append('*.gnb0.x2App[0].client.connectAddress = "gnb1%x2ppp0"')
L.append('*.gnb0.x2App[1].client.connectAddress = "gnb2%x2ppp0"')
L.append('*.gnb0.x2App[2].client.connectAddress = "gnb3%x2ppp0"')
L.append('*.gnb1.x2App[0].client.connectAddress = "gnb0%x2ppp0"')
L.append('*.gnb1.x2App[1].client.connectAddress = "gnb2%x2ppp0"')
L.append('*.gnb1.x2App[2].client.connectAddress = "gnb3%x2ppp0"')
L.append('*.gnb2.x2App[0].client.connectAddress = "gnb0%x2ppp0"')
L.append('*.gnb2.x2App[1].client.connectAddress = "gnb1%x2ppp0"')
L.append('*.gnb2.x2App[2].client.connectAddress = "gnb3%x2ppp0"')
L.append('*.gnb3.x2App[0].client.connectAddress = "gnb0%x2ppp0"')
L.append('*.gnb3.x2App[1].client.connectAddress = "gnb1%x2ppp0"')
L.append('*.gnb3.x2App[2].client.connectAddress = "gnb2%x2ppp0"')
L.append('**.schedulingDisciplineDl = "PF"')
L.append('**.schedulingDisciplineUl = "PF"')
L.append('*.numUe = 30')
L.append('')

ue_home_gnb = {}
for u in range(30):
    x = random.uniform(200, 1800)
    y = random.uniform(200, 1800)
    speed = random.uniform(3, 8)
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
L.append(f'*.server.numApps = {30 * NUM_PHASES}')
L.append(f'*.ue[*].numApps = {NUM_PHASES}')
L.append('')

qos_assignments = {u: assign_qos() for u in range(30)}

for u in range(30):
    home_g = ue_home_gnb[u]
    min_len, max_len = QOS_PROFILES[qos_assignments[u]]
    for pi, phase in enumerate(phases):
        t_start, t_end = phase[0], phase[1]
        state = phase[2 + home_g]
        mean_rate = MEAN_RATE[state]
        gap = GAPS[state]
        stop = max(t_end - gap, t_start + 0.1)
        sai = u * NUM_PHASES + pi
        L.append(f'*.ue[{u}].app[{pi}].typename = "UdpBasicApp"')
        L.append(f'*.ue[{u}].app[{pi}].destAddresses = "server"')
        L.append(f'*.ue[{u}].app[{pi}].destPort = {4000+sai}')
        L.append(f'*.ue[{u}].app[{pi}].localPort = {5000+sai}')
        L.append(f'*.ue[{u}].app[{pi}].messageLength = int(uniform({min_len},{max_len})) * 1B')
        L.append(f'*.ue[{u}].app[{pi}].sendInterval = exponential({mean_rate}s)')
        L.append(f'*.ue[{u}].app[{pi}].startTime = {t_start}s')
        L.append(f'*.ue[{u}].app[{pi}].stopTime = {stop}s')
        L.append(f'*.server.app[{sai}].typename = "UdpSink"')
        L.append(f'*.server.app[{sai}].localPort = {4000+sai}')

L.append('')
for p in (37, 43, 46):
    L.append(f'[Config {CFG}_p{p}]')
    L.append(f'extends = {CFG}')
    L.append(f'**.eNodeBTxPower = {p}dBm')
    L.append('')

fname = f'omnetpp_s{SEED}.ini'
with open(fname, 'w') as f:
    f.write('\n'.join(L))
with open(f'ue_assignment_s{SEED}.txt', 'w') as f:
    f.write(f"seed = {SEED}\n")
    for u in range(30):
        f.write(f"UE {u}: home_gnb={ue_home_gnb[u]}, qos={qos_assignments[u]}\n")

from collections import Counter
print(f"Written {fname}, {len(L)} lines")
print("home gNB distribution:", Counter(ue_home_gnb.values()))
