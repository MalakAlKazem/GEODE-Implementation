"""
compare_pairs.py -- the energy/QoS trade-off of each sleep action.

    SEED=22 python3 compare_pairs.py

For every (baseline, sleep-k) pair it reports, on IDENTICAL network
states, how much energy the sleep saved and what it cost in delay,
jitter and packet loss -- per QoS class, since a URLLC UE and an mMTC
UE care about very different things.

This is the ground truth CMOA has to learn to predict: "if I sleep cell
k now, what happens?" Measured, not modelled.

Three things it computes
------------------------
1. ESR = energy saved / baseline total energy, the deck's metric
   (slide 20: ESR = ΔE(A*) / Σ E_i^active x 100%).
2. Delay/jitter/loss deltas at MEAN and MEDIAN. Both matter and they
   disagree sharply here: sleeping a cell roughly doubles mean delay
   while barely moving the median, i.e. most UEs are fine and a
   minority suffer badly. A threshold check on the mean would miss
   exactly the UEs being harmed.
3. Per-QoS-class violation rates against two threshold sets:
     - DECK: the 3GPP-derived numbers in the contribution deck
       (uRLLC D<=1ms, eMBB D<=10ms, mMTC D<=100ms)
     - EMPIRICAL: the baseline's own 95th percentile per class
   The deck thresholds are expected to be unreachable in this
   simulation -- baseline median delay is ~22ms and the minimum
   observed anywhere is 9.8ms, so uRLLC's 1ms and eMBB's 10ms are
   violated in essentially every snapshot even with all cells awake.
   CMOA calibrated on them would reject every action, which is not a
   useful decision model. The empirical set is offered as the
   alternative to discuss with the supervisor: it asks "does this
   action make things materially worse than normal operation?" rather
   than "does this network meet 3GPP targets?", which it never does.
"""

import os

import numpy as np
import pandas as pd

SEED = int(os.environ.get("SEED", "22"))

PAIRS = [
    (37, [0, 1, 2, 3]),
    (43, [0, 1, 2, 3]),
    (46, [0, 1, 2, 3]),
]

DECK_THRESHOLDS = {          # delay_ms, jitter_ms, packet_loss
    "URLLC": (1.0, 0.5, 0.00001),
    "eMBB": (10.0, 5.0, 0.001),
    "mMTC": (100.0, 50.0, 0.01),
}


def load(tag):
    d = f"results/extracted_sleep_s{SEED}_{tag}"
    k = pd.read_csv(f"{d}/kpi_targets.csv")
    u = pd.read_csv(f"{d}/ue_inputs.csv")[["snapshot_id", "ue_index", "qos_class"]]
    e = pd.read_csv(f"{d}/energy_targets.csv")
    return k.merge(u, on=["snapshot_id", "ue_index"]), e


def violation_rate(df, thr):
    """Fraction of (UE, snapshot) rows breaching that class's limits."""
    out = {}
    for cls, (d, j, p) in thr.items():
        sub = df[df.qos_class == cls]
        if sub.empty:
            out[cls] = float("nan")
            continue
        bad = (sub.delay_ms > d) | (sub.jitter_ms > j) | (sub.packet_loss > p)
        out[cls] = bad.mean() * 100
    return out


print(f"seed {SEED}: energy/QoS trade-off per sleep action\n")

rows = []
for power, cells in PAIRS:
    base_k, base_e = load(f"p{power}")
    base_energy = base_e.estimated_gnb_power_w.sum()

    # empirical thresholds = baseline 95th percentile per class
    emp = {}
    for cls in DECK_THRESHOLDS:
        sub = base_k[base_k.qos_class == cls]
        emp[cls] = (sub.delay_ms.quantile(0.95),
                    sub.jitter_ms.quantile(0.95),
                    max(sub.packet_loss.quantile(0.95), 1e-6))

    print(f"{'='*76}\np{power}  baseline: mean delay {base_k.delay_ms.mean():.1f}ms  "
          f"median {base_k.delay_ms.median():.1f}ms  energy {base_energy/1e3:.1f} kW-snapshots")
    print(f"  empirical thresholds (baseline p95): " +
          "  ".join(f"{c} D<={emp[c][0]:.0f}ms" for c in emp))
    print(f"  deck violation rate at BASELINE: " +
          "  ".join(f"{c} {v:.1f}%" for c, v in violation_rate(base_k, DECK_THRESHOLDS).items()))
    print()

    for k in cells:
        sk, se = load(f"p{power}_sleep{k}")
        sleep_energy = se.estimated_gnb_power_w.sum()
        esr = (base_energy - sleep_energy) / base_energy * 100

        dv_deck = violation_rate(sk, DECK_THRESHOLDS)
        dv_emp = violation_rate(sk, emp)
        bv_emp = violation_rate(base_k, emp)

        print(f"  sleep gnb{k}:  ESR {esr:5.2f}%   "
              f"delay mean {base_k.delay_ms.mean():.0f}->{sk.delay_ms.mean():.0f}ms "
              f"(x{sk.delay_ms.mean()/base_k.delay_ms.mean():.2f})   "
              f"median {base_k.delay_ms.median():.1f}->{sk.delay_ms.median():.1f}ms")
        print(f"    p95 delay {base_k.delay_ms.quantile(.95):.0f}->{sk.delay_ms.quantile(.95):.0f}ms   "
              f"jitter mean {base_k.jitter_ms.mean():.0f}->{sk.jitter_ms.mean():.0f}ms   "
              f"loss mean {base_k.packet_loss.mean()*100:.2f}->{sk.packet_loss.mean()*100:.2f}%")
        print(f"    violations vs DECK thresholds:      " +
              "  ".join(f"{c} {dv_deck[c]:5.1f}%" for c in dv_deck))
        print(f"    violations vs EMPIRICAL (base p95): " +
              "  ".join(f"{c} {bv_emp[c]:.0f}%->{dv_emp[c]:.0f}%" for c in dv_emp))
        print()

        rows.append({"power": power, "sleep_gnb": k, "ESR_%": round(esr, 2),
                     "delay_mean_base": round(base_k.delay_ms.mean(), 1),
                     "delay_mean_sleep": round(sk.delay_ms.mean(), 1),
                     "delay_med_base": round(base_k.delay_ms.median(), 1),
                     "delay_med_sleep": round(sk.delay_ms.median(), 1),
                     "delay_p95_base": round(base_k.delay_ms.quantile(.95), 1),
                     "delay_p95_sleep": round(sk.delay_ms.quantile(.95), 1),
                     "jitter_mean_base": round(base_k.jitter_ms.mean(), 1),
                     "jitter_mean_sleep": round(sk.jitter_ms.mean(), 1),
                     "loss_mean_base": round(base_k.packet_loss.mean(), 5),
                     "loss_mean_sleep": round(sk.packet_loss.mean(), 5)})

df = pd.DataFrame(rows)
df.to_csv(f"sleep_tradeoff_s{SEED}.csv", index=False)
print("=" * 76)
print(df.to_string(index=False))
print(f"\nwritten to sleep_tradeoff_s{SEED}.csv")
print()
print("Reading this: ESR is what the deck's energy-saving metric measures.")
print("The median barely moving while the mean doubles is the key fact --")
print("most UEs are unaffected, a minority are badly hurt. A CMOA safety")
print("check on mean delay would approve actions that break specific UEs;")
print("p95 or per-class violation rate is the honest test.")
