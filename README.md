# GEODE — Graph-based Energy and quality-Of-service Digital-twin Engine

Implementation accompanying the Master's thesis

> M. AlKazem, *Graph-Based Digital Twin for Energy and Performance Modeling in
> 6G Networks*, Research Master in Information Systems and Data Intelligence,
> Lebanese University – Faculty of Science, in collaboration with Université de
> Lorraine, 2026. Supervisors: Dr. Hassan Harb, Pr. Moufida Maimour.

GEODE is a graph-based network digital twin for a 6G radio access network. It

1. represents each network snapshot as a heterogeneous graph of base stations
   and user equipment (**C1**);
2. encodes it with a recurrent heterogeneous message-passing GNN (**C2**);
3. jointly predicts per-base-station energy and per-device delay, jitter and
   packet loss with mixture-of-experts heads (**C3**);
4. screens candidate base-station sleep actions on a counterfactual graph and
   rejects those predicted to degrade QoS (**C4**, the CMOA loop);
5. monitors its own error in operation and fine-tunes when it drifts (**C5**).

Every result is compared against a **graph-blind control**: the same model with
message passing removed.

---

## Repository layout

```
GEODE-Implementation/
├── geode/                 primary study — Simu5G simulated 6G RAN (scenario06c)
│   └── results/           CSV / JSON / Markdown outputs behind the Chapter 5 tables
├── geode_5g3e/            the same design adapted to the real 5G3E testbed
│   └── results/
├── preprocessing/         5G3E raw data → graph tensors
├── simulation/            Simu5G scenario configurations, run scripts and extraction scripts
├── requirements.txt
└── README.md
```

### Architecture (`geode/`)

| file | component | what it does |
|---|---|---|
| `c1_graph.py` | C1 | builds the heterogeneous graph: base-station and UE nodes; serving, interference, aggregation (and optional handover) relations |
| `c2_hmpgnn.py` | C2 | recurrent encoder: `SAGEConv` on mean-aggregated relations, `GATv2Conv` on interference, `GRUCell` update, `LayerNorm`, `HeteroConv` sum; T = 4 rounds, d = 64 |
| `c3_heads.py` | C3 | four MoE heads (4 experts, top-2 gating); two-stage packet-loss head (occurrence + severity) |
| `c4_cmoa.py` | C4 | counterfactual operator Ω (nearest-cell or oracle reassignment); safety gate on the predicted 95th-percentile delay ratio (ρ < 1.5) and energy gate (ΔÊ > 0) |
| `c5_lifecycle.py` | C5 | rolling delay-MAPE trigger (W = 20, η = 25 %, N_min = 30), chronological 70/30 split, head-only fine-tuning with frozen encoder |
| `wdt_model.py` | — | full GEODE model and graph-blind control |
| `wdt_data.py` | — | loads extracted runs, builds and caches graphs, phase-aligned splits, train-only normalisation |
| `wdt_checkpoint.py` | — | loads a trained checkpoint with its normalisers |

### Experiments (`geode/`)

| script | purpose | thesis |
|---|---|---|
| `train_wdt_s06c.py` | leave-one-topology-out training of GEODE and the control (3 topologies × 3 initialisations) | Tables 5.4, 5.5, 5.7–5.13 |
| `train_singletask_s06c.py` | separate delay-only / jitter-only models | §5.4.1 |
| `baseline_classical.py` | classical (non-graph) baselines | §5.3 |
| `evaluate_c4_compare.py` | leak-free C4 evaluation: each checkpoint scored only on its held-out topology | Tables 5.10, 5.18, 5.19 |
| `c4_calibrate.py` | safety-threshold calibration; dedicated counterfactual predictor | Tables 5.17, 5.20 |
| `evaluate_c4.py`, `ensemble_eval.py` | earlier single-checkpoint C4 evaluation and ensembling | §5.7.3 |
| `c5_lifecycle.py`, `c5_drift_test.py` | same-dataset replay and distribution-shift replays (scenario06b, scenario07) | Tables 5.21–5.24, Fig. 5.5 |
| `audit_s06c.py` | dataset audits: label repeat rates, phase alignment | §4.6, §5.4.1 |
| `make_thesis_figures.py` | Chapter 5 figures | Figs. 5.1–5.5 |

### 5G3E (`geode_5g3e/`)

| file | purpose | thesis |
|---|---|---|
| `train_wdt_5g3e.py` | training on Day 1 + Day 2, test on held-out Day 3; architecture sweep | Tables 5.1, 5.3 |
| `cmoa_5g3e.py` | verification loop on 5G3E (round-robin same-site reassignment) | Table 5.2 |
| `lifecycle_5g3e.py` | lifecycle with injected drift | §5.2.4 |
| `baseline_classical.py` | Random Forest with / without lag features | §5.2.2 |
| `c2_hmpgnn.py`, `c3_heads.py`, `wdt_model.py`, `wdt_data.py` | 5G3E variant of the model (site-level heads, measured backhaul relation, T = 10) | §4.3.3 |

### Simulation and extraction (`simulation/`)

One folder per scenario (`scenario06b`, `scenario06c`, `scenario07`,
`scenario08`), each placed under `simu5g-1.4.4/simulations/nr/` when run.

| file pattern | purpose |
|---|---|
| `gen_ini*.py` | generate the per-seed `omnetpp_s<seed>.ini` (topology, per-device mobility class and start positions, traffic schedule, power, sleep configs) |
| `omnetpp_s<seed>.ini`, `demo.xml`, `ue_assignment_s<seed>.txt` | generated scenario configuration used for the reported runs |
| `run_*.sh`, `sweep_*.sh` | run the baselines (3 powers) and single-cell sleep configurations; set `SIMU5G_WS` to your workspace (default `~/simu5g-workspace`) |
| `extract_scenario*.py` | network inputs and energy (base-station / UE features, serving edges, EARTH energy) |
| `extract_kpi_*.py` | per-device delay, jitter and packet loss per 3-second phase (`extract_kpi_1s_*` for the one-second variant, Table 5.14) |
| `extract_sinr_*.py`, `merge_sinr_into_ue_inputs.py` | uplink SINR extraction and merge into the UE features |
| `repair_kpi_grid*.py`, `trim_*.py`, `fix_positions.py` | label-grid repair, settling-window trimming, position reconstruction |
| `extract_sleep_pairs.py`, `verify_sleep.py`, `compare_pairs.py` | paired baseline/sleep runs for the verification-loop ground truth |
| `scenario08/` | the eight-cell scenario that could not be completed (§4.5.4), kept for resumption |

Extraction runs on the SQLite exports produced by `opp_scavetool` and filters
every statistic by both name and module path (§4.6.1). Scripts need Python ≥ 3.12.

### Results

`geode/results/` holds the outputs the thesis tables were read from:

- `checkpoint_metrics/results_*.json` — per-topology, per-initialisation metrics
  for the adopted model (`results_adopted_nolag.json`, `results_final.json`),
  the full specification (`results_nolag_recover.json`), each ablation
  (`results_no_*.json`, `results_rounds4.json`) and diagnostic variants
  (`results_starved.json`, `results_*lag*.json`, `results_stride_check.json`,
  `results_kpi1s_*.json`).
- `c4_compare_*.csv` and `C4_COMPARE_RESULTS.md` — leak-free verification-loop
  results.
- `c4_results.csv`, `c4_single.csv`, `c4_ens.csv` — the **earlier** evaluation
  that scored one checkpoint against all three topologies; kept only for the
  correction reported in §5.7.3 and superseded by `c4_compare_*`.
- `c5_*.csv` — lifecycle replays (`c5_drift_scenario07_autotrigger.csv` is the
  scenario07 replay of Tables 5.23–5.24).

---

## Datasets

The extracted datasets are **not stored in this repository** because of their
size; the trained models are (see *Trained models* below).

### 1. Simulated 6G RAN — Simu5G (primary dataset)

Generated by the author with **OMNeT++ 6.4.0**, **INET 4.6.0** and the
**Simu5G 1.4.4** 5G NR library, following the 3GPP TR 38.901 Urban Macro
reference deployment.

| dataset | configuration | runs | role |
|---|---|---|---|
| scenario06c | 6 base stations, 60 devices, 500 m spacing, 25 RB, seeds 61–63 × 37/43/46 dBm, plus 16 paired single-cell sleep runs | 25 | primary dataset (training, C4 ground truth) |
| scenario06b | 6 base stations, 60 devices, 1000 m spacing, seeds 51–53 | 9 | distribution-shift replay (C5) |
| scenario07 | 4 base stations, 30 devices, 1000 m spacing, 50 RB | 20 + 39 sleep pairs | distribution-shift replay (C5), earlier reference results |

Each run is exported to SQLite with `opp_scavetool` and extracted into
per-snapshot CSVs (100 ms snapshots; QoS measured per 3-second traffic phase).
Base-station energy follows the EARTH linear load model,

P = N_tx · (P_idle + γ · P_tx · β),  β = RB_used / RB_max

with N_tx = 2 transceiver chains, P_idle = 130 W per chain, γ = 4.7, P_tx the
transmit power in watts and β the fraction of uplink resource blocks in use per
100 ms snapshot (RB_max = 25 in scenario06c, 50 in scenario06b/07). An idle cell
draws 260 W and a fully loaded one 307 / 448 / 634 W at 37 / 43 / 46 dBm. A
reduced sleep draw (150 W) is applied only to unloaded cells in the first 0.5 s
of a run; afterwards a slept cell keeps the 260 W idle floor, so a sleep action
saves only the load-proportional term.

**Download.** The extracted datasets are attached to the
[`datasets-v1` release](https://github.com/MalakAlKazem/GEODE-Implementation/releases/tag/datasets-v1):

| archive | contents | size |
|---|---|---|
| `scenario06c.zip` | primary dataset: 9 baseline runs and 16 paired sleep runs | 102 MB |
| `scenario07.zip` | scenario07 runs, sleep pairs, and the scenario06b runs (`extracted_s06b*`) used for the distribution-shift replay | 122 MB |

Unzip both into `data/` so that the folders are `data/scenario06c/` and
`data/scenario07/`. Each `extracted_*` folder holds one run as per-snapshot
CSVs: `gnb_inputs.csv`, `ue_inputs.csv`, `serving_edges.csv`,
`energy_targets.csv`, `kpi_targets.csv` (plus `kpi_targets_1s.csv` and
`sinr_features.csv` where available).

### 2. Real 5G testbed — 5G3E

The real-data study uses the public **5G3E** dataset, collected on the 5G
testbed of the **CEDRIC laboratory, Conservatoire National des Arts et Métiers
(CNAM), Paris**. It contains time-series radio, compute and network metrics
from a multi-site srsRAN deployment driven by real operator traffic traces.

- Dataset: <https://github.com/cedric-cnam/5G3E-dataset>
- Reference:
  D. C. Phung, N.-E.-H. Yellas, S. Bin Ruba and S. Secci, "An Open Dataset for
  Beyond-5G Data-driven Network Automation Experiments," in *Proc. 1st
  International Conference on 6G Networking (6GNet)*, Paris, France, 2022.

This work uses three days of the dataset (Day 1 and Day 2 for training, Day 3
held out for testing). `preprocessing/preprocess_5g3e.py` converts the raw
files into the per-snapshot graphs, the train/val/test split (`split.pt`,
last 15 % of the training days as validation) and the train-only normalisers
(`norm_stats.pt`) used by `geode_5g3e/`:

- delay: measured per-site latency;
- jitter: short-window variation of the latency;
- energy: derived from server CPU utilisation, E = 130 + 470 · CPU (W);
- packet loss: block error rate (constant at zero in the used days, so its
  head is trained but not reported).

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.5.1 + CUDA 12.1 and PyTorch Geometric on a
single NVIDIA RTX 4050 (6 GB).

Place the data as below, or point the environment variables at it:

| data | default location | environment variable |
|---|---|---|
| scenario06c extracted runs | `data/scenario06c/` | `GEODE_S06C` |
| scenario07 extracted runs | `data/scenario07/` | `GEODE_S07` |
| 5G3E raw data (`day_1/`, `day_2/`, `day_3/`) | `data/5g3e_raw/` | `GEODE_5G3E_RAW` |
| 5G3E processed tensors | `data/5g3e_processed/` | `GEODE_5G3E` |

## Trained models

All trained models behind the reported results are included, so the
evaluations can be rerun without retraining.

- `geode/checkpoints/` — 114 primary-dataset models, named
  `wdt_<tag>_<gnn|blind>_test<seed>_init<k>.pt`: `gnn` is GEODE, `blind` the
  graph-blind control, `test<seed>` the held-out topology (61, 62, 63) and
  `init<k>` the initialisation.

  | tag | configuration | thesis |
  |---|---|---|
  | `adopted_nolag`, `final` | adopted architecture (4 rounds, handover dropped) | Tables 5.4, 5.5, 5.18 |
  | `nolag_recover`, `full` | full specification (10 rounds, handover kept) | Tables 5.9, 5.10, 5.12 |
  | `no_e2`, `no_e3`, `no_e4`, `no_attn`, `no_gru`, `no_moe`, `rounds4` | single-component ablations | Table 5.8 |
  | `starved` | load features removed | §5.4.3 |
  | `adopted_energy_lag`, `energy_lag_only` | clean energy lag feature (adopted / full) | Table 5.7 |
  | `deep_energy` | full specification with a deeper energy head (exploratory) | — |
  | `stride15`, `stride_check` | snapshot sampling rate | Table 5.13 |
  | `kpi1s_check`, `kpi1s_log` | one-second QoS labels | Table 5.14 |

- `geode_5g3e/checkpoints_5g3e/` — 19 5G3E models, named
  `wdt5g3e_<tag>_<gnn|blind>_init<k>.pt`, trained on Day 1 + Day 2:
  `full` (plain), `full_lag` (lag features), `full_delta` (lag + delta jitter
  target; Table 5.1, used by `cmoa_5g3e.py` and `lifecycle_5g3e.py`) and
  `combined_day3test`. The architecture-sweep models behind Table 5.3 were not
  kept; `train_wdt_5g3e.py` retrains them.

Primary-dataset checkpoints store their normalisers and are loaded with
`geode/wdt_checkpoint.py`. 5G3E checkpoints use the normalisers saved with the
processed data (`norm_stats.pt`) and are loaded with
`load_model_from_checkpoint` in `geode_5g3e/cmoa_5g3e.py`.

## Usage

```bash
# 5G3E: preprocess, then train and evaluate
python preprocessing/preprocess_5g3e.py --days 1 2 --test_days 3
cd geode_5g3e
python train_wdt_5g3e.py --help
python cmoa_5g3e.py
python lifecycle_5g3e.py

# Primary dataset
cd ../geode
python train_wdt_s06c.py --help        # AdamW, lr 1e-3, wd 1e-4, batch 32, ≤200 epochs, patience 20
python evaluate_c4_compare.py          # verification loop, leak-free
python c4_calibrate.py                 # threshold calibration
python c5_drift_test.py --help         # lifecycle replays
python make_thesis_figures.py          # figures → ./figures
```

## Evaluation protocol (summary)

- **Leave-one-topology-out**: each of the three scenario06c topologies is held
  out in turn; three initialisations each, nine models per reported figure.
- **Phase-aligned splits**: split boundaries snap to 3-second traffic-phase
  edges so broadcast QoS labels never straddle train and test.
- **Train-only normalisation**: feature/label normalisers and packet-loss bin
  edges are fitted on training snapshots only.
- **C4 ground truth**: 640 labelled sleep opportunities from real paired
  simulator runs; an action is unsafe when its measured 95th-percentile delay
  ratio reaches 1.5.
