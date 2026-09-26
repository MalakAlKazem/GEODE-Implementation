# Does the graph help predict REAL cell-sleep effects? gnn vs blind on 06c's 640 real sleep-run measurements

`evaluate_c4_compare.py`. The ordinary held-out-snapshot comparison
(train_wdt_s06c.py's own R2 table) tests "predict another ordinary
snapshot" -- once a lag feature is available that's dominated by
inertia (this instant looks like the last one), not the regime C4
actually operates in: predicting the effect of an action (a cell
sleeping) the model has never seen play out, on NEIGHBOURING cells
whose inputs changed because of that action. `evaluate_c4.py` already
had exactly this test built, against 640 real paired sleep-run
measurements (not a synthetic counterfactual) -- it had just never
been run gnn-vs-blind head to head, or with any temporal feature. This
is that comparison, split into two leakage-clean gates (see
SWEEP_RESULTS.md's leakage correction: 06c's delay/jitter/PL labels
are broadcast per 3s phase, so a lag feature on those two targets
would be near-total label leakage, not signal -- energy is clean).

Checkpoints: `nolag_recover` (plain `full`, no lag anywhere -- used for
the safety gate) and `energy_lag_only` (gnb gets its own previous
energy_W only, ue gets nothing -- used for the energy gate, so there
is no leakage pathway anywhere in the graph, even indirectly through
message passing). Every checkpoint matched to its own held-out seed
only (leak-free); every lag value, in both the baseline and
counterfactual graph, comes from the baseline run's own real history,
never the sleep run's (matches this codebase's existing oracle vs
"nearest" deployable distinction -- a real decision-time system only
has real history from before the action).

## Safety gate: delay/jitter d95 ratio vs the real measured value

| | gnn | blind |
|---|---|---|
| d95 ratio correlation (pred vs true), nearest | **+0.246** | NaN |
| d95 ratio correlation (pred vs true), oracle | **+0.294** | NaN |
| real unsafe actions caught, nearest | **124/233 (53.2%)** | **0/233 (0.0%)** |
| real unsafe actions caught, oracle | **125/233 (53.6%)** | **0/233 (0.0%)** |

blind's `pred_d95_ratio` has so little variation across different
sleeping cells that its correlation with the real measured ratio is
degenerate (NaN) -- consistent with a model that cannot represent
"which cell went to sleep" at all: its safety flag never fires
correctly, on any of the 233 real unsafe actions in the test set.
gnn's predicted d95 ratio has a real, moderate positive correlation
with the true value and catches about half of the genuinely unsafe
actions. Neither number is good in absolute terms (a real deployed
system would need much better than 53% recall), but the GAP is
total: blind is not a viable substitute for this task at all, gnn is
a flawed-but-functioning one.

## Energy gate: real measured saving_w vs predicted

| | gnn | blind |
|---|---|---|
| saving correlation, nearest | -0.020 | **+0.023** |
| saving correlation, oracle | -0.021 | **+0.013** |
| MAE, nearest | 44.4 W | **34.4 W** |
| MAE, oracle | 44.2 W | **34.6 W** |

Both correlations are noise-level -- a known, pre-existing limit
(c4_cmoa.py's own docstring: twin energy MAPE ~8% on a ~1800W network
puts real per-action savings, median |true| 14.8W here, below the
error floor). But blind has lower error and a non-negative
correlation where gnn's is slightly negative -- corroborating
SWEEP_RESULTS.md's clean energy_lag_only finding (blind 0.925 vs gnn
0.878 R2 on ordinary snapshots) in a completely different evaluation
regime: genuinely real counterfactual sleep events, not held-out
ordinary snapshots.

## The finding

The graph's value on 06c is task-specific, not uniform. It is
essential for the safety/QoS decision -- propagating the effect of
one cell's action onto its neighbours is exactly what message passing
is for, and a blind model is structurally incapable of it, which
shows up starkly in real measured data (0% unsafe-action recall, not
just a smaller R2 gap). It adds nothing -- possibly a small negative
-- for predicting the sleeping cell's own energy change, where the
task reduces to "read off this node's own recent history," something
blind does at least as well without any message passing at all.

## Manuscript framing

This is a stronger, more specific claim than either "the graph helps"
or "the graph doesn't help" on 06c. State it as: **message passing is
necessary for the safety-critical counterfactual reasoning task C4
exists to do (catching unsafe actions via real network effects on
neighbouring cells), and not needed (or even mildly counterproductive
once fair temporal information is available) for single-node energy
regression.** The ordinary-snapshot R2 tables answer a different,
easier question (predict this node, right now) that undersells the
graph's actual value on the one task where it matters most for this
dataset, and oversells its value on the one task where it doesn't.

## Is the energy-gate correlation worse than the pre-existing c4_single.csv/c4_ens.csv (0.28/0.32)?

No -- checked directly, and the old number is very likely leaked, not
better. evaluate_c4.py's original `main()` loads ONE `--checkpoint`
and loops over `--seeds` (default all 3) using that SAME checkpoint
for every seed -- there is no per-seed checkpoint matching anywhere in
it. c4_single.csv's row counts (61:480, 62:320, 63:480, all 3 seeds
fully populated) are consistent with exactly that usage. Reproduced
it directly: ran the ORIGINAL evaluate_c4.py with one checkpoint
(test61's) against all 3 seeds, the naive/default way --

| | leaky (1 checkpoint x all 3 seeds) | leak-free (this file's method) |
|---|---|---|
| saving corr, nearest | 0.194 | -0.020 (gnn) / +0.023 (blind) |
| saving corr, oracle | 0.224 | -0.021 (gnn) / +0.013 (blind) |

0.19-0.22 lands right next to the old reported 0.28/0.32. test61's
checkpoint was trained WITHOUT seed 61 but WITH seeds 62 and 63 --
including their sleep-run data -- so evaluating it on seeds 62/63's
real sleep events (2/3 of the pooled 640 measurements) tests it on
data it had already partly seen. That inflates the correlation. The
old 0.28/0.32 headline should be read as optimistic/leaked, not as a
target this file's (leak-free) -0.02/+0.02 failed to reach.

## full vs adopted: which architecture is this evaluation actually run on?

Everything above uses `full` (all 4 relations including handover, 10
message-passing rounds) -- the raw spec-following config this codebase
defaults to. The manuscript reference document instead reports an
**adopted** architecture as final: 3 relations (handover dropped -- an
ablation found it contributes ~nothing), 4 rounds (a 6-gNB graph needs
far fewer than 10 to reach every node). Re-ran everything in this file
on `adopted` (`--no-e3 --rounds 4`, tags `adopted_nolag` /
`adopted_energy_lag`) to check which is actually better, rather than
assume.

Ordinary snapshot prediction -- adopted wins on every axis:

| | full | adopted |
|---|---|---|
| energy gap, no lag | +0.241 | **+0.290** |
| delay gap, no lag | +0.417 | **+0.462** |
| jitter gap, no lag | +0.415 | **+0.454** |
| energy R2, gnn+lag | 0.878 | **0.906** |
| energy gap, lag | -0.047 | **-0.019** (still loses to blind, but by less) |

Real sleep-event data -- NOT a clean sweep for adopted:

| | full | adopted |
|---|---|---|
| safety recall, nearest | **53.2%** | 47.6% |
| safety recall, oracle | **53.6%** | 51.5% |
| energy-gate MAE, nearest | 44.4W | **38.4W** |
| energy-gate corr, oracle | -0.021 | **+0.016** |

blind's numbers are IDENTICAL between the two architectures on every
single test (ordinary R2 and real-data C4 alike) -- not a coincidence:
blind never reads `edge_index`, so `--no-e3`/`--rounds` cannot affect
it at all, and the checkpoints are otherwise identical (same data,
same seed). A useful confirmation the blind-control methodology is
implemented correctly, independent of everything else in this file.

FINDING: `adopted` is the better predictor across the board and roughly
ties or edges `energy_lag_only`'s comparison on the energy gate, but
`full` keeps a real edge specifically on the safety gate against real
measured data (53% vs 48-52% recall) -- the one test that matters most
for this project's actual purpose. Plausible mechanism, not proven:
the counterfactual (a cell asleep) is more out-of-distribution than an
ordinary snapshot, and the extra depth/edge `adopted` correctly found
unnecessary for routine prediction may still help propagate that
unusual signal further. Do not report this as "adopted is simply
better" -- report it as a real trade-off: better predictor, possibly
less safe verification loop.

## Caveats

- Both gates' absolute numbers are modest (53% unsafe recall, ~0
  energy correlation for either model) -- this is a comparison of
  gnn vs blind, not a claim that either is a finished, deployable
  safety system.
- The safety gate's checkpoints carry no lag feature at all (see
  SWEEP_RESULTS.md's leakage correction) -- worth revisiting if a
  genuinely non-leaky temporal signal for delay/jitter is ever found
  on 06c (none is currently known, given this dataset's 3s KPI
  measurement resolution).
- `d95 corr +nan` for blind is reported as-is, not hidden or replaced
  with 0 -- it is itself part of the finding (no representable
  variation to correlate), not a computation failure to work around.
