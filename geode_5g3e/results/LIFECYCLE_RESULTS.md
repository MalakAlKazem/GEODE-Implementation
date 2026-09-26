# C5 -- DT lifecycle fine-tuning, 5G3E v7

`lifecycle_5g3e.py`, ported from `src/v3/lifecycle_v3.py`. Freeze target
is simpler than v3's module-name list: `WirelessDT6G` already separates
the whole HMPGNN encoder (`model.encoder`) from the four prediction
heads (`model.heads`) as two top-level attributes.

Model used: `full_delta` (gnn, rounds=10, all edges, lag+delta on),
init 1 -- same checkpoint as `cmoa_5g3e.py`'s C4 run.

## Phase 1 -- natural drift check

None. Confirmed by `cmoa_5g3e.py`'s full Day3 run: rolling energy MAPE
stayed ~0.13% across all 6,973 snapshots. The DT trained on Day1+2
generalises accurately to a genuinely unseen day -- same conclusion
v3 reached on its own (weaker) Day2 test. C5's trigger mechanism can
only be demonstrated with injected drift, same as v3's own showcased
result.

## Phase 2 -- injected drift + recovery

Drift injected on the last 1,000 Day3 test snapshots: energy labels
shifted +4.5 sigma (+142.2W), delay labels shifted +2.0 sigma
(+65.7ms) -- same magnitudes v3 used. Lag features (own recent
history) are NOT shifted, so the injected drift is a genuine
distribution shift the model hasn't already adapted to via its
temporal input.

| | v3 (old model) | **v7 (new model)** |
|---|---|---|
| Rolling MAPE after injection | 21.7% | 21.9% |
| C5 trigger threshold | 20% | 20% (crossed at ~snapshot 32 of the drifted tail) |
| Encoder frozen | 126,016 params (72.4%) | 138,688 params (61.5%) |
| Heads trainable | 48,050 params (27.6%) | 86,840 params (38.5%) |
| Energy MAPE before FT | 21.74% | 21.93% |
| **Energy MAPE after FT** | 1.87% | **1.00%** |
| Recovery | ~19.9 points | **20.93 points** |
| Fine-tuning convergence | epoch 9 (early stop) | epoch 30 (hit max epochs, still improving -- no plateau) |

## The finding

The recovery mechanism works, and works cleanly: a drift injection that
pushes rolling energy MAPE to ~22% (comfortably past the 20% trigger)
is fully corrected by freezing the encoder and fine-tuning only the
prediction heads on the last 1,000 snapshots -- down to 1.00% MAPE,
better than v3's own 1.87% recovery. The fine-tuning curve never
plateaued within the 30-epoch budget (loss and MAPE were still
improving epoch-over-epoch at the end), unlike v3's which converged
and early-stopped at epoch 9 -- worth noting as a difference in
optimisation dynamics between the two architectures, not necessarily
a "better" or "worse" one on its own.

## Manuscript framing

This completes the C1-C5 picture for 5G3E with the new, properly
re-verified architecture: the prediction model generalises to a
genuinely unseen day without drifting, the digital-twin decision loop
(C4) makes real, defensible trade-offs rather than rubber-stamping
everything, and the lifecycle mechanism (C5) demonstrably recovers
accuracy when drift is present, even though none occurred naturally in
this real dataset. All three claims now rest on the genuinely held-out
Day3 split rather than the easier Day1/Day2 setup the original v3
results were built on.

## Not yet done / caveats

- Drift magnitude (+4.5 sigma / +2.0 sigma) is inherited unchanged from
  v3, not derived from any real observed drift on 5G3E -- it is a
  demonstration of the mechanism, not a measurement of how much the
  real network actually drifts.
- Fine-tuning is evaluated on the same drifted snapshots it trains on
  (matching v3's own framing) -- this shows the mechanism corrects the
  specific drift it's given, not that it generalises to a *different*
  kind of drift than the one injected.
- Only energy/delay losses are used during fine-tuning (jitter's loss
  term is zeroed out in the fine-tuning loss here, since with
  predict_delta the jitter head no longer has the same
  binary+bin-classification structure v3's ft_loss assumed) --
  jitter's recovery under drift was not measured.
