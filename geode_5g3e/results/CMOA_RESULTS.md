# C4 -- CMOA-DT digital-twin decision loop, 5G3E v7

`cmoa_5g3e.py`, ported from `src/v3/cmoa_v3.py`'s C-M-O-A structure to the
v7 model (blind control, ablation-tested, lag+delta-aware). Two real
differences from v3's version:

1. Runs on the actual held-out Day 3 test set (6,973 snapshots), not a
   validation split of the same training day.
2. Packet loss is dropped from the safety check -- it's a measured
   constant (0.0) across the whole 5G3E dataset, so a PL<=1% constraint
   would always trivially pass and add nothing real to check.

Model used: `full_delta` (gnn, rounds=10, all edges, lag+delta on),
init 1 -- our best-performing config from the whole investigation.

## Counterfactual construction bug, found and fixed

The first full run below (kept in the table for the record) had a real
bug in `make_counterfactual()`, the function that builds the "what if
gNB X sleeps" graph fed to the model for the safety check. It zeroed
all 5 of the sleeping gNB's own node features (`nof_ue`, one-hot site
id, `mean_rsrp`) and left every other gNB's inputs untouched -- i.e. it
asked the model to score a physically incoherent state (a gNB that is
asleep, but whose UEs are simply left connected to nothing, and whose
neighbours see no change at all) rather than the state that would
actually exist post-sleep (UEs handed off to a neighbouring cell,
raising that neighbour's load and changing its mean RSRP). This is the
same class of bug already caught and fixed in 06c's C4 -- a
counterfactual has to match the *form* the model was trained on, not
just be "zeroed out."

Fix, in `make_counterfactual()`:
- the sleeping gNB's own 5 features are now left completely untouched
  (no zeroing at all -- they're irrelevant anyway, since C4 never reads
  gNB X's own prediction once it's asleep);
- any UE previously connected to the sleeping gNB is reassigned
  round-robin to an active gNB on the *same site* (falling back to any
  active gNB if none exists on-site), rewriting both directions of the
  `ue<->gnb` edge index;
- the receiving neighbour's `mean_rsrp` feature is recomputed from the
  (denormalised, then renormalised) RSRP of its now-larger UE set, so
  its inputs reflect the real post-handoff load;
- the sleeping gNB's backhaul edges are dropped, as before.

## Full run: all 6,973 Day 3 test snapshots

| | v3 (old model, Day1 val split, 1,068 snapshots) | v7, buggy counterfactual (genuine Day3 test) | **v7, fixed counterfactual (genuine Day3 test)** |
|---|---|---|---|
| Approval rate | 100% of candidates | 49.0% of candidates | **49.1%** of candidates |
| Mean ESR | 14.26% | 7.03% | **7.04%** |
| Cumulative dE | 1,018.88 kWh | 3,246.79 kWh | **3,252.07 kWh** |
| Drift | none detected | rolling MAPE ~0.13%, C5 never fired | rolling MAPE ~0.13%, C5 never fired |

## The finding

v3's 100% approval rate was almost certainly an artifact of testing on a
low-utilisation validation split of the *same* training day -- not a
real property of the safety mechanism. On genuinely unseen data with a
properly-validated model, the digital twin **rejects about half** of the
naive "sleep the highest-energy tower" candidates, because roughly half
the time doing so would push a neighbouring tower's delay or jitter over
the QoS limit. That conclusion holds under both the buggy and the fixed
counterfactual.

**The aggregate numbers barely moved after fixing the bug** (49.0% ->
49.1% approval, 7.03% -> 7.04% ESR, 3,246.79 -> 3,252.07 kWh) even
though the fix is real and does change individual predictions -- a
direct per-snapshot diff of old vs. new counterfactual QoS outputs
confirms measurable shifts on affected neighbours (e.g. one gNB's
predicted J_max moved 4.40ms -> 3.96ms on a snapshot where it absorbed
stranded UEs). The reason the aggregate is stable is that on this
dataset most candidate actions are not decided near the QoS threshold
boundary -- they are comfortably safe or clearly unsafe regardless of
whether the neighbour's `mean_rsrp` is nudged by a fraction of a sigma
-- so a few flipped decisions near the margin wash out in 13,946 total
candidates. This is a genuine robustness finding, not a null result:
it means the safety gate's *aggregate* verdict on 5G3E was not an
artifact of the calibration bug, but the *per-decision* QoS margins it
reports for any single candidate are now trustworthy in a way they
were not before, since they reflect a physically coherent post-sleep
state rather than a zeroed-out one.

Sample approved action (snap 0, fixed run): sleep gNB #11, 523.81W ->
62.86W (460.95W saved), post-sleep D_max=157.9ms<=180ms,
J_max=4.4ms<=20ms.

Sample REJECTED action (snap 2, fixed run): candidate gNB #2 (511.26W)
blocked -- would push J to 21.3ms, over the 20ms limit. The safety
mechanism correctly blocking an unsafe action, not rubber-stamping
everything.

Energy savings are lower than v3's number (7.04% vs 14.26% mean ESR)
but honest, and substantial in absolute terms (3,252.07 kWh over the
test period). The digital twin's own predictions never drifted across
the entire held-out day -- rolling energy MAPE stayed at ~0.13%,
nowhere near the 20% drift threshold, so C5 (lifecycle fine-tuning) was
never triggered naturally (see `LIFECYCLE_RESULTS.md` for the injected-
drift demonstration of C5 itself).

## Manuscript framing

This is a stronger result than v3's, precisely because it's less
flattering: "the DT approves everything and saves 14% energy" invites a
reviewer to ask whether the safety check does anything at all. "The DT
makes real trade-offs, blocks about half of naive sleep candidates for
genuine safety reasons, and still delivers substantial honestly-measured
savings without ever losing prediction accuracy" is the defensible
version of the same system.

## Not yet done

- C5 (lifecycle fine-tuning / drift-recovery) is now built and validated
  for v7 -- see `LIFECYCLE_RESULTS.md`. Since no natural drift occurred
  on Day3, it's demonstrated via injected drift (v3's approach: shift
  labels by several sigma and show the freeze-encoder-update-heads
  recovery), not a natural trigger from this data.
- E_THRESH (480W) and the QoS thresholds (D_MAX=180ms, J_MAX=20ms) are
  inherited unchanged from v3's testbed-adapted values -- not re-tuned
  for the v7 model's prediction distribution. Worth a sensitivity check
  before treating 49.1%/7.04% as a fixed number rather than a threshold
  choice.
- K_CANDIDATES=2 always saturates (every snapshot generates exactly 2
  candidates, since E_THRESH sits below the training-set energy mean) --
  worth checking whether a higher threshold changes the qualitative
  picture.
