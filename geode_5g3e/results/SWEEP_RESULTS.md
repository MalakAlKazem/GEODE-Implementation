# 5G3E ablation sweep results

Mean +/- std over 3 inits (seeds 1/7/13), Day 2 held-out test, physical units.
Reference config is `full` (nothing removed). blind is edge-independent, so
for the four edge ablations (no_e1..no_e4) its numbers are identical to
`full`'s blind by construction -- only gnn moves.

## full (reference)

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.50% +/- 0.08% / 0.986 | 4.47% +/- 0.03% / 0.165 | R2 +0.821 |
| delay MAPE / R2  | 15.54% +/- 0.26% / 0.453 | 16.47% +/- 0.06% / 0.388 | R2 +0.065 |
| jitter w1std_acc | 70.9% +/- 2.2% | 69.5% +/- 0.7% | +1.4pp |
| packet_loss | N/A (constant 0.0 on this testbed) | | |

## no_e1 (drop fabricated round-robin serving edges) -- DONE

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.31% +/- 0.04% / 0.994 | 4.47% +/- 0.03% / 0.165 | R2 +0.829 |
| delay MAPE / R2  | 15.62% +/- 0.27% / 0.452 | 16.47% +/- 0.06% / 0.388 | R2 +0.064 |
| jitter w1std_acc | 71.8% +/- 0.4% | 69.5% +/- 0.7% | +2.3pp |
| packet_loss | N/A | | |

Finding: no_e1 matches or slightly beats `full` on every metric, AND has
much lower variance across seeds (energy std 0.08%->0.04%, jitter w1std
std 2.2%->0.4%). Consistent with E1 being fabricated (round-robin, not
measured) -- removing pure noise stabilises training, not just "doesn't
hurt." blind numbers are identical to full's blind by construction
(WirelessDT6GBlind never touches edge_index_dict).

## no_e2 (drop intra-site interference edges) -- DONE

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.57% +/- 0.18% / 0.984 | 4.47% +/- 0.03% / 0.165 | R2 +0.819 |
| delay MAPE / R2  | 15.37% +/- 0.21% / 0.443 | 16.47% +/- 0.06% / 0.388 | R2 +0.055 |
| jitter w1std_acc | 70.3% +/- 1.9% | 69.5% +/- 0.7% | -0.003 R2 gap (essentially zero) |
| packet_loss | N/A | | |

Finding: statistically indistinguishable from `full` and `no_e1` on
energy/delay -- interference edges add nothing measurable on 5G3E,
matching the codebase's prior note that E2 was "rejected twice on
scenario07" for weak inter-cell coupling. Now confirmed on real data
too. Higher energy variance (+-0.18% vs +-0.04-0.08%) traces to 2/3
inits plateauing at epoch 1 (scheduler didn't get to act), not
necessarily an E2-specific effect.

## no_e3 (drop cross-site "handover" edges) -- DONE

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.75% +/- 0.06% / 0.974 | 4.47% +/- 0.03% / 0.165 | R2 +0.809 |
| delay MAPE / R2  | 16.17% +/- 0.33% / 0.406 | 16.47% +/- 0.06% / 0.388 | R2 +0.018 |
| jitter w1std_acc | 70.4% +/- 0.4% | 69.5% +/- 0.7% | +0.005 R2 gap |
| packet_loss | N/A | | |

Finding: FIRST ablation with a real, consistent cost -- worst energy AND
worst delay MAPE of the four edge configs tested, delay's R2 gap cut
~3x (0.065 -> 0.018). All 3 inits plateaued at epoch 1 (vs the mix seen
elsewhere), another sign this config trains differently, not just
noisier. IMPORTANT CAVEAT: unlike 06c's E3 (built from real detected
per-UE serving-cell changes), 5G3E's E3 is a STATIC cross-site
partition -- real handover detection isn't possible here since E1
serving assignment is itself fabricated round-robin (no real PCI data).
So this result says "cross-site structural connectivity helps" (likely
by giving cross-site load info a more direct path than routing through
the single backhaul/server aggregation point), NOT "real handovers
help" -- do not overclaim the latter in the manuscript.

## no_e4 (drop backhaul edge -- only path to cpu_util) -- DONE -- HEADLINE RESULT

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 4.39% +/- 0.10% / 0.185 | 4.47% +/- 0.03% / 0.165 | **R2 +0.020 (was +0.81-0.83 everywhere else)** |
| delay MAPE / R2  | 17.02% +/- 0.53% / 0.386 | 16.47% +/- 0.06% / 0.388 | R2 -0.002 (also collapsed) |
| jitter w1std_acc | 70.6% +/- 0.2% | 69.5% +/- 0.7% | +0.006 R2 gap |
| packet_loss | N/A | | |

Finding: THE structural confirmation. Every other edge ablation (e1/e2/e3)
left the energy R2 gap at +0.81-0.83 -- removing e4 drops it to +0.020,
indistinguishable from zero, and MAPE jumps from 0.3-0.75% up to 4.39%,
landing exactly on blind's level (4.44-4.51% everywhere else). Confirms:
without the backhaul edge, gnn structurally BECOMES blind for energy,
because cpu_util has no other path to a gNB. All 3 inits independently
confirm this (energy R2 range 0.156-0.203 vs the usual 0.97-1.00).

Manuscript caution: 4.39% MAPE alone reads as "fine" (< 10% target) if
taken in isolation -- R2 is what exposes that the model stopped doing
anything structural. Don't let MAPE alone carry this row.

Delay's already-small graph advantage also vanishes here (gap ~0), and
this config has the worst delay variance of the whole sweep (+-0.53%
vs +-0.18-0.33% elsewhere) -- losing e4 appears to destabilise training
broadly, not just blind the model to energy specifically.

## no_moe (MoE heads -> plain MLP) -- DONE

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.54% +/- 0.08% / 0.983 | 4.50% +/- 0.04% / 0.158 | R2 +0.825 |
| delay MAPE / R2  | 15.33% +/- 0.13% / 0.455 | 16.70% +/- 0.11% / 0.386 | R2 +0.070 |
| jitter w1std_acc | 71.3% +/- 0.7% | 69.2% +/- 0.5% | +0.005 R2 gap |
| packet_loss | N/A | | |

Note: unlike the e1-e4 edge ablations, no_moe changes head architecture,
which blind ALSO uses -- so blind's numbers here are a genuinely
different run, not an exact repeat of full's blind.

Finding: matches or slightly beats `full` on every metric, delay
variance notably tighter (+-0.13% vs +-0.26%). THIRD independent
dataset (after scenario07, scenario06c) confirming the codebase's
locked "single-task beats joint MoE" finding -- c3_heads.py's docstring
explicitly said a repeat loss on a second dataset would strengthen that
claim; this is now a third, on real testbed data.

## no_gru (recurrent update -> plain overwrite) -- DONE

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.59% +/- 0.17% / 0.984 | 4.47% +/- 0.03% / 0.165 | R2 +0.819 |
| delay MAPE / R2  | 15.82% +/- 0.43% / 0.438 | 16.47% +/- 0.06% / 0.388 | R2 +0.050 |
| jitter w1std_acc | 71.2% +/- 0.4% | 69.5% +/- 0.7% | +0.001 R2 gap (near zero) |
| packet_loss | N/A | | |

Finding: real but modest degradation, NOT a collapse like no_e4. Energy
essentially untouched (gap +0.819 vs full's +0.821) -- GRU isn't what
carries energy's advantage, the backhaul edge is (see no_e4). Delay's
R2 gap shrank ~25% (0.065->0.050) and its variance nearly doubled
(+-0.26% -> +-0.43%, 2nd-worst in the sweep after no_e4's +-0.53%).
Jitter's already-thin gap went essentially to zero. Consistent with
c2_hmpgnn.py's own over-smoothing warning for 10 rounds without a GRU
gate on a small graph -- real, just not catastrophic here.

## no_attn (GATv2 attention -> plain SAGEConv mean on interference edges) -- DONE

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.65% +/- 0.17% / 0.979 | 4.47% +/- 0.03% / 0.165 | R2 +0.814 |
| delay MAPE / R2  | 15.99% +/- 0.42% / 0.433 | 16.47% +/- 0.06% / 0.388 | R2 +0.045 |
| jitter w1std_acc | 71.7% +/- 1.0% | 69.5% +/- 0.7% | +0.011 R2 gap |
| packet_loss | N/A | | |

Finding: lands in the same narrow band as no_e2 (edges removed entirely):
full D-gap 0.065 > no_e2 D-gap 0.055 ~= no_attn D-gap 0.045. Coherent
story -- the interference relation itself doesn't carry much signal on
5G3E, so whether you remove it (no_e2) or keep it but drop the
attention-weighting (no_attn), the effect is about the same small dip.
GATv2 attention isn't what made E2 useful, because E2 wasn't
particularly useful to begin with.

## rounds4 (T=10 -> T=4 message-passing rounds) -- DONE -- BEST CONFIG OF THE SWEEP

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.43% +/- 0.05% / 0.991 | 4.47% +/- 0.03% / 0.165 | R2 +0.826 |
| delay MAPE / R2  | 15.28% +/- 0.08% / 0.463 | 16.47% +/- 0.06% / 0.388 | **R2 +0.075 (best delay gap of the sweep)** |
| jitter w1std_acc | **71.9% +/- 0.1% (best of the sweep)** | 69.5% +/- 0.7% | +0.004 R2 gap |
| packet_loss | N/A | | |

Finding: NOT just "as good as full" -- best delay MAPE/R2 gap and best
jitter w1std of the ENTIRE sweep, near-best energy, and dramatically
tighter variance everywhere (delay +-0.08% vs full's +-0.26%; jitter
+-0.1% vs full's +-2.2%, ~20x tighter). Directly confirms
c2_hmpgnn.py's own docstring suspicion: "10 rounds is far more than
needed to reach every node" on a 13-gNB graph, and "the spec's 10 is a
starting point... not a value validated here." T=4 is a genuine
improvement, not a compromise.

===============================================================================
FULL SWEEP SUMMARY (9/9 configs done)
===============================================================================

| Config   | Energy MAPE / R2 gap      | Delay MAPE / R2 gap        | Jitter w1std   |
|----------|---------------------------|-----------------------------|----------------|
| full     | 0.50%+/-0.08% / +0.821    | 15.54%+/-0.26% / +0.065     | 70.9%+/-2.2%   |
| no_e1    | 0.31%+/-0.04% / +0.829    | 15.62%+/-0.27% / +0.064     | 71.8%+/-0.4%   |
| no_e2    | 0.57%+/-0.18% / +0.819    | 15.37%+/-0.21% / +0.055     | 70.3%+/-1.9%   |
| no_e3    | 0.75%+/-0.06% / +0.809    | 16.17%+/-0.33% / +0.018     | 70.4%+/-0.4%   |
| no_e4    | 4.39%+/-0.10% / +0.020    | 17.02%+/-0.53% / -0.002     | 70.6%+/-0.2%   |
| no_moe   | 0.54%+/-0.08% / +0.825    | 15.33%+/-0.13% / +0.070     | 71.3%+/-0.7%   |
| no_gru   | 0.59%+/-0.17% / +0.819    | 15.82%+/-0.43% / +0.050     | 71.2%+/-0.4%   |
| no_attn  | 0.65%+/-0.17% / +0.814    | 15.99%+/-0.42% / +0.045     | 71.7%+/-1.0%   |
| rounds4  | 0.43%+/-0.05% / +0.826    | 15.28%+/-0.08% / +0.075     | 71.9%+/-0.1%   |

TAKEAWAYS:
1. no_e4 (backhaul edge) is THE essential component -- removing it
   collapses gnn to blind's level on energy (R2 gap +0.020 vs +0.81-0.83
   everywhere else). Every other ablation leaves this gap intact.
2. no_e1 (fabricated round-robin serving edges) and rounds4 (T=10->4)
   both BEAT full outright, on every metric, with much tighter variance.
   Neither is a "safe to drop" result -- both are genuine improvements.
3. no_e2 and no_attn are neutral-to-slightly-negative: the interference
   relation doesn't carry much signal on 5G3E regardless of whether it's
   attention-weighted or plain-averaged.
4. no_moe is neutral-to-slightly-positive -- third independent dataset
   (after scenario07, scenario06c) confirming MoE routing doesn't beat
   a plain MLP head.
5. no_e3 and no_gru are the two real (modest, not catastrophic) costs --
   both erode delay's stability and R2 gap without touching energy much.
   no_e3's caveat: 5G3E's E3 is a static cross-site partition, not real
   detected handovers (E1 serving assignment is fabricated round-robin,
   so genuine handover detection isn't possible here).

NEXT STEP SUGGESTED: combine no_e1 + rounds4 into one leaner config
(drop fabricated E1, T=4) and check whether the improvements compound.

===============================================================================
RE-VERIFICATION ROUND 2: remaining 8 sweep configs on Day3, in progress
===============================================================================
no_e1 (Day3, 3 inits): energy MAPE 0.52%+/-0.10% R2 0.987 | delay MAPE
18.79%+/-0.19% R2 0.278 gap -0.018 (NEGATIVE). Confirms the reversal:
no_e1 was an improvement on the old split, is now clearly WORSE than
`full` on Day3 (full: energy 0.36%/R2 0.993, delay 17.64%/R2 0.356 gap
+0.060).

no_e2 (Day3, 3 inits): energy MAPE 0.32%+/-0.08% R2 0.995 gap +0.699 |
delay MAPE 18.56%+/-0.28% R2 0.324 gap +0.028. Smaller, more modest
degradation than no_e1 -- roughly matches the old split's qualitative
finding (E2 doesn't matter much), though delay is measurably softer
than `full` here too.

no_e3 (Day3, 3 inits): energy MAPE 0.35%+/-0.02% R2 0.994 gap +0.702 |
delay MAPE 17.57%+/-0.14% R2 0.366 gap +0.066 | jitter w1std
77.7%+/-0.8%. ANOTHER FULL REVERSAL: on the old split no_e3 was the
worst config in the entire sweep (delay MAPE 16.17%, gap crushed to
+0.018). On Day3 it TIES or slightly beats `full` on every metric.
The handover edges (recall: a static cross-site partition on 5G3E, not
real detected handovers -- see earlier caveat) don't matter at all on
the genuine generalisation test.

no_e4 (Day3, 3 inits): energy MAPE 4.12%+/-0.01% R2 0.247 gap -0.045
(was 0.36%/+0.698 for full) | delay MAPE 17.21%+/-0.15% R2 0.374 gap
+0.075 (best delay gap of this re-verification round) | jitter w1std
78.0%+/-0.2%. HOLDS -- this is the one finding that survives untouched
on both splits, as expected since it's structural: the backhaul edge
is the only path to cpu_util regardless of which day is held out.
Side note: delay got slightly BETTER without E4, possibly because that
edge carries nothing delay-relevant and removing it reduces noise in
the shared representation.

no_moe (Day3, 3 inits): energy MAPE 0.44%+/-0.02% R2 0.991 gap +0.700 |
delay MAPE 18.11%+/-0.69% R2 0.313 gap +0.012 | jitter w1std
78.0%+/-0.3%. PARTIAL REVERSAL: direction still holds (MoE doesn't
help, gaps stay positive), but old split found no_moe MORE stable than
full -- here delay variance is 7x WORSE (+/-0.69% vs full's +/-0.09%).
Correct "no_moe is more stable" in the manuscript; "MoE doesn't help"
still stands.

RUNNING TALLY so far: no_e1 reversed (was better, now worse), no_e2
held (mattered little both times), no_e3 reversed (was worst, now
ties full), no_e4 HELD (structural, confirmed on both splits), no_moe
partially reversed (direction held, stability claim didn't). no_gru (Day3, 3 inits): energy MAPE 0.31%+/-0.06% R2 0.995 gap +0.703 |
delay MAPE 18.18%+/-0.07% R2 0.337 gap +0.038 | jitter w1std
78.1%+/-0.3%. HELD -- same qualitative story as the old split (real but
modest cost, concentrated in delay, energy barely touched): old split
delay gap +0.050, here +0.038. Only ablation so far whose conclusion
(direction AND rough magnitude) survived cleanly.

no_attn (Day3, 3 inits): energy MAPE 0.39%+/-0.10% R2 0.992 gap +0.700 |
delay MAPE 18.25%+/-0.50% R2 0.322 gap +0.022 (less than half of full's
+0.060, 5x more variance) | jitter w1std 78.5%+/-0.4%. Direction
roughly held (attention was already neutral-to-slightly-negative on
the old split, in the same cluster as no_e2), just more clearly
negative and noisier here.

rounds4 (Day3, 3 inits): energy MAPE 0.28%+/-0.04% R2 0.996 gap +0.704 |
delay MAPE 17.74%+/-0.38% R2 0.351 gap +0.051 | jitter w1std 78.1%+/-0.5%.
Ties/marginally edges `full`, NOT a wide win like the old sweep found --
old split called rounds4 "best of the entire study"; here it's
comparable to full, with somewhat more delay variance.

===============================================================================
RE-VERIFICATION ROUND 2 -- COMPLETE. Final table, all 9 original configs
on Day3 (train Day1+2, test genuinely unseen Day3):
===============================================================================

| Config   | Energy MAPE / R2 gap  | Delay MAPE / R2 gap      | Jitter w1std |
|----------|-----------------------|--------------------------|--------------|
| full     | 0.36% / +0.698        | 17.64% / +0.060          | 77.9%        |
| no_e1    | 0.52% / +0.692        | 18.79% / -0.018 (WORST)  | 78.2%        |
| no_e2    | 0.32% / +0.699        | 18.56% / +0.028          | 78.2%        |
| no_e3    | 0.35% / +0.702        | 17.57% / +0.066 (BEST)   | 77.7%        |
| no_e4    | 4.12% / -0.045 (CRASH)| 17.21% / +0.075          | 78.0%        |
| no_moe   | 0.44% / +0.700        | 18.11%+/-0.69% / +0.012  | 78.0%        |
| no_gru   | 0.31% / +0.703        | 18.18% / +0.038          | 78.1%        |
| no_attn  | 0.39% / +0.700        | 18.25%+/-0.50% / +0.022  | 78.5%        |
| rounds4  | 0.28% / +0.704        | 17.74%+/-0.38% / +0.051  | 78.1%        |

WHAT SURVIVED RE-VERIFICATION, PER CONFIG:
  no_e1   REVERSED  (was an improvement on old split, now the single
                     worst config -- delay gap goes negative)
  no_e2   HELD      (mattered little on both splits)
  no_e3   REVERSED  (was the worst config on old split, now ties/edges
                     out `full` -- arguably the best config on Day3)
  no_e4   HELD      (structural -- collapses energy on both splits,
                     this is the one finding that never needed
                     re-verifying in the first place)
  no_moe  PARTIAL   (direction held -- MoE doesn't help -- but the
                     "more stable" claim from the old split did not:
                     7x worse delay variance here)
  no_gru  HELD      (same modest cost, concentrated in delay, on both
                     splits)
  no_attn HELD-ish  (was already neutral-to-slightly-negative on old
                     split; more clearly negative and noisier here,
                     same direction)
  rounds4 WEAKENED  (was "best of the entire study" on old split by a
                     wide margin; here it only ties/marginally edges
                     `full`, not a dramatic win)

MANUSCRIPT-READY CONCLUSION:
  - E4 (backhaul edge) is the only architectural finding that is
    unconditionally true regardless of evaluation methodology.
  - E1 (fabricated serving edges) should be KEPT, reversing the earlier
    recommendation to drop it -- the easy test made real signal look
    like noise.
  - E3 (handover/cross-site edges, themselves a static partition, not
    real detected handovers) and reduced rounds (T=4) are both mild,
    real improvements over the full spec-following config -- worth
    reporting as a lean recommended variant, but NOT with the old
    sweep's dramatic framing ("best of the entire study").
  - E2, MoE, GRU, attention: minor, consistent, modest costs to remove
    -- keep them, but they are not major levers either way.
  - This entire re-verification exercise is itself a manuscript point:
    conclusions from a same-adjacent-day test do not reliably transfer
    to a genuinely held-out day, and 4 of 8 individual-ablation
    conclusions (no_e1, no_e3, no_moe's stability claim, rounds4's
    magnitude) changed meaningfully once tested properly.

===============================================================================
DELTA-PREDICTION: predict the CHANGE from lag, not the absolute value
===============================================================================
Motivated directly by the lag finding: if most of the answer is already
"close to what it was a moment ago", the head only needs to learn the
smaller residual on top, not the full absolute value. Implemented via
c3_heads.py's predict_delta flag (energy/delay/jitter heads all predict
delta; jitter switches from BinnedHead to a plain regression head,
since a fixed bin range sized for absolute values has no natural
meaning for a delta target). wdt_data.py stores lag_y_norm explicitly
on the graph; train_wdt_5g3e.py reconstructs the absolute prediction
(lag + predicted delta) before scoring.

`full` config + lag + delta, Day3 test, 3 inits confirmed:

| target | gnn+lag+delta | blind+lag+delta | RF+lag | gap |
|---|---|---|---|---|
| energy MAPE/R2 | ~0.17% / ~0.999 | ~1.77% / ~0.859 | 1.80% / 0.855 | +0.140 |
| delay MAPE/R2  | ~10.88% / ~0.703 | ~11.11% / ~0.697 | 11.38% / 0.687 | +0.006 |
| jitter R2/w1std | ~0.726 / 94.4% | ~0.728 / 94.4% | 0.723 / -- | ~0 |

RESULT: jitter's R2 jumped from 0.479 (absolute, gnn+lag) to ~0.726
(delta) -- FINALLY matching/slightly exceeding RF+lag's 0.723, the one
comparison this whole investigation hadn't closed. Confirmed over 3
inits, tight consistency (w1std identical to 1 decimal across all 6
runs). Energy/delay also improved slightly (0.19%->0.17% MAPE,
10.99%->10.88% MAPE).

IMPORTANT CAVEAT: the jitter improvement is NOT a new graph advantage --
gnn+lag+delta and blind+lag+delta are statistically tied on jitter
(gap ~0, blind even marginally ahead on 2/3 inits). Delta-framing
unlocked the temporal signal more effectively for BOTH models equally.
Energy remains the only target where the gap over blind is large and
attributable to the graph itself. Report this precisely: "our best
model beats RF+lag on jitter" is true; "the graph explains jitter's
improvement" is not.

MANUSCRIPT RECOMMENDATION: report predict_delta as part of the final
5G3E training recipe (it strictly improves or ties every target, no
downside found), but keep the causal claims separated by target exactly
as above -- energy = graph value, delay/jitter = temporal value via
lag+delta framing, not graph value.

===============================================================================
COMBINED CONFIG: no_e1 + rounds4 together -- DONE -- NEW BEST OVERALL
===============================================================================

| target | gnn | blind | gap |
|---|---|---|---|
| energy MAPE / R2 | 0.31% +/- 0.03% / 0.996 | 4.47% +/- 0.03% / 0.165 | **+0.831 (best of the whole study)** |
| delay MAPE / R2  | 15.48% +/- 0.02% / 0.457 | 16.47% +/- 0.06% / 0.388 | +0.069 |
| jitter w1std_acc | 71.6% +/- 0.3% | 69.5% +/- 0.7% | -0.001 |
| packet_loss | N/A | | |

Ties no_e1's best energy MAPE (0.31%) while getting the highest energy
R2 gap seen anywhere (+0.831, beating no_e1 alone's +0.829). Delay MAPE
(15.48%) doesn't quite match rounds4 alone (15.28%), but its variance
(+-0.02%) is the tightest of the entire study -- essentially
deterministic across the 3 seeds. Net result: a clear win over `full`
on every axis, with by far the most consistent results across seeds of
any config tested.

RECOMMENDATION FOR THE MANUSCRIPT: report `combined` (no_e1 dropped,
T=4) as the recommended WirelessDT-6G configuration for 5G3E, not
`full`. It is simpler (fewer message-passing rounds, one less edge
type), matches the spec-following `full` config's graph-value story
(energy R2 gap +0.83 vs blind), and is measurably more accurate and
more reproducible.

===============================================================================
SUPERSEDED: re-verification on a genuine held-out day (Day 3)
===============================================================================
Everything above was measured on Day1-train / Day2-test -- NOT a real
generalisation test, since Day1/Day2 are adjacent and likely share
session-specific quirks. Once a real Day 3 arrived, we rebuilt the
split as Day1+2-train (11,722/2,068) / Day3-test (6,974, genuinely
unseen) and re-ran the key configs. Two of the "recommended" claims
above did NOT survive:

  - `combined` (no_e1 + rounds4) stopped being the best config. On Day3,
    `full` (rounds=10, all edges) clearly beat it on delay (R2 0.356 vs
    0.267, MAPE 17.64% vs 18.98%, gap +0.060 vs -0.028), energy/jitter
    roughly tied. rounds4's advantage on the old split looks like it was
    partly fit to that easier split, not a real property of the model.
  - naive + Random Forest baselines (baseline_classical.py, same
    features blind gets, nothing more) showed RF matching or beating
    gnn on delay/jitter on Day3 -- the graph's already-thin delay/
    jitter advantage from the old split had NOT survived a genuine
    generalisation test.

===============================================================================
THE HEADLINE FINDING: temporal signal, not architecture, was the gap
===============================================================================
Hypothesis: none of naive/RF/blind/gnn had ever seen a gNB's own recent
history -- every prediction was made from one isolated snapshot.
Delay/jitter are congestion effects that build up over several
snapshots, so this looked like a missing TEMPORAL signal, not a
spatial/architectural one.

Cheap test first: added ONE feature to RF -- each gNB's own
[energy,delay,jitter] from the previous snapshot ("RF+lag").
Result (Day3 test):
  energy   RF R2 0.290 -> RF+lag R2 0.855  (MAPE 4.03% -> 1.80%)
  delay    RF R2 0.283 -> RF+lag R2 0.687  (MAPE 18.72% -> 11.38%)
  jitter   RF R2 0.163 -> RF+lag R2 0.723  (MAPE 57.42% -> 26.04%)
RF+lag alone (no graph, no learned representation) beat our full `gnn`
(no lag) on delay and jitter. This is a far bigger effect than any
single architecture ablation in this whole sweep (which moved R2 by
hundredths; this moved it by tenths).

Gave the SAME lag feature to gnn and blind (SnapshotDataset(use_lag=True)
in wdt_data.py, appends each gNB's own previous-snapshot
[energy,delay,jitter] to its 5 input features; day-boundary snapshots
have no valid lag and are dropped). Confirmed over 3 inits, Day3 test:

| target | gnn+lag | blind+lag | RF+lag | gap (gnn+lag - blind+lag) |
|---|---|---|---|---|
| energy MAPE/R2 | 0.19%+/-0.03% / 0.998 | 1.78%+/-0.01% / 0.857 | 1.80% / 0.855 | **+0.141 -- graph still essential** |
| delay MAPE/R2  | 10.99%+/-0.09% / 0.699 | 11.26%+/-0.01% / 0.694 | 11.38% / 0.687 | +0.005 -- ~zero, time did the work |
| jitter MAPE/R2/w1std | 57.32% / 0.479 / 93.4% | 57.28% / 0.480 / 93.4% | 26.04%(MAPE) / 0.723 / -- | ~0 gap vs blind; RF+lag still leads on R2 |

CONCLUSION FOR THE MANUSCRIPT:
  - Temporal information (even one lag step) matters more than every
    architecture choice tested in this study combined.
  - Energy: the graph remains essential even with time available --
    cpu_util is a specific current value only the backhaul edge can
    reveal; a temporal proxy from a gNB's own history gets close but
    stays far behind (10x+ MAPE difference).
  - Delay: once both models see their own history, the graph's
    contribution vanishes to near-zero. Most of delay's earlier
    "graph advantage" was actually unaccounted-for temporal signal.
  - Jitter: still an open question. gnn+lag and blind+lag are
    statistically tied (graph adds nothing here either), and RF+lag's
    R2 (0.723) is still ahead of both (0.479-0.480) -- a classical
    method with one lag feature beats the graph model on jitter's
    strict fit metric, even though gnn+lag's w1std tolerance-band
    number (93.4%) looks good in isolation. Do not claim a jitter win
    without addressing this.

NEXT STEPS DISCUSSED, NOT YET DONE:
  - Real temporal architecture (GRU across snapshots, not just within
    one snapshot's message-passing rounds) -- lag is a 1-step proxy;
    a proper temporal mechanism might close jitter's remaining gap.
  - XGBoost as a second classical baseline, to check RF wasn't a fluke.
  - Same baseline ladder (naive/RF/RF+lag) needed on scenario06c for
    cross-dataset consistency -- naive+RF done (see below), RF+lag on
    06c not yet run.

===============================================================================
CROSS-DATASET CHECK: scenario06c classical baselines (src/v6/baseline_classical.py)
===============================================================================
Same fairness rule (RF gets exactly what blind gets, nothing more),
same 3-seed rotation methodology as the existing gnn/blind numbers.

| target | naive R2 | RF R2 | blind R2 | gnn R2 (full config) |
|---|---|---|---|---|
| energy | -0.012 | **0.698** | 0.225 | 0.532 |
| delay  | -0.020 | 0.062 | 0.067 | 0.522 |
| jitter | -0.032 | -0.029 | 0.050 | 0.497 |
| packet_loss (AUROC) | -- | 0.822 | 0.828 | 0.812 |

FINDING -- the OPPOSITE pattern from 5G3E: on 06c, RF beats gnn AND
blind on energy (0.698 vs 0.532 vs 0.225); gnn dominates RF on
delay/jitter (0.52/0.50 vs 0.06/-0.03); packet_loss ties across all
three. Cross-dataset conclusion: which target benefits from the graph
vs. a classical baseline is NOT consistent across datasets -- energy
needs the graph on 5G3E but not on 06c, delay/jitter need the graph on
06c but not (clearly) on 5G3E. Report this explicitly rather than
picking whichever result flatters the method; neither dataset alone
tells the whole story.

HYPOTHESIS for why RF beats gnn on 06c's energy specifically: negative
transfer from joint multi-task training. gnn's encoder is shared across
all 4 loss terms (Huber(energy)+MSE(delay)+jitter loss+PL loss); if
energy's true function is simple/local (matches GNB_COLS demand
features closely) while delay/jitter benefit from richer shared
representation, the joint objective may pull the encoder toward what
helps delay/jitter at energy's expense -- while RF, trained on energy
alone, has no such competition. Possible breadcrumb: c3_heads.py's
"single-task beats joint MoE... locked decision from the scenario07
work" may originally refer to this same joint-vs-separate-models
question, not just MoE-vs-plain-head. NOT YET TESTED. Proposed
diagnostic (single-task-only gnn for energy) was set aside as "not our
committed architecture" -- more actionable next step agreed instead:
try a capacity boost on the energy head ALONE (deeper head, shared
encoder kept) to see if that narrows the gap without abandoning joint
training.

RESULT (src/v6/c3_heads.py's _deep_head, 2x width x3 layers plain MLP,
energy head only, --deep-energy flag, full 3-seed rotation):
  energy R2   gnn 0.532 -> 0.462   blind 0.225 -> 0.160   gap 0.307 -> 0.301
  delay R2    gnn 0.522 -> 0.463 (unaffected head, plausible run noise)
  jitter R2   gnn 0.497 -> 0.440 (unaffected head, plausible run noise)
  packet_loss AUROC 0.812 -> 0.815 (unaffected head, unchanged)

NEGATIVE RESULT: capacity boost does not close the RF gap and does not
move the gnn-vs-blind gap at all (+0.307 -> +0.301, within noise).
gnn's ABSOLUTE energy R2 got worse, not better, moving further from
RF's 0.698. Rules out "energy head is capacity-starved" as the
explanation. Remaining, untested hypotheses: (a) genuine negative
transfer from the shared encoder being pulled toward what helps
delay/jitter (which stayed strong here) at energy's expense, or (b)
RF's tree/threshold inductive bias is simply better suited to this
specific function than any neural approach tried so far (MoE, plain,
now deep MLP), independent of capacity. Neither confirmed. The
single-task diagnostic (deliberately not tried, since it's a different
architecture, not a tunable of the committed one) is the remaining way
to test (a) specifically.

===============================================================================
06c RF+lag: same temporal diagnostic, MUCH bigger effect than 5G3E
===============================================================================
src/v6/baseline_classical.py's run_lag(), same 3-seed rotation as every
other 06c number. 06c splits targets across node types (5G3E has all 3
on one gNB node), so lag splits the same way: energy's RF+lag gets the
gNB's own previous estimated_gnb_power_w; delay/jitter's RF+lag gets
the UE's own previous [delay_ms, jitter_ms]. First snapshot of each run,
per node, dropped (no valid lag) rather than fabricated.

| target | RF (no lag) | RF+lag | gnn (no lag, for reference) |
|---|---|---|---|
| energy | 0.698 | **0.972 +/- 0.001** (MAPE 1.19%+/-0.11%) | 0.532 |
| delay  | 0.062 | **0.940 +/- 0.008** (MAPE 7.23%+/-0.51%) | 0.522 |
| jitter | -0.029 | **0.945 +/- 0.008** (MAPE 16.74%+/-1.48%) | 0.497 |

RESULT: far bigger than 5G3E's lag effect (which moved R2 by 0.4-0.6 on
delay/jitter and left energy's RF+lag well below gnn+lag). Here RF+lag
jumps R2 by +0.27 (energy) to +0.97 (jitter) and, using ONLY a flat
tree model plus one previous-snapshot value per node (no graph, no
learned representation, no lag on gnn at all), lands ABOVE gnn's
(un-lagged) numbers on all three targets simultaneously -- including
energy, where 06c's gnn was already the weaker model relative to RF
even without any lag feature.

CRITICAL CAVEAT -- this is NOT yet a fair comparison. Exactly like the
5G3E investigation, the honest next step is gnn+lag / blind+lag on 06c
before drawing any conclusion about the graph's value here: RF+lag's
apparent dominance could evaporate once gnn/blind get the same temporal
information (matching 5G3E's outcome for delay/jitter -- lag closed the
RF-vs-gnn gap once both saw it), OR the graph could still matter for
some subset of targets once temporal signal is equalised (matching
5G3E's outcome for energy, where the graph stayed essential even with
lag). Do NOT report "RF beats gnn on 06c" using these numbers alone --
report it as "un-lagged gnn vs lagged RF," which is not the like-for-
like comparison the manuscript needs. gnn+lag/blind+lag on 06c is now
the single highest-value remaining run, given the size of this effect.

===============================================================================
06c gnn+lag / blind+lag: the fair comparison, run -- REVERSES the
headline delay/jitter finding, not just the energy anomaly
===============================================================================
Built src/v6/wdt_data.py's add_lag_features() (mirrors the RF+lag
diagnostic's node-type split: gnb gets its own previous
estimated_gnb_power_w, ue gets its own previous [delay_ms, jitter_ms])
and train_wdt_s06c.py's --use-lag flag. Ran the SAME methodology as
every other reported 06c number: 3-seed rotation x 3 inits, rounds=10
(NOT rounds4 -- 06c's own sweep_summary.json already showed rounds4 is
worse here: energy R2 0.452 vs full's 0.532, delay 0.477 vs 0.522,
jitter 0.448 vs 0.497 -- opposite of 5G3E, where T=4 tied/beat T=10).
25 run-start snapshots (no valid lag) dropped from train/val/test.

| target | gnn (no lag) | blind (no lag) | gap | gnn+lag | blind+lag | gap | RF+lag |
|---|---|---|---|---|---|---|---|
| energy | 0.532 | 0.225 | +0.307 | 0.928+/-0.017 | 0.919+/-0.024 | **+0.009 (5/9 pos)** | 0.972+/-0.001 |
| delay  | 0.522 | 0.067 | +0.455 | 0.974+/-0.013 | 0.987+/-0.008 | **-0.013 (0/9 pos)** | 0.940+/-0.008 |
| jitter | 0.497 | 0.050 | +0.447 | 0.969+/-0.008 | 0.990+/-0.003 | **-0.021 (0/9 pos)** | 0.945+/-0.008 |

gnn+lag physical units: energy MAPE 2.87% (target <10%, met), delay
MAPE 23.54% (target <17.39% M3Net, NOT met despite R2=0.974 -- high R2
with a high MAPE here means the model tracks the shape well but still
misses on the smallest-magnitude rows, which dominate a MAPE average),
jitter MAPE 54.63% (no formal target).

THE FINDING -- bigger than the original RF-vs-energy question:
This was meant to check whether 06c's RF-beats-gnn-on-energy anomaly
had the same temporal-signal explanation as 5G3E's. It does (energy's
gap collapses from +0.307 to +0.009, statistically noise, 5/9 favors
gnn -- a coin flip). But delay and jitter -- which were 06c's STRONGEST
reported graph-value claims (gap +0.455 and +0.447, "gnn dominates RF
and blind" in the existing manuscript framing) -- do not just weaken,
they REVERSE: blind+lag beats gnn+lag in every single one of the 9
rotation/init combinations on both targets. Once both models see their
own last snapshot's value, message passing adds nothing measurable on
this dataset for ANY of the three continuous targets, and for two of
them it's actively a (small) net negative.

CONTRAST WITH 5G3E -- why this doesn't invalidate the whole approach:
5G3E's energy kept a real graph advantage even with lag (gap +0.14),
independently confirmed by a structural mechanism: no_e4 (dropping the
backhaul edge) collapses gnn to blind's level specifically for energy,
because cpu_util has no other path to a gNB node. That is a causal,
edge-specific story, not just a correlation that happened to survive
one check. No comparable structural confirmation exists for 06c -- once
lag is accounted for, nothing here has an equivalent no_e4 result to
point to. The two datasets together tell a consistent, more general
lesson: an apparent gnn > blind gap measured WITHOUT giving both models
access to temporal context is not trustworthy evidence of the graph's
causal contribution on its own, and should not be reported as such
without either (a) re-checking it survives a fair lag feature, or
ideally (b) an edge-ablation result that independently explains WHY
that specific relation should matter for that specific target.

MANUSCRIPT-CRITICAL REVISION NEEDED: any existing manuscript text
describing 06c's gnn as "dominating on delay/jitter" (using the
un-lagged 0.522/0.067 and 0.497/0.050 numbers) needs to be reframed.
The honest version: 06c's single-snapshot gnn does outperform its
single-snapshot blind control substantially on delay/jitter, but that
advantage is now shown to be a proxy for missing temporal information
rather than a demonstrated benefit of message passing -- once both
models get the trivial one-step lag feature RF also benefits from, the
advantage disappears (energy) or reverses (delay, jitter). This is a
negative-for-the-graph result and should be reported as such, not
smoothed over -- it is also a genuinely interesting cross-dataset
finding in its own right (temporal signal dominates spatial/graph
signal far more completely on 06c than on 5G3E, where at least energy's
graph value survived).

===============================================================================
CORRECTION -- 06c's delay/jitter/PL lag numbers above are VOID (label
leakage), and the energy-only clean re-run SHARPENS the reversal
===============================================================================
Discovered while designing a follow-up C4 test (see evaluate_c4_compare.py):
06c's delay_ms/jitter_ms/packet_loss come from kpi_targets.csv, which
broadcasts ONE measurement across each 3-second phase's 30 gnb-snapshot
rows (documented in wdt_data.py's KPI_FILE comment). Verified directly:
0/2400 phases have more than one distinct delay value, and lag_delay_ms
== current delay_ms EXACTLY on 96.7% of rows (69,600/72,000, same-phase
pairs). A "previous snapshot's own delay" feature is therefore not a
temporal signal for these three targets -- it is the label itself,
handed to the model as an input, for essentially every row. The
"gnn+lag/blind+lag delay/jitter R2 ~0.97-0.99" numbers immediately
above, and the matching RF+lag numbers (0.940/0.945) reported earlier
in this file, are VOID: they measured which model most cleanly learns
to copy one input dimension, not whether the graph helps. RETRACTED,
not merely caveated.

energy_targets.csv has NO such broadcast -- verified 7.9% exact-repeat
rate snapshot-to-snapshot (genuinely fresh almost every second), so
lag_energy_w is legitimate. But the ALREADY-REPORTED gnn+lag energy
number (0.928, gap +0.009 over blind+lag's 0.919) used the "full lag"
checkpoint, which also carried the leaky UE-side lag_delay_ms/
lag_jitter_ms -- for gnn specifically (not blind, which never
aggregates across nodes) those could reach the energy prediction via
ue->gnb message passing, even though energy's own direct inputs were
clean. Re-ran with a config that has ZERO leakage pathway anywhere:
gnb gets lag_energy_w only, ue gets nothing at all
(--use-lag-energy-only in train_wdt_s06c.py, tag energy_lag_only).

| target | gnn (no lag) | blind (no lag) | gnn+full-lag (LEAKY) | blind+full-lag (LEAKY) | gnn+ENERGY-lag-only (clean) | blind+ENERGY-lag-only (clean) |
|---|---|---|---|---|---|---|
| energy | 0.532 | 0.225 | 0.928 | 0.919 (gap +0.009) | 0.878+/-0.018 | **0.925+/-0.014 (gap -0.047, 0/9 gnn wins)** |
| delay  | 0.522 | 0.067 | VOID (leaked) | VOID (leaked) | 0.503+/-0.052 | 0.098+/-0.052 (gap +0.405, matches un-lagged) |
| jitter | 0.497 | 0.050 | VOID (leaked) | VOID (leaked) | 0.486+/-0.055 | 0.084+/-0.053 (gap +0.402, matches un-lagged) |

RESULT: removing the leakage pathway didn't soften the energy finding,
it SHARPENED it. With the leaky full-lag checkpoint gnn and blind were
a near-tie on energy (+0.009, 5/9 gnn). With zero leakage possible,
blind wins outright, in ALL 9 rotation/init runs (-0.047). This is
consistent with gnn having gotten a small illegitimate boost through
the leaky UE-side feature reaching it via message passing in the
"full lag" run -- once that path is closed, blind is the clearly
better energy model, not a tied one. Delay/jitter land close to the
original un-lagged numbers, as expected (this config gives ue nothing
new, so nothing should change there beyond ordinary training noise).

REVISED HEADLINE FOR THE MANUSCRIPT:
  - Energy: once genuine (non-leaky) temporal signal is available,
    blind OUTPERFORMS gnn on 06c, not just ties it. Use the
    energy_lag_only numbers (0.878 vs 0.925), not the earlier full-lag
    ones (0.928 vs 0.919) -- the latter had a possible leakage-mediated
    advantage for gnn specifically.
  - Delay/jitter: NO valid lag-based claim can be made for 06c on
    these two targets with this dataset's measurement resolution.
    Report the un-lagged numbers (gnn substantially beats blind,
    0.522/0.067 and 0.497/0.050) as-is -- they were never contaminated
    -- but do NOT claim this has been "explained" or "explained away"
    by a temporal-signal story the way energy has. That question
    remains genuinely open for 06c's delay/jitter, and cannot be
    answered with a lag feature on this dataset at all.
  - Cross-check performed: 5G3E was independently verified for the
    same broadcast issue and is clean (energy 0.35%, delay 2.59%,
    jitter 3.36% consecutive-snapshot exact-repeat rate, all consistent
    with genuinely fresh per-snapshot measurement) -- none of the 5G3E
    lag/delta findings from this file are affected.
(stale placeholders removed -- actual no_e3/no_e4/no_moe/no_gru/no_attn
Day3 results are recorded earlier in this file, at the "RE-VERIFICATION
ROUND 2" section and its RUNNING TALLY entries. rounds4 is the last
config, running now.)
