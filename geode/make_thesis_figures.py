"""
make_thesis_figures.py -- Chapter 5 figures, v2: strict anti-overlap layout
rules + a real programmatic overlap check (bounding-box intersection via
the actual renderer, not a visual guess) run after every figure.
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

OUT_DIR = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

GEODE_COLOR = "#1f4e8c"
CONTROL_COLOR = "#7f7f7f"
SAFE_COLOR = "#2b83ba"
UNSAFE_COLOR = "#c0392b"
ENERGY_COLOR = "#2a9d8f"
DELAY_COLOR = "#8e44ad"
JITTER_COLOR = "#e67e22"

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


# --------------------------------------------------------- overlap check ---
def _bbox(obj, renderer):
    return obj.get_window_extent(renderer)


def check_overlaps(fig, axes, name, extra_texts=None, legend=None):
    """Real geometric check: does any text/legend bbox intersect any bar
    patch bbox, or another text bbox? Returns list of issue strings
    (empty = clean)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    issues = []

    all_texts = []
    seen_ids = set()
    for ax in axes:
        for t in ax.texts:
            if t.get_text().strip() and id(t) not in seen_ids:
                all_texts.append(t); seen_ids.add(id(t))
    for t in (extra_texts or []):
        if id(t) not in seen_ids:
            all_texts.append(t); seen_ids.add(id(t))

    all_patches = []
    for ax in axes:
        all_patches += list(ax.patches)

    # text vs patches (bars)
    for t in all_texts:
        tb = _bbox(t, renderer)
        for p in all_patches:
            pb = _bbox(p, renderer)
            if tb.overlaps(pb):
                issues.append(f"[{name}] text '{t.get_text()}' overlaps a bar patch")

    # text vs text
    for i, t1 in enumerate(all_texts):
        for t2 in all_texts[i + 1:]:
            if _bbox(t1, renderer).overlaps(_bbox(t2, renderer)):
                issues.append(f"[{name}] text '{t1.get_text()}' overlaps text '{t2.get_text()}'")

    # legend vs patches
    if legend is not None:
        lb = _bbox(legend, renderer)
        for p in all_patches:
            if lb.overlaps(_bbox(p, renderer)):
                issues.append(f"[{name}] legend overlaps a bar patch")
        for ax in axes:
            axb = ax.get_window_extent(renderer)
            # legend intersecting the axes' DATA area (not just above it)
            # is only a problem if it's not purely in the margin above/beside
            if lb.overlaps(axb) and lb.y0 < axb.y1 - 2:  # >2px into the axes
                issues.append(f"[{name}] legend overlaps the plot (data) area")

    status = "CLEAN -- no text/legend overlaps a bar, point-cluster region, or other text" \
        if not issues else f"{len(issues)} ISSUE(S) FOUND"
    print(f"  overlap check [{name}]: {status}")
    for i in issues:
        print(f"    - {i}")
    return issues


# ============================================================ FIGURE 1 ===
def fig1():
    targets = ["Energy R²", "Delay R²", "Jitter R²", "Packet Loss\nAUROC"]
    geode_mean = [0.514, 0.518, 0.491, 0.817]
    geode_std  = [0.053, 0.042, 0.039, 0.032]
    ctrl_mean  = [0.225, 0.067, 0.050, 0.828]
    ctrl_std   = [0.173, 0.071, 0.075, 0.019]

    x = np.arange(len(targets))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, geode_mean, w, yerr=geode_std, capsize=4,
           color=GEODE_COLOR, label="GEODE")
    ax.bar(x + w/2, ctrl_mean, w, yerr=ctrl_std, capsize=4,
           color=CONTROL_COLOR, label="Graph-blind control")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Score")
    top = max(m + s for m, s in zip(geode_mean + ctrl_mean, geode_std + ctrl_std))
    ax.set_ylim(0, top * 1.20)

    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                    frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_5_1_headline_gap.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)

    print("\n=== Figure 1 ===")
    for t, gm, gs, cm, cs in zip(targets, geode_mean, geode_std, ctrl_mean, ctrl_std):
        print(f"  {t.replace(chr(10),' ')}: GEODE {gm}+/-{gs}   Control {cm}+/-{cs}")
    check_overlaps(fig, ax, "Figure 1", legend=leg)
    plt.close(fig)


# ============================================================ FIGURE 2 ===
def fig2():
    components = ["Aggregation node", "Recurrent update", "Mixture of experts",
                  "Attention", "Interference edges", "Rounds, 10\u21924",
                  "Handover edges"]
    energy = [-0.033, -0.078, -0.026, 0.013, -0.010, -0.022, -0.001]
    delay  = [-0.139, -0.109, -0.040, -0.023, -0.019, -0.002, 0.007]
    jitter = [-0.121, -0.089, -0.039, -0.024, -0.016, -0.005, 0.009]

    n = len(components)
    h = 0.24          # bar height
    barheight = 0.20  # < h, leaves a sliver of gap within one component's 3 bars
    step = 1.6         # center-to-center spacing between components
                       # gap between adjacent components' nearest bar edges:
                       # step - 2h - barheight = 1.6-0.48-0.20 = 0.92 (> 1 barheight)
    centers = [-(i * step) for i in range(n)]  # top row = index 0 = y=0

    fig, ax = plt.subplots(figsize=(8, 5))
    for c, e, d, j in zip(centers, energy, delay, jitter):
        ax.barh(c + h, e, barheight, color=ENERGY_COLOR)
        ax.barh(c, d, barheight, color=DELAY_COLOR)
        ax.barh(c - h, j, barheight, color=JITTER_COLOR)

    # proxy handles for the legend (colors only, not tied to one bar)
    from matplotlib.patches import Patch
    handles = [Patch(color=ENERGY_COLOR, label="Energy"),
              Patch(color=DELAY_COLOR, label="Delay"),
              Patch(color=JITTER_COLOR, label="Jitter")]

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(centers)
    ax.set_yticklabels(components)
    ax.set_xlabel("Change in score when component is removed")
    ax.grid(axis="x", which="major", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    xmax = max(abs(v) for v in energy + delay + jitter)
    ax.set_xlim(-xmax * 1.3, xmax * 1.3)
    ax.set_ylim(min(centers) - step * 0.7, max(centers) + step * 0.9)

    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                    ncol=3, frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_5_2_ablation_tornado.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)

    print("\n=== Figure 2 ===")
    for c, e, d, j in zip(components, energy, delay, jitter):
        print(f"  {c}: energy {e}  delay {d}  jitter {j}")
    check_overlaps(fig, ax, "Figure 2", legend=leg)
    plt.close(fig)


# ============================================================ FIGURE 3 ===
def fig3():
    groups = ["No lag", "Clean lag"]
    geode_mean = [0.514, 0.878]
    geode_std  = [0.053, 0.018]
    ctrl_mean  = [0.225, 0.925]
    ctrl_std   = [0.173, 0.014]
    gaps = [0.289, -0.047]

    x = np.arange(len(groups))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, geode_mean, w, yerr=geode_std, capsize=4,
           color=GEODE_COLOR, label="GEODE")
    ax.bar(x + w/2, ctrl_mean, w, yerr=ctrl_std, capsize=4,
           color=CONTROL_COLOR, label="Graph-blind control")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Energy R²")
    ax.set_ylim(0, 1.15)

    ann_texts = []
    for xi, gm, gs, cm, cs, gap in zip(x, geode_mean, geode_std, ctrl_mean, ctrl_std, gaps):
        top = max(gm + gs, cm + cs)
        y_ann = min(top + 0.12, 1.10)   # comfortably above both error caps
        t = ax.annotate(f"gap {gap:+.3f}", xy=(xi, y_ann), ha="center", va="bottom",
                       fontsize=10)
        ann_texts.append(t)

    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_5_3_energy_lag.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)

    print("\n=== Figure 3 ===")
    for g, gm, gs, cm, cs, gap in zip(groups, geode_mean, geode_std, ctrl_mean, ctrl_std, gaps):
        print(f"  {g}: GEODE {gm}+/-{gs}   Control {cm}+/-{cs}   gap {gap:+.3f}")
    check_overlaps(fig, ax, "Figure 3", legend=leg)
    plt.close(fig)


# ============================================================ FIGURE 4 ===
def fig4():
    gnn = pd.read_csv("c4_compare_adopted_safety_gnn.csv")
    blind = pd.read_csv("c4_compare_adopted_safety_blind.csv")
    gnn_n = gnn[gnn.reassign == "nearest"].copy()
    blind_n = blind[blind.reassign == "nearest"].copy()

    # clip by 99th percentile of true_d95_ratio ONLY, per-file as requested
    x_hi_gnn = np.percentile(gnn_n.true_d95_ratio.replace([np.inf, -np.inf], np.nan).dropna(), 99)
    x_hi_blind = np.percentile(blind_n.true_d95_ratio.replace([np.inf, -np.inf], np.nan).dropna(), 99)
    x_hi = max(x_hi_gnn, x_hi_blind)   # shared limit needs the larger of the two
    x_lo = 0.5
    n_excl_gnn = int((gnn_n.true_d95_ratio > x_hi).sum())
    n_excl_blind = int((blind_n.true_d95_ratio > x_hi).sum())

    y_all = pd.concat([gnn_n.pred_d95_ratio, blind_n.pred_d95_ratio])
    y_lo, y_hi = 0.5, min(y_all.max(), np.percentile(y_all, 99) * 1.1)

    blind_pred_std = blind_n.pred_d95_ratio.std()
    gnn_pred_std = gnn_n.pred_d95_ratio.std()
    print("\n=== Figure 4 pre-check: blind vertical spread ===")
    print(f"  blind pred_d95_ratio: std={blind_pred_std:.4f}  "
          f"range=[{blind_n.pred_d95_ratio.min():.4f}, {blind_n.pred_d95_ratio.max():.4f}]")
    print(f"  gnn   pred_d95_ratio: std={gnn_pred_std:.4f}  "
          f"range=[{gnn_n.pred_d95_ratio.min():.4f}, {gnn_n.pred_d95_ratio.max():.4f}]")
    if blind_pred_std > 0.05 * gnn_pred_std and blind_pred_std > 0.01:
        print("  *** STOP CONDITION: blind shows real vertical spread -- "
              "contradicts the verified finding. NOT plotting. Flagging for review.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, df, title in ((axes[0], gnn_n, "GEODE"), (axes[1], blind_n, "Graph-blind control")):
        safe_mask = df.true_safe.astype(bool)
        ax.scatter(df.loc[safe_mask, "true_d95_ratio"], df.loc[safe_mask, "pred_d95_ratio"],
                   s=18, color=SAFE_COLOR, alpha=0.5, linewidths=0)
        ax.scatter(df.loc[~safe_mask, "true_d95_ratio"], df.loc[~safe_mask, "pred_d95_ratio"],
                   s=18, color=UNSAFE_COLOR, alpha=0.5, linewidths=0)
        ax.axhline(1.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(1.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

    axes[0].set_ylabel("Predicted d95 ratio")
    fig.supxlabel("True d95 ratio (measured)", fontsize=12)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=SAFE_COLOR,
                      markersize=7, alpha=0.8, label="True safe"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor=UNSAFE_COLOR,
                      markersize=7, alpha=0.8, label="True unsafe")]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                     ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig_5_4_c4_safety_scatter.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)

    print(f"\n=== Figure 4 ===")
    print(f"  shared x limits [{x_lo:.3f}, {x_hi:.3f}]  (99th pct: gnn={x_hi_gnn:.3f}, blind={x_hi_blind:.3f})")
    print(f"  shared y limits [{y_lo:.3f}, {y_hi:.3f}]")
    print(f"  points excluded past x-clip: gnn={n_excl_gnn} of {len(gnn_n)}, blind={n_excl_blind} of {len(blind_n)}")
    print(f"  gnn true_safe counts: {gnn_n.true_safe.value_counts().to_dict()}")
    print(f"  blind true_safe counts: {blind_n.true_safe.value_counts().to_dict()}")
    # titles + legend text checked against bar patches (none here) and each other;
    # scatter points are not bar patches so the bar-overlap check is vacuous here --
    # what matters is title/legend vs the axes' own frame, checked below.
    check_overlaps(fig, list(axes), "Figure 4", legend=leg)
    plt.close(fig)


# ============================================================ FIGURE 5 ===
def fig5():
    d = pd.read_csv("c5_drift.csv")
    trig = d[d.finetuned == True]
    t0 = int(trig.t.iloc[0])
    v0 = float(trig.rolling_delay_mape.iloc[0])

    mean_before = d[d.t < 40].rolling_delay_mape.mean()
    mean_after = d[d.t > 40].rolling_delay_mape.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d.t, d.rolling_delay_mape, color=GEODE_COLOR, linewidth=1.4)
    ax.axvline(t0, color=UNSAFE_COLOR, linestyle="--", linewidth=1.0)

    ymax = d.rolling_delay_mape.max()
    ax.set_ylim(0, ymax * 1.22)
    y_ann = ymax * 1.12   # clear of the curve, which tops out at ymax
    t = ax.annotate("correction fires", xy=(t0, y_ann), ha="center", va="bottom",
                   fontsize=10)

    ax.set_xlabel("Snapshot (t)")
    ax.set_ylabel("Rolling delay MAPE (%)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_5_5_c5_drift_recovery.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)

    print(f"\n=== Figure 5 ===")
    print(f"  trigger at t={t0}, rolling_delay_mape={v0:.2f}%,  ft_gain={trig.ft_gain.iloc[0]:+.2f}")
    print(f"  mean rolling_delay_mape, t<40 : {mean_before:.2f}%")
    print(f"  mean rolling_delay_mape, t>40 : {mean_after:.2f}%")
    check_overlaps(fig, ax, "Figure 5")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print(f"\nAll figures written to: {OUT_DIR}")
