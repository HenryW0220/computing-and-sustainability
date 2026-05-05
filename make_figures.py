#!/usr/bin/env python3
"""Generate Week 5 presentation figures."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT        = Path("d:/NU/CS")
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PPO_EVAL = ROOT / "logs/ppo/ALE-Pong-v5_6/evaluations.npz"
DQN_EVAL = ROOT / "logs/dqn/ALE-Pong-v5_1/evaluations.npz"

# Color-blind-friendly palette (Wong 2011)
C_PPO = "#E69F00"   # amber
C_DQN = "#0072B2"   # blue

PLT_PARAMS = {
    "font.size":        14,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  13,
    "figure.dpi":       150,
}
plt.rcParams.update(PLT_PARAMS)

# ---------------------------------------------------------------------------
# Load eval data
# ---------------------------------------------------------------------------
def load_eval(path: Path):
    d = np.load(str(path))
    ts   = d["timesteps"].astype(float)
    mean = np.array([r.mean() for r in d["results"]], dtype=float)
    return ts, mean

ppo_ts, ppo_mean = load_eval(PPO_EVAL)
dqn_ts, dqn_mean = load_eval(DQN_EVAL)

ppo_best_idx = int(np.argmax(ppo_mean))
dqn_best_idx = int(np.argmax(dqn_mean))

ppo_best_step, ppo_best_rew = ppo_ts[ppo_best_idx], ppo_mean[ppo_best_idx]
dqn_best_step, dqn_best_rew = dqn_ts[dqn_best_idx], dqn_mean[dqn_best_idx]

# Wall-clock estimates (linear approximation from known totals)
PPO_TOTAL_MIN = 91.0   # minutes for 2M steps
DQN_TOTAL_MIN = 56.2   # minutes for 1M steps

ppo_wc_best = (ppo_best_step / 2_000_000) * PPO_TOTAL_MIN   # ~89 min
dqn_wc_best = (dqn_best_step / 1_000_000) * DQN_TOTAL_MIN   # ~53 min

# ============================================================================
# Figure 1 — week5_training_summary.png
# ============================================================================
fig = plt.figure(figsize=(14, 6))
gs  = GridSpec(1, 2, figure=fig, wspace=0.38)

# --- Subplot A: Reward curves ---
ax1 = fig.add_subplot(gs[0])

ax1.plot(ppo_ts / 1e6, ppo_mean, color=C_PPO, lw=2.0, label="PPO (2M steps)")
ax1.plot(dqn_ts / 1e6, dqn_mean, color=C_DQN, lw=2.0, label="DQN (1M steps)")

# Best-reward markers
ax1.scatter([ppo_best_step / 1e6], [ppo_best_rew],
            color=C_PPO, s=120, zorder=5, marker="*")
ax1.annotate(f"Best: {ppo_best_rew:.1f}\n@ {ppo_best_step/1e6:.2f}M",
             xy=(ppo_best_step / 1e6, ppo_best_rew),
             xytext=(ppo_best_step / 1e6 - 0.55, ppo_best_rew + 1.2),
             fontsize=11, color=C_PPO,
             arrowprops=dict(arrowstyle="->", color=C_PPO, lw=1.2))

ax1.scatter([dqn_best_step / 1e6], [dqn_best_rew],
            color=C_DQN, s=120, zorder=5, marker="*")
ax1.annotate(f"Best: {dqn_best_rew:.1f}\n@ {dqn_best_step/1e6:.2f}M",
             xy=(dqn_best_step / 1e6, dqn_best_rew),
             xytext=(dqn_best_step / 1e6 - 0.55, dqn_best_rew - 2.5),
             fontsize=11, color=C_DQN,
             arrowprops=dict(arrowstyle="->", color=C_DQN, lw=1.2))

ax1.axhline(y=-21, color="gray", lw=0.8, ls="--", alpha=0.6, label="Chance (-21)")
ax1.set_xlabel("Timesteps (millions)")
ax1.set_ylabel("Mean Eval Reward")
ax1.set_title("A — Reward Curves: PPO vs DQN")
ax1.legend(loc="lower right")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlim(left=0)
ax1.set_ylim(-22.5, -7)

# --- Subplot B: Normalized efficiency bar chart ---
ax2 = fig.add_subplot(gs[1])

metrics      = ["Steps to\nBest Reward", "Wall-clock to\nBest Reward"]
ppo_vals     = [1.0, 1.0]                                          # baseline
dqn_vals     = [dqn_best_step / ppo_best_step,
                dqn_wc_best / ppo_wc_best]                        # relative

x    = np.arange(len(metrics))
w    = 0.32

bars_ppo = ax2.bar(x - w/2, ppo_vals, w, color=C_PPO, label="PPO (baseline = 1.0)",
                   edgecolor="white", linewidth=0.5)
bars_dqn = ax2.bar(x + w/2, dqn_vals, w, color=C_DQN, label="DQN (relative to PPO)",
                   edgecolor="white", linewidth=0.5)

# Value labels
raw_labels = [
    [f"1.975M steps\n(91 min)", f"1.975M steps\n(89 min)"],
    [f"{dqn_best_step/1e6:.3f}M steps\n(53 min)",
     f"{dqn_wc_best:.0f} min\n({dqn_vals[1]:.2f}×)"],
]
for i, (bp, bd) in enumerate(zip(bars_ppo, bars_dqn)):
    ax2.text(bp.get_x() + bp.get_width()/2, bp.get_height() + 0.02,
             "1.00×", ha="center", va="bottom", fontsize=10, color=C_PPO, fontweight="bold")
    ax2.text(bd.get_x() + bd.get_width()/2, bd.get_height() + 0.02,
             f"{dqn_vals[i]:.2f}×", ha="center", va="bottom",
             fontsize=10, color=C_DQN, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.set_ylabel("Normalized Cost (PPO = 1.0)")
ax2.set_title("B — Sample & Time Efficiency\n(lower = more efficient)")
ax2.set_ylim(0, 1.35)
ax2.legend(loc="upper right", fontsize=11)
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("Week 5 — PPO vs DQN on ALE/Pong-v5", fontsize=16, fontweight="bold", y=1.01)

out1 = FIGURES_DIR / "week5_training_summary.png"
fig.savefig(str(out1), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out1}")

# ============================================================================
# Figure 2 — week5_efficiency_comparison.png  (table)
# ============================================================================
table_data = [
    ["Steps to best reward",       "1.975 M",    "0.950 M",    "DQN uses 48% of\nPPO steps"],
    ["Wall-clock to best reward",  "~89 min",    "~53 min",    "DQN uses 60% of\nPPO time"],
    ["Best reward achieved",       "−16.2",      "−9.8",       "DQN +6.4 reward\nhigher"],
    ["Energy per step (J/step)",   "TBD",        "TBD",        "TBD — Week 6\nZeus target"],
]
col_labels = ["Metric", "PPO", "DQN", "DQN vs PPO"]

fig2, ax = plt.subplots(figsize=(13, 4.2))
ax.axis("off")

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(14)
tbl.scale(1, 3.2)

# Header row styling
for col in range(len(col_labels)):
    cell = tbl[0, col]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")

# Row colours: alternate white / light grey; highlight "Energy" row (TBD) in light yellow
row_colors = ["#FDFEFE", "#EBF5FB", "#FDFEFE", "#FEFDE7"]
for row in range(1, len(table_data) + 1):
    for col in range(len(col_labels)):
        cell = tbl[row, col]
        cell.set_facecolor(row_colors[row - 1])
        # Highlight DQN column values in blue tint for rows 1-3
        if col == 2 and row <= 3:
            cell.set_facecolor("#D6EAF8")
        # Highlight "DQN vs PPO" column
        if col == 3 and row <= 3:
            cell.set_text_props(color="#0072B2", fontweight="bold")
        if row == 4:
            cell.set_text_props(color="#888888", style="italic")

ax.set_title("Sample & Compute Efficiency — PPO vs DQN (ALE/Pong-v5)",
             fontsize=16, fontweight="bold", pad=18)

out2 = FIGURES_DIR / "week5_efficiency_comparison.png"
fig2.savefig(str(out2), dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved -> {out2}")
