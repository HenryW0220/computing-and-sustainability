#!/usr/bin/env python3
"""Generate week5_reward_curves.png from TensorBoard event files."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

ROOT   = Path("d:/NU/CS")
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True)

C_PPO = "#E69F00"   # amber  (Wong palette)
C_DQN = "#0072B2"   # blue

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13,
})

# ---------------------------------------------------------------------------
# Load eval/mean_reward from TensorBoard event files
# ---------------------------------------------------------------------------
def load_scalar(run_dir: str, tag: str):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values

ppo_steps, ppo_eval = load_scalar(
    str(ROOT / "tb/ALE-Pong-v5/PPO_4"), "eval/mean_reward")
dqn_steps, dqn_eval = load_scalar(
    str(ROOT / "tb/ALE-Pong-v5/DQN_1"), "eval/mean_reward")

# Also load rollout reward for smoother background trace
ppo_roll_steps, ppo_roll = load_scalar(
    str(ROOT / "tb/ALE-Pong-v5/PPO_4"), "rollout/ep_rew_mean")
dqn_roll_steps, dqn_roll = load_scalar(
    str(ROOT / "tb/ALE-Pong-v5/DQN_1"), "rollout/ep_rew_mean")

ppo_best_idx = int(np.argmax(ppo_eval))
dqn_best_idx = int(np.argmax(dqn_eval))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Rollout traces (light, background)
ax.plot(ppo_roll_steps / 1e6, ppo_roll,
        color=C_PPO, lw=0.8, alpha=0.25)
ax.plot(dqn_roll_steps / 1e6, dqn_roll,
        color=C_DQN, lw=0.8, alpha=0.25)

# Eval traces (solid, foreground)
ax.plot(ppo_steps / 1e6, ppo_eval,
        color=C_PPO, lw=2.2, label="PPO — eval reward (2M steps)")
ax.plot(dqn_steps / 1e6, dqn_eval,
        color=C_DQN, lw=2.2, label="DQN — eval reward (1M steps)")

# Best-reward stars
ax.scatter([ppo_steps[ppo_best_idx] / 1e6], [ppo_eval[ppo_best_idx]],
           color=C_PPO, s=200, zorder=6, marker="*", linewidths=0)
ax.annotate(
    f"PPO best: {ppo_eval[ppo_best_idx]:.1f}\n@ {ppo_steps[ppo_best_idx]/1e6:.2f}M",
    xy=(ppo_steps[ppo_best_idx] / 1e6, ppo_eval[ppo_best_idx]),
    xytext=(ppo_steps[ppo_best_idx] / 1e6 - 0.45,
            ppo_eval[ppo_best_idx] + 1.8),
    fontsize=12, color=C_PPO,
    arrowprops=dict(arrowstyle="->", color=C_PPO, lw=1.3),
)

ax.scatter([dqn_steps[dqn_best_idx] / 1e6], [dqn_eval[dqn_best_idx]],
           color=C_DQN, s=200, zorder=6, marker="*", linewidths=0)
ax.annotate(
    f"DQN best: {dqn_eval[dqn_best_idx]:.1f}\n@ {dqn_steps[dqn_best_idx]/1e6:.2f}M",
    xy=(dqn_steps[dqn_best_idx] / 1e6, dqn_eval[dqn_best_idx]),
    xytext=(dqn_steps[dqn_best_idx] / 1e6 - 0.42,
            dqn_eval[dqn_best_idx] - 3.0),
    fontsize=12, color=C_DQN,
    arrowprops=dict(arrowstyle="->", color=C_DQN, lw=1.3),
)

# Chance baseline
ax.axhline(-21, color="gray", lw=1.0, ls="--", alpha=0.55, label="Chance level (−21)")

ax.set_xlabel("Timesteps (millions)")
ax.set_ylabel("Mean Eval Reward")
ax.set_title("Week 5 — PPO vs DQN on ALE/Pong-v5\n"
             "Eval reward every 25k steps  |  faint lines = rollout reward",
             pad=10)
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(left=0)
ax.set_ylim(-22.5, -6)

fig.tight_layout()
out = OUTDIR / "week5_reward_curves.png"
fig.savefig(str(out), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"PPO best eval: {ppo_eval[ppo_best_idx]:.1f} @ step {ppo_steps[ppo_best_idx]:,.0f}")
print(f"DQN best eval: {dqn_eval[dqn_best_idx]:.1f} @ step {dqn_steps[dqn_best_idx]:,.0f}")
