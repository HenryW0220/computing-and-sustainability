#!/usr/bin/env python3
"""
update_week5_figure.py — Read eco-profile.json files, compute mean J/step for
training and inference phases, and regenerate week5_efficiency_comparison.png.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DQN_PROFILE = ROOT / "results" / "dqn_pong" / "eco-profile.json"
PPO_PROFILE = ROOT / "results" / "ppo_pong" / "eco-profile.json"
OUT_FIG = ROOT / "figures" / "week5_efficiency_comparison.png"


def load_profile(path: Path) -> tuple[float, float]:
    """Return (mean_train_J_per_step, mean_infer_J_per_step) from an eco-profile."""
    train_j: list[float] = []
    infer_j: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label: str = rec["label"]
            joules: float = rec["joules"]
            lower = label.lower()
            if "training" in lower:
                train_j.append(joules)
            elif "inference" in lower:
                infer_j.append(joules)

    if not train_j or not infer_j:
        raise ValueError(f"Missing training or inference records in {path}")

    mean_train = sum(train_j) / len(train_j)
    mean_infer = sum(infer_j) / len(infer_j)
    print(f"  {path.parent.name}: train {mean_train:.4f} J/step  "
          f"infer {mean_infer:.4f} J/step  "
          f"({len(train_j)} train, {len(infer_j)} infer records)")
    return mean_train, mean_infer


def fmt(val: float, unit: str = "J/step") -> str:
    return f"{val:.3f} {unit}"


def main() -> None:
    for p in (DQN_PROFILE, PPO_PROFILE):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            print("Run measure_energy.py first.", file=sys.stderr)
            sys.exit(1)

    print("[update] Loading profiles ...")
    ppo_train_j, ppo_infer_j = load_profile(PPO_PROFILE)
    dqn_train_j, dqn_infer_j = load_profile(DQN_PROFILE)

    # Ratio strings for comparison column
    train_ratio = dqn_train_j / ppo_train_j
    infer_ratio = dqn_infer_j / ppo_infer_j
    if train_ratio < 1:
        train_cmp = f"DQN uses {train_ratio:.2f}× PPO\n(train)"
    else:
        train_cmp = f"DQN {train_ratio:.2f}× PPO\n(train)"
    if infer_ratio < 1:
        infer_cmp = f"DQN uses {infer_ratio:.2f}× PPO\n(infer)"
    else:
        infer_cmp = f"DQN {infer_ratio:.2f}× PPO\n(infer)"

    energy_summary = (
        f"Train: {fmt(dqn_train_j)} vs {fmt(ppo_train_j)}\n"
        f"{train_cmp}"
    )

    table_data = [
        ["Steps to best reward",          "1.975 M",              "0.950 M",              "DQN uses 48% of\nPPO steps"],
        ["Wall-clock to best reward",     "~89 min",              "~53 min",              "DQN uses 60% of\nPPO time"],
        ["Best reward achieved",          "−16.2",                "−9.8",                 "DQN +6.4 reward\nhigher"],
        ["Energy per step (J/step)\n  train / infer",
         f"{ppo_train_j:.3f} / {ppo_infer_j:.3f}",
         f"{dqn_train_j:.3f} / {dqn_infer_j:.3f}",
         f"train {train_ratio:.2f}× | infer {infer_ratio:.2f}×\n(DQN / PPO)"],
    ]
    col_labels = ["Metric", "PPO", "DQN", "DQN vs PPO"]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 3.2)

    # Header row
    for col in range(len(col_labels)):
        cell = tbl[0, col]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    row_colors = ["#FDFEFE", "#EBF5FB", "#FDFEFE", "#EAF7EA"]  # green tint for energy row
    for row in range(1, len(table_data) + 1):
        for col in range(len(col_labels)):
            cell = tbl[row, col]
            cell.set_facecolor(row_colors[row - 1])
            if col == 2 and row <= 3:
                cell.set_facecolor("#D6EAF8")
            if col == 3 and row <= 3:
                cell.set_text_props(color="#0072B2", fontweight="bold")
            if row == 4:
                if col == 2:
                    cell.set_facecolor("#D6EAF8")
                cell.set_text_props(color="#1A6B1A", fontweight="bold")

    ax.set_title(
        "Sample & Compute Efficiency — PPO vs DQN (ALE/Pong-v5)",
        fontsize=16, fontweight="bold", pad=18,
    )

    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(str(OUT_FIG), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[update] Saved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
