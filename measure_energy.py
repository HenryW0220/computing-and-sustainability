#!/usr/bin/env python3
"""
measure_energy.py — Profile PPO energy on ALE/Pong-v5 using harness.run().

DQN results already exist in results/dqn_pong/eco-profile.json (700 records).
This script only runs PPO and writes results/ppo_pong/eco-profile.json.
"""

import os
from pathlib import Path

from harness.harness import run
from sb3_models import PPOModel

ENV_NAME = "ALE/Pong-v5"
ROOT = Path(__file__).parent
PPO_OUTDIR = ROOT / "results" / "ppo_pong"

NUM_TRAIN_STEPS = 500
NUM_EVALS = 200


def main() -> None:
    # Clear any partial eco-profile from the aborted previous run.
    stale = PPO_OUTDIR / "eco-profile.json"
    if stale.exists():
        stale.unlink()
        print(f"[measure] Cleared stale {stale}")

    PPO_OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Profiling PPO — {NUM_TRAIN_STEPS} train steps, {NUM_EVALS} eval steps")
    print(f"  env: {ENV_NAME}   outdir: {PPO_OUTDIR}")
    print(f"{'='*60}\n")

    model = PPOModel(seed=42)
    run(
        model=model,
        env_name=ENV_NAME,
        num_train_steps=NUM_TRAIN_STEPS,
        num_evals=NUM_EVALS,
        outdir=str(PPO_OUTDIR),
    )
    model.close()
    print("\n[measure] Done. Results ->", PPO_OUTDIR / "eco-profile.json")


if __name__ == "__main__":
    main()
