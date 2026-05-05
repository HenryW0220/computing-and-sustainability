#!/usr/bin/env python3
"""
train_with_zeus.py — RL training wrapper with Zeus energy profiling hooks.

Usage:
    python train_with_zeus.py \\
        --algo ppo --env ALE/Pong-v5 \\
        --n-timesteps 2000000 \\
        --output-dir ./results/run_001 \\
        --hyperparams clip_range:0.1
"""

# WARNING: subprocess approach captures only end-to-end energy. For Idea 1
# (training vs inference separation), this must be refactored to use the SB3
# API directly with ZeusMonitor windows around model.learn() and
# evaluate_policy() separately. Week 6 task.

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL training with Zeus energy profiling")
    p.add_argument("--algo",         type=str, required=True,
                   help="RL algorithm (e.g. ppo, a2c, dqn)")
    p.add_argument("--env",          type=str, required=True,
                   help="Gym environment ID (e.g. ALE/Pong-v5)")
    p.add_argument("--n-timesteps",  type=int, default=1_000_000,
                   help="Total training timesteps")
    p.add_argument("--output-dir",   type=str, required=True,
                   help="Directory to write energy.json / metadata.json / paths.json")
    p.add_argument("--hyperparams",  type=str, nargs="+", default=[],
                   help="Hyperparameter overrides forwarded to rl_zoo3 (e.g. clip_range:0.1)")
    p.add_argument("--tensorboard-log", type=str, default="./tb/",
                   help="TensorBoard log root directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        return "unknown"


def collect_metadata(args: argparse.Namespace) -> dict:
    """Collect system and run metadata for reproducibility."""
    import torch

    hp_hash = hashlib.sha256(
        json.dumps(sorted(args.hyperparams)).encode()
    ).hexdigest()[:12]

    return {
        "algo":                 args.algo,
        "env":                  args.env,
        "n_timesteps":          args.n_timesteps,
        "hyperparams_overrides": args.hyperparams,
        "hyperparams_hash":     hp_hash,
        "git_commit":           _git_commit(),
        "torch_version":        torch.__version__,
        "cuda_version":         torch.version.cuda or "N/A",
        "gpu_name":             (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "cpu"),
    }


# ---------------------------------------------------------------------------
# rl_zoo3 subprocess
# ---------------------------------------------------------------------------

def build_zoo_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m", "rl_zoo3.train",
        "--algo",            args.algo,
        "--env",             args.env,
        "-n",                str(args.n_timesteps),
        "--progress",
        "--tensorboard-log", args.tensorboard_log,
    ]
    if args.hyperparams:
        cmd += ["--hyperparams"] + args.hyperparams
    return cmd


def run_zoo(cmd: list[str]) -> tuple[int, str]:
    """Stream rl_zoo3 stdout to the terminal and capture it for parsing."""
    lines: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def parse_checkpoint_path(zoo_output: str) -> str | None:
    """Extract the checkpoint dir from rl_zoo3's 'Log path: ...' line."""
    m = re.search(r"Log path:\s*(\S+)", zoo_output)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_energy(output_dir: Path, args: argparse.Namespace,
                 elapsed_s: float,
                 total_energy_J: float | None,
                 avg_power_W: float | None) -> None:
    data = {
        "env":           args.env,
        "algo":          args.algo,
        "n_timesteps":   args.n_timesteps,
        "time_s":        round(elapsed_s, 2),
        "total_energy_J": total_energy_J,  # None until ZeusMonitor integrated
        "avg_power_W":    avg_power_W,     # None until ZeusMonitor integrated
    }
    path = output_dir / "energy.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"[zeus] energy.json   -> {path}")


def write_metadata(output_dir: Path, metadata: dict) -> None:
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    print(f"[zeus] metadata.json -> {path}")


def write_paths(output_dir: Path, checkpoint_path: str | None,
                tensorboard_log: str) -> None:
    data = {
        "checkpoint_dir":     checkpoint_path,
        "tensorboard_log_dir": str(Path(tensorboard_log).resolve()),
    }
    path = output_dir / "paths.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"[zeus] paths.json    -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = collect_metadata(args)

    # ------------------------------------------------------------------
    # TODO: integrate ZeusMonitor here — begin energy window
    # from zeus.monitor import ZeusMonitor
    # monitor = ZeusMonitor(gpu_indices=[0])
    # monitor.begin_window("training")
    # ------------------------------------------------------------------
    t_start = time.perf_counter()

    cmd = build_zoo_command(args)
    print(f"[zeus] Command: {' '.join(cmd)}\n")
    returncode, zoo_output = run_zoo(cmd)

    elapsed_s = time.perf_counter() - t_start
    # ------------------------------------------------------------------
    # TODO: integrate ZeusMonitor here — end energy window
    # measurement = monitor.end_window("training")
    # total_energy_J = measurement.total_energy
    # avg_power_W    = total_energy_J / measurement.time  # use measurement.time, not elapsed_s
    total_energy_J: float | None = None
    avg_power_W:    float | None = None
    # ------------------------------------------------------------------

    if returncode != 0:
        print(f"[zeus] Training exited with code {returncode}", file=sys.stderr)
        sys.exit(returncode)

    checkpoint_path = parse_checkpoint_path(zoo_output)

    write_energy(output_dir, args, elapsed_s, total_energy_J, avg_power_W)
    write_metadata(output_dir, metadata)
    write_paths(output_dir, checkpoint_path, args.tensorboard_log)

    print(f"\n[zeus] Finished in {elapsed_s:.1f}s")
    print(f"  Checkpoint : {checkpoint_path or '(not found in output)'}")
    print(f"  TensorBoard: {args.tensorboard_log}")
    print(f"  Output dir : {output_dir}")


if __name__ == "__main__":
    main()
