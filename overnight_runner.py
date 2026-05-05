#!/usr/bin/env python3
"""
overnight_runner.py — Waits for PPO (bdgyzkfjk) to finish, then runs DQN,
then writes overnight_summary.txt.

Run once and leave it; no user interaction needed.
"""

import subprocess
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PYTHON        = sys.executable                          # D:/miniconda/envs/wm/python.exe
REPO_ROOT     = Path("d:/NU/CS")

PPO_EVAL_FILE = REPO_ROOT / "logs/ppo/ALE-Pong-v5_6/evaluations.npz"
PPO_LOG_DIR   = REPO_ROOT / "logs/ppo/ALE-Pong-v5_6"
PPO_TARGET    = 2_000_000
PPO_DONE_THRESH = 1_975_000   # eval every 25k; last checkpoint >= this means done

DQN_CMD = [
    PYTHON, str(REPO_ROOT / "train_with_zeus.py"),
    "--algo",           "dqn",
    "--env",            "ALE/Pong-v5",
    "--n-timesteps",    "1000000",
    "--output-dir",     str(REPO_ROOT / "results/dqn_pong"),
    "--tensorboard-log", str(REPO_ROOT / "tb/"),
]

SUMMARY_FILE   = REPO_ROOT / "overnight_summary.txt"
PPO_FAIL_LOG   = REPO_ROOT / "ppo_failed.log"
NVIDIA_SMI_OUT = REPO_ROOT / "nvidia_smi_before_dqn.txt"

POLL_INTERVAL  = 30   # seconds between PPO-done checks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ppo_latest_timestep() -> int:
    if not PPO_EVAL_FILE.exists():
        return 0
    data = np.load(str(PPO_EVAL_FILE))
    return int(data["timesteps"][-1])


def any_rl_zoo_running() -> bool:
    """Returns True if a Python rl_zoo3 training process appears active."""
    result = subprocess.run(
        ["powershell", "-c",
         "Get-Process python -ErrorAction SilentlyContinue "
         "| Where-Object {$_.CPU -gt 1} "
         "| Measure-Object | Select-Object -ExpandProperty Count"],
        capture_output=True, text=True
    )
    count = result.stdout.strip()
    return count not in ("", "0")


def ppo_stats() -> dict:
    data = np.load(str(PPO_EVAL_FILE))
    ts   = data["timesteps"]
    rews = data["results"]
    best_idx  = int(np.argmax([r.mean() for r in rews]))
    return {
        "final_mean_reward": round(float(rews[-1].mean()), 2),
        "best_mean_reward":  round(float(rews[best_idx].mean()), 2),
        "best_at_step":      int(ts[best_idx]),
        "final_timestep":    int(ts[-1]),
        "checkpoint_dir":    str(PPO_LOG_DIR),
        "tb_log_dir":        str(REPO_ROOT / "tb/ALE-Pong-v5/PPO_4"),
    }


def run_nvidia_smi() -> None:
    log("Running nvidia-smi ...")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    NVIDIA_SMI_OUT.write_text(result.stdout + result.stderr)
    log(f"nvidia-smi saved -> {NVIDIA_SMI_OUT}")
    # Print first few lines so we can see GPU state in the log
    for line in result.stdout.splitlines()[:15]:
        print(f"  {line}")


def run_dqn() -> tuple[int, str, str, float]:
    """Run DQN. Returns (returncode, stdout, stderr_tail, elapsed_s)."""
    log(f"Starting DQN: {' '.join(str(x) for x in DQN_CMD)}")
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        DQN_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(REPO_ROOT),
    )
    stdout_lines, stderr_lines = [], []
    # Stream stdout; collect stderr separately (non-blocking not needed here)
    import threading
    def collect_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)
    t = threading.Thread(target=collect_stderr, daemon=True)
    t.start()
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        stdout_lines.append(line)
    proc.wait()
    t.join(timeout=5)
    elapsed = time.perf_counter() - t0
    return proc.returncode, "".join(stdout_lines), "".join(stderr_lines), elapsed


def dqn_stats_from_output(stdout: str, elapsed_s: float) -> dict:
    """Parse paths.json written by train_with_zeus.py for the DQN run."""
    import re, glob as _glob
    # train_with_zeus writes to results/dqn_pong/<timestamp>/
    pattern = str(REPO_ROOT / "results/dqn_pong/*/paths.json")
    files = sorted(_glob.glob(pattern))
    paths = json.loads(Path(files[-1]).read_text()) if files else {}

    # Parse eval npz if available
    ckpt = paths.get("checkpoint_dir", "")
    eval_file = Path(ckpt) / "evaluations.npz" if ckpt else None
    final_reward = best_reward = best_step = None
    if eval_file and eval_file.exists():
        data = np.load(str(eval_file))
        ts, rews = data["timesteps"], data["results"]
        best_idx = int(np.argmax([r.mean() for r in rews]))
        final_reward = round(float(rews[-1].mean()), 2)
        best_reward  = round(float(rews[best_idx].mean()), 2)
        best_step    = int(ts[best_idx])

    # Also check energy.json written by train_with_zeus
    energy_files = sorted(_glob.glob(str(REPO_ROOT / "results/dqn_pong/*/energy.json")))
    energy = json.loads(Path(energy_files[-1]).read_text()) if energy_files else {}

    return {
        "final_mean_reward": final_reward,
        "best_mean_reward":  best_reward,
        "best_at_step":      best_step,
        "wall_clock_s":      round(elapsed_s, 1),
        "checkpoint_dir":    ckpt,
        "tb_log_dir":        paths.get("tensorboard_log_dir", ""),
        "energy_json":       energy,
    }


def write_summary(ppo: dict, ppo_rc: int, dqn: dict | None, dqn_rc: int | None,
                  dqn_stderr_tail: str, ppo_wall_s: float) -> None:
    lines = [
        "=" * 60,
        "OVERNIGHT TRAINING SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "--- PPO (ALE/Pong-v5, 2M steps) ---",
        f"  Return code       : {ppo_rc}",
        f"  Final mean reward : {ppo['final_mean_reward']}",
        f"  Best mean reward  : {ppo['best_mean_reward']} (at {ppo['best_at_step']:,} steps)",
        f"  Final timestep    : {ppo['final_timestep']:,}",
        f"  Wall-clock time   : {ppo_wall_s:.0f}s ({ppo_wall_s/60:.1f} min)",
        f"  Checkpoint dir    : {ppo['checkpoint_dir']}",
        f"  TensorBoard log   : {ppo['tb_log_dir']}",
        "",
    ]
    if dqn is not None:
        lines += [
            "--- DQN (ALE/Pong-v5, 1M steps) ---",
            f"  Return code       : {dqn_rc}",
            f"  Final mean reward : {dqn['final_mean_reward']}",
            f"  Best mean reward  : {dqn['best_mean_reward']} (at {dqn.get('best_at_step')} steps)",
            f"  Wall-clock time   : {dqn['wall_clock_s']}s ({dqn['wall_clock_s']/60:.1f} min)",
            f"  Checkpoint dir    : {dqn['checkpoint_dir']}",
            f"  TensorBoard log   : {dqn['tb_log_dir']}",
            f"  Energy report     : {dqn['energy_json']}",
        ]
        if dqn_rc != 0 and dqn_stderr_tail:
            lines += [
                "",
                "  --- DQN stderr (last 50 lines) ---",
            ] + [f"  {l}" for l in dqn_stderr_tail.splitlines()[-50:]]
    else:
        lines += [
            "--- DQN ---",
            "  NOT STARTED (PPO failed — see ppo_failed.log)",
        ]
    lines += ["", "=" * 60]
    SUMMARY_FILE.write_text("\n".join(lines))
    log(f"Summary written -> {SUMMARY_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("overnight_runner started — waiting for PPO to finish ...")

    # --- Wait for PPO ---
    ppo_start_wait = time.perf_counter()
    while True:
        latest = ppo_latest_timestep()
        still_running = any_rl_zoo_running()
        log(f"PPO eval latest={latest:,}  rl_zoo_running={still_running}")

        if latest >= PPO_DONE_THRESH and not still_running:
            log("PPO appears complete (eval reached target, no active process).")
            break
        if not still_running and latest < PPO_DONE_THRESH:
            log(f"ERROR: No rl_zoo process running but PPO only reached {latest:,} steps — assumed FAILED.")
            err_msg = (f"PPO failed or was killed.\n"
                       f"Latest eval timestep: {latest:,} (expected >= {PPO_DONE_THRESH:,})\n"
                       f"Detected at: {datetime.now()}\n")
            PPO_FAIL_LOG.write_text(err_msg)
            write_summary(
                ppo={"final_mean_reward": None, "best_mean_reward": None,
                     "best_at_step": None, "final_timestep": latest,
                     "checkpoint_dir": str(PPO_LOG_DIR),
                     "tb_log_dir": str(REPO_ROOT / "tb/ALE-Pong-v5/PPO_4")},
                ppo_rc=1, dqn=None, dqn_rc=None,
                dqn_stderr_tail="", ppo_wall_s=time.perf_counter() - ppo_start_wait,
            )
            sys.exit(1)
        time.sleep(POLL_INTERVAL)

    ppo_wall_s_approx = time.perf_counter() - ppo_start_wait  # rough (runner started mid-PPO)
    ppo = ppo_stats()
    log(f"PPO final: mean={ppo['final_mean_reward']}  best={ppo['best_mean_reward']} @ {ppo['best_at_step']:,}")

    # --- nvidia-smi ---
    run_nvidia_smi()

    # --- Run DQN ---
    dqn_rc, dqn_stdout, dqn_stderr, dqn_wall_s = run_dqn()
    log(f"DQN finished: returncode={dqn_rc}  wall={dqn_wall_s:.0f}s")

    dqn = dqn_stats_from_output(dqn_stdout, dqn_wall_s)

    # --- Summary ---
    write_summary(ppo, ppo_rc=0, dqn=dqn, dqn_rc=dqn_rc,
                  dqn_stderr_tail=dqn_stderr, ppo_wall_s=ppo_wall_s_approx)
    log("All done.")


if __name__ == "__main__":
    main()
