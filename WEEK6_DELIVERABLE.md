# Week 6 Deliverable — Energy Profiling: DQN vs PPO on ALE/Pong-v5

## Objective

Instrument DQN and PPO training with fine-grained GPU energy measurement using the Zeus
`ZeusMonitor` harness, and fill in the "Energy per step" row of the Week 5 efficiency
comparison table with real numbers.

---

## Setup

**Hardware:** NVIDIA GeForce RTX 4070 Ti (285 W TDP, 12 GB VRAM)  
**Framework:** Stable-Baselines3 + ALE/Pong-v5 (84×84 grayscale, 4-frame stack)  
**Energy measurement:** Zeus `ZeusMonitor` (`approx_instant_energy=True`) via NVML power polling

### Harness Interface (`harness/harness.py`)

The profiling harness defines a `Model` abstract class with three methods:

```python
def setup(env_name: str)   # build env + model
def train_step()           # one collect + gradient-update cycle
def eval(inp)              # one inference pass
```

`harness.run()` wraps each call in an `energy_window` context manager and appends one
JSON record per step to `eco-profile.json`:

```json
{"label": "training:step=0", "joules": 340.57, "seconds": 6.05}
{"label": "inference:step=0", "joules": 25.12, "seconds": 0.48}
```

---

## Implementation (`sb3_models.py`)

| | DQNModel | PPOModel |
|---|---|---|
| Step size | 256 env steps | 2048 env steps |
| Gradient steps / train_step | 64 (train_freq=4) | ~32 (4 epochs × 8 batches) |
| Replay buffer | 100k transitions | — (on-policy) |
| Eval step | 200 greedy env steps | 200 greedy env steps |

Both models use `CnnPolicy` on GPU. Each `train_step()` calls
`model.learn(total_timesteps=STEP_SIZE, reset_num_timesteps=False)`.

---

## Results

### Raw measurements (500 train steps, 200 eval steps each)

| Metric | DQN | PPO |
|---|---|---|
| Mean energy / train step | **47.3 J** | **340.6 J** |
| Mean time / train step | 0.83 s | 6.05 s |
| Mean energy / eval step | 23.6 J | 25.1 J |
| Mean time / eval step | 0.45 s | 0.48 s |
| Total train energy (500 steps) | 23,651 J (6.6 Wh) | 170,285 J (47.3 Wh) |

### Normalized by env steps

The raw "J/step" numbers are not directly comparable because each algorithm's
step covers a different number of environment frames (256 vs 2048).

| Metric | DQN | PPO |
|---|---|---|
| Env steps per train_step | 256 | 2048 |
| J per 1,000 env steps | **184.8 J** | **166.3 J** |
| Env steps covered (500 steps) | 128,000 | 1,024,000 |

When normalized to env steps, DQN and PPO consume nearly identical GPU energy
(~175 J / 1k env steps). The apparent 7.2× gap in raw J/step comes entirely
from PPO processing 8× more env frames per step.

### Updated efficiency table

| Metric | PPO | DQN | DQN vs PPO |
|---|---|---|---|
| Steps to best reward | 1.975 M | 0.950 M | DQN uses 48% of PPO steps |
| Wall-clock to best reward | ~89 min | ~53 min | DQN uses 60% of PPO time |
| Best reward achieved | −16.2 | −9.8 | DQN +6.4 reward higher |
| **Energy / train step (J/step)** | **340.6 J** | **47.3 J** | **7.2× (raw); ~1× normalized** |
| **Energy / eval step (J/step)** | **25.1 J** | **23.6 J** | **DQN ≈ PPO** |

---

## Analysis

**1. Training energy is dominated by gradient computation, not env steps.**  
PPO's higher J/step is a direct consequence of its update rule (multiple epochs over
a large rollout buffer). DQN's smaller step and single gradient update per 4 frames
is cheaper per call, but not per frame of experience.

**2. Inference cost is algorithm-agnostic.**  
Both models run the same CNN forward pass with no gradient tape, so eval energy
is nearly identical (~24–25 J per 200-step eval chunk, ~0.12 J per env step).

**3. DQN is the greener choice on this task.**  
DQN reached a better policy (−9.8 vs −16.2) using fewer env steps, less
wall-clock time, and less total GPU energy (6.6 Wh vs 47.3 Wh over the
measured 500-step window). For Pong specifically, the off-policy replay buffer
gives DQN a clear sample-efficiency advantage that translates directly into
lower energy cost per unit of performance.

**4. Limitation — measurement granularity.**  
`approx_instant_energy=True` uses NVML power polling (~100 ms resolution).
Steps shorter than ~200 ms would have noisy readings; DQN's 0.83 s/step and
PPO's 6.05 s/step are both well above this threshold. CPU and DRAM energy
are not captured (Zeus RAPL is unsupported on this machine).

---

## Artifacts

| File | Description |
|---|---|
| `harness/harness.py` | Zeus-based profiling harness |
| `sb3_models.py` | DQNModel and PPOModel implementations |
| `measure_energy.py` | Measurement runner |
| `update_week5_figure.py` | Figure regeneration script |
| `results/dqn_pong/eco-profile.json` | 700 records (500 train + 200 eval) |
| `results/ppo_pong/eco-profile.json` | 700 records (500 train + 200 eval) |
| `figures/week5_efficiency_comparison.png` | Updated comparison table |
