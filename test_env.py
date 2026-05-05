import torch
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# --- 1. CUDA check ---
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

# --- 2. Register ALE envs ---
gym.register_envs(ale_py)
print(f"\nALE/Pong-v5 registered: {'ALE/Pong-v5' in gym.envs.registry}")

# --- 3. Build vectorized Atari env (4 parallel) ---
env = make_atari_env("ALE/Pong-v5", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
print(f"Obs space  : {env.observation_space}")
print(f"Action space: {env.action_space}")

# --- 4. PPO agent on GPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO(
    "CnnPolicy",
    env,
    device=device,
    verbose=1,
    n_steps=128,
    batch_size=256,
)
print(f"\nPPO device  : {model.device}")

# --- 5. Train 1000 steps ---
print("\nTraining for 1000 steps ...")
model.learn(total_timesteps=1000)

env.close()
print("\n=== All checks passed ===")
