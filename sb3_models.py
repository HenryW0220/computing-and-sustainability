"""DQNModel and PPOModel — harness.Model subclasses for SB3 on ALE/Pong-v5."""

import warnings

import ale_py
import gymnasium as gym

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from harness.harness import Model

gym.register_envs(ale_py)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_env(env_name: str, n_envs: int = 1, seed: int = 0) -> VecFrameStack:
    env = make_atari_env(env_name, n_envs=n_envs, seed=seed)
    return VecFrameStack(env, n_stack=4)


class DQNModel(Model):
    """One train_step = 256 env steps + gradient updates; one eval = 200 inference steps."""

    STEP_SIZE = 256
    EVAL_STEPS = 200

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._env = None
        self._eval_env = None
        self._model = None
        self._obs = None
        self._first_step = True

    @property
    def name(self) -> str:
        return "dqn"

    def setup(self, env_name: str, **kwargs) -> None:
        self._env = _make_env(env_name, n_envs=1, seed=self._seed)
        self._eval_env = _make_env(env_name, n_envs=1, seed=self._seed + 1)
        self._model = DQN(
            "CnnPolicy",
            self._env,
            device=_DEVICE,
            verbose=0,
            seed=self._seed,
            buffer_size=100_000,
            learning_starts=self.STEP_SIZE,
            batch_size=32,
            train_freq=4,
            gradient_steps=1,
        )
        self._obs = self._eval_env.reset()

    def train_step(self) -> None:
        self._model.learn(
            total_timesteps=self.STEP_SIZE,
            reset_num_timesteps=self._first_step,
            progress_bar=False,
            log_interval=None,
        )
        self._first_step = False

    def eval(self, inp=None) -> None:
        obs = self._obs
        for _ in range(self.EVAL_STEPS):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, done, _ = self._eval_env.step(action)
            if done.any():
                obs = self._eval_env.reset()
        self._obs = obs

    def close(self) -> None:
        if self._env:
            self._env.close()
        if self._eval_env:
            self._eval_env.close()


class PPOModel(Model):
    """One train_step = one full PPO rollout (2048 steps); one eval = 200 inference steps."""

    STEP_SIZE = 2048
    EVAL_STEPS = 200

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._env = None
        self._eval_env = None
        self._model = None
        self._obs = None
        self._first_step = True

    @property
    def name(self) -> str:
        return "ppo"

    def setup(self, env_name: str, **kwargs) -> None:
        self._env = _make_env(env_name, n_envs=1, seed=self._seed)
        self._eval_env = _make_env(env_name, n_envs=1, seed=self._seed + 1)
        self._model = PPO(
            "CnnPolicy",
            self._env,
            device=_DEVICE,
            verbose=0,
            seed=self._seed,
            n_steps=self.STEP_SIZE,
            batch_size=256,
            n_epochs=4,
        )
        self._obs = self._eval_env.reset()

    def train_step(self) -> None:
        self._model.learn(
            total_timesteps=self.STEP_SIZE,
            reset_num_timesteps=self._first_step,
            progress_bar=False,
            log_interval=None,
        )
        self._first_step = False

    def eval(self, inp=None) -> None:
        obs = self._obs
        for _ in range(self.EVAL_STEPS):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, done, _ = self._eval_env.step(action)
            if done.any():
                obs = self._eval_env.reset()
        self._obs = obs

    def close(self) -> None:
        if self._env:
            self._env.close()
        if self._eval_env:
            self._eval_env.close()
