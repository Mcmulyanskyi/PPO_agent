from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import PPOConfig


def make_raw_env(env_id: str, render_mode: str | None = None):
    return gym.make(env_id, render_mode=render_mode)


def make_train_env(config: PPOConfig):
    raw_env = make_raw_env(config.env_id)
    env = DummyVecEnv([lambda: raw_env])

    if config.norm_obs or config.norm_reward:
        env = VecNormalize(
            env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=config.clip_obs,
            clip_reward=config.clip_reward,
        )

    return env


def make_eval_env(
    config: PPOConfig,
    vecnormalize_path: str | Path | None = None,
    render_mode: str | None = None,
):
    raw_env = make_raw_env(config.env_id, render_mode=render_mode)
    env = DummyVecEnv([lambda: raw_env])

    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
    elif config.norm_obs or config.norm_reward:
        # Evaluation without saved normalization stats is possible,
        # but it is not equivalent to training.
        env = VecNormalize(
            env,
            norm_obs=config.norm_obs,
            norm_reward=False,
            clip_obs=config.clip_obs,
            clip_reward=config.clip_reward,
        )
        env.training = False

    return env
