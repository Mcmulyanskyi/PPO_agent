from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .agent import PPOAgent
from .config import PPOConfig
from .envs import make_eval_env
from .utils import get_device, seed_everything


def evaluate(
    config: PPOConfig,
    checkpoint_path: str | Path,
    vecnormalize_path: str | Path | None = None,
    episodes: int = 10,
    render: bool = False,
):
    checkpoint_path = Path(checkpoint_path)
    if vecnormalize_path is not None:
        vecnormalize_path = Path(vecnormalize_path)

    device = get_device()
    render_mode = "human" if render else None
    env = make_eval_env(config, vecnormalize_path=vecnormalize_path, render_mode=render_mode)
    seed_everything(config.seed, env)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(
        obs_dim,
        action_dim,
        env.action_space.low,
        env.action_space.high,
        config,
        device,
    )
    agent.load_state_dict(checkpoint, load_optimizer=False)
    agent.policy.eval()
    agent.value.eval()

    rewards = []

    try:
        for episode in range(episodes):
            obs_batch = env.reset()
            obs = obs_batch.squeeze(0).astype(np.float32)
            done = False
            episode_reward = 0.0

            while not done:
                action = agent.deterministic_action(obs)
                obs_batch, reward_batch, done_batch, _ = env.step(action)

                obs = obs_batch.squeeze(0).astype(np.float32)
                reward = float(reward_batch[0])
                done = bool(done_batch[0])
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"eval_episode={episode + 1}, return={episode_reward:.3f}")

    finally:
        env.close()

    print(f"mean_return={np.mean(rewards):.3f}, std_return={np.std(rewards):.3f}")
    return rewards
