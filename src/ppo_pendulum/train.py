from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .agent import PPOAgent
from .buffer import RolloutBuffer
from .config import PPOConfig
from .envs import make_train_env
from .utils import ensure_dir, get_device, seed_everything


def linear_lr(
    initial_lr: float,
    min_lr: float,
    global_step: int,
    total_timesteps: int,
) -> float:
    progress = min(global_step / total_timesteps, 1.0)
    return max(initial_lr * (1.0 - progress), min_lr)


def save_checkpoint(
    agent: PPOAgent,
    env,
    config: PPOConfig,
    checkpoint_path: Path,
    vecnormalize_path: Path,
    global_step: int,
) -> None:
    payload = {
        **agent.state_dict(),
        "config": config.to_dict(),
        "global_step": global_step,
    }
    torch.save(payload, checkpoint_path)

    if hasattr(env, "save"):
        env.save(str(vecnormalize_path))


def train(config: PPOConfig) -> Path:
    device = get_device()
    env = make_train_env(config)
    seed_everything(config.seed, env)

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

    experiment_name = f"PPO_Pendulum_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(config.log_dir, experiment_name))

    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    checkpoint_path = checkpoint_dir / config.checkpoint_name
    vecnormalize_path = checkpoint_dir / config.vecnormalize_name

    buffer = RolloutBuffer(config.rollout_steps, obs_dim, action_dim)

    obs_batch = env.reset()
    state = obs_batch.squeeze(0).astype(np.float32)

    global_step = 0
    update_idx = 0
    episode_reward = 0.0
    episode_length = 0
    all_episode_rewards: list[float] = []

    try:
        while global_step < config.total_timesteps:
            update_idx += 1

            for t in range(config.rollout_steps):
                global_step += 1

                action, logp, value = agent.act(state)
                next_obs_batch, reward_batch, done_batch, _ = env.step(action)

                next_state = next_obs_batch.squeeze(0).astype(np.float32)
                reward = float(reward_batch[0])
                done = bool(done_batch[0])

                buffer.add(
                    index=t,
                    obs=state,
                    action=action,
                    logprob=logp,
                    reward=reward,
                    done=done,
                    value=value,
                )

                episode_reward += reward
                episode_length += 1

                if done:
                    all_episode_rewards.append(episode_reward)
                    writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    print(
                        f"global_step={global_step}, "
                        f"episodic_return={episode_reward:.3f}, "
                        f"episodic_length={episode_length}"
                    )

                    episode_reward = 0.0
                    episode_length = 0

                    next_state = env.reset().squeeze(0).astype(np.float32)

                state = next_state

                if global_step >= config.total_timesteps:
                    break

            state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
            with torch.no_grad():
                last_value = agent.value(state_tensor).cpu().numpy().flatten()[0]

            returns, advantages = buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            current_lr = linear_lr(
                config.learning_rate,
                config.min_learning_rate,
                global_step,
                config.total_timesteps,
            )
            agent.set_learning_rate(current_lr)
            writer.add_scalar("charts/learning_rate", current_lr, global_step)

            stats = agent.update(buffer, returns, advantages)

            writer.add_scalar("losses/policy_loss", stats.policy_loss, global_step)
            writer.add_scalar("losses/value_loss", stats.value_loss, global_step)
            writer.add_scalar("losses/entropy_loss", stats.entropy_loss, global_step)
            writer.add_scalar("losses/total_loss", stats.total_loss, global_step)
            writer.add_scalar("losses/approx_kl", stats.approx_kl, global_step)
            writer.add_scalar("losses/clip_fraction", stats.clip_fraction, global_step)

            if all_episode_rewards:
                avg_reward = float(np.mean(all_episode_rewards[-10:]))
                writer.add_scalar("charts/average_10_episode_reward", avg_reward, global_step)
                print(
                    f"step={global_step}, "
                    f"avg_last_10_ep_reward={avg_reward:.1f}, "
                    f"policy_loss={stats.policy_loss:.4f}, "
                    f"value_loss={stats.value_loss:.4f}, "
                    f"lr={current_lr:.6f}"
                )

            if update_idx % config.save_every_updates == 0:
                save_checkpoint(
                    agent=agent,
                    env=env,
                    config=config,
                    checkpoint_path=checkpoint_path,
                    vecnormalize_path=vecnormalize_path,
                    global_step=global_step,
                )

    finally:
        save_checkpoint(
            agent=agent,
            env=env,
            config=config,
            checkpoint_path=checkpoint_path,
            vecnormalize_path=vecnormalize_path,
            global_step=global_step,
        )
        writer.close()
        env.close()

    return checkpoint_path
