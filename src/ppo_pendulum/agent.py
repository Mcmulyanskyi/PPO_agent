from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import RolloutBuffer
from .config import PPOConfig
from .models import Policy, ValueNetwork


@dataclass
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    approx_kl: float
    clip_fraction: float


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low,
        action_high,
        config: PPOConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device

        self.policy = Policy(
            obs_dim,
            action_dim,
            action_low,
            action_high,
            hidden_size=config.hidden_size,
        ).to(device)
        self.value = ValueNetwork(obs_dim, hidden_size=config.hidden_size).to(device)

        self.optimizer = optim.AdamW(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=config.learning_rate,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )

    def act(self, obs: np.ndarray):
        obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, logp_tensor, _ = self.policy.get_action(obs_tensor)
            value_tensor = self.value(obs_tensor)

        return (
            action_tensor.cpu().numpy(),
            logp_tensor.cpu().numpy(),
            value_tensor.cpu().numpy(),
        )

    def deterministic_action(self, obs: np.ndarray):
        obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.get_deterministic_action(obs_tensor)
        return action.cpu().numpy()

    def set_learning_rate(self, current_lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

    def update(self, buffer: RolloutBuffer, returns: np.ndarray, advantages: np.ndarray):
        cfg = self.config
        device = self.device

        b_obs = torch.from_numpy(buffer.obs).float().to(device)
        b_actions = torch.from_numpy(buffer.actions).float().to(device)
        b_oldlogp = torch.from_numpy(buffer.logprobs).float().to(device)
        b_returns = torch.from_numpy(returns.reshape(-1, 1)).float().to(device)
        b_values = torch.from_numpy(buffer.values).float().to(device)
        b_advantages = torch.from_numpy(advantages.reshape(-1, 1)).float().to(device)

        batch_size = cfg.rollout_steps
        indices = np.arange(batch_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_minibatches = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = indices[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_oldlogp = b_oldlogp[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / (mb_advantages.std() + 1e-8)

                new_logp, entropy_each = self.policy.get_logprob_entropy(mb_obs, mb_actions)
                new_values = self.value(mb_obs).view(-1, 1)

                logratio = new_logp - mb_oldlogp
                ratio = torch.exp(logratio)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                v_loss_unclipped = (new_values - mb_returns).pow(2)
                v_clipped = mb_values + torch.clamp(
                    new_values - mb_values, -cfg.clip_eps, cfg.clip_eps
                )
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = -cfg.ent_coef * entropy_each.mean()
                loss = policy_loss + cfg.vf_coef * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                num_minibatches += 1

        denom = max(num_minibatches, 1)
        return PPOUpdateStats(
            policy_loss=total_policy_loss / denom,
            value_loss=total_value_loss / denom,
            entropy_loss=total_entropy_loss / denom,
            total_loss=total_loss / denom,
            approx_kl=total_approx_kl / denom,
            clip_fraction=total_clip_fraction / denom,
        )

    def state_dict(self) -> Dict:
        return {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, checkpoint: Dict, load_optimizer: bool = False) -> None:
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
