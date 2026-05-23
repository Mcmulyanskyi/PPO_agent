from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int, action_dim: int):
        self.rollout_steps = rollout_steps
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, action_dim), dtype=np.float32)
        self.logprobs = np.zeros((rollout_steps, 1), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps,), dtype=np.float32)
        self.dones = np.zeros((rollout_steps,), dtype=np.float32)
        self.values = np.zeros((rollout_steps, 1), dtype=np.float32)

    def add(self, index: int, obs, action, logprob, reward, done, value) -> None:
        self.obs[index] = np.asarray(obs, dtype=np.float32)
        self.actions[index] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.logprobs[index] = np.asarray(logprob, dtype=np.float32).reshape(1)
        self.rewards[index] = float(reward)
        self.dones[index] = float(done)
        self.values[index] = np.asarray(value, dtype=np.float32).reshape(1)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ):
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1, 0]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_nonterminal
                - self.values[t, 0]
            )
            last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + self.values.flatten()
        return returns.astype(np.float32), advantages.astype(np.float32)
