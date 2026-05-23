from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def layer_init(layer: nn.Module, gain: float = np.sqrt(2), bias_const: float = 0.0):
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low,
        action_high,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.layer1 = layer_init(nn.Linear(obs_dim, hidden_size), gain=np.sqrt(2))
        self.layer2 = layer_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2))
        self.mean = layer_init(nn.Linear(hidden_size, action_dim), gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        action_scale = (np.asarray(action_high) - np.asarray(action_low)) / 2.0
        action_bias = (np.asarray(action_high) + np.asarray(action_low)) / 2.0

        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x: torch.Tensor):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)

        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_deterministic_action(self, x: torch.Tensor):
        mean, _ = self.forward(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_logprob_entropy(self, x: torch.Tensor, action: torch.Tensor):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)

        y = (action - self.action_bias) / self.action_scale
        y = torch.clamp(y, -0.999999, 0.999999)
        x_t = 0.5 * (torch.log1p(y) - torch.log1p(-y))

        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__()

        self.layer1 = layer_init(nn.Linear(obs_dim, hidden_size), gain=np.sqrt(2))
        self.layer2 = layer_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2))
        self.v = layer_init(nn.Linear(hidden_size, 1), gain=1.0)

    def forward(self, x: torch.Tensor):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.v(x)
