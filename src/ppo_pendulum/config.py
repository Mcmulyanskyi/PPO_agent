from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PPOConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 42

    total_timesteps: int = 1_500_000
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64

    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.05
    vf_coef: float = 1.0
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    hidden_size: int = 128
    weight_decay: float = 0.01
    adam_eps: float = 1e-5

    norm_obs: bool = True
    norm_reward: bool = False
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "ppo_pendulum.pt"
    vecnormalize_name: str = "vecnormalize.pkl"
    save_every_updates: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> PPOConfig:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return PPOConfig(**data)
