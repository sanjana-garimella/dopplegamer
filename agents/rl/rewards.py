"""Reward shapers for the BC+RL combined agent.

The dual-reward formulation blends competitive reward (env reward, +1/-1/0) with
a behavioral-cloning bonus that rewards matching a reference policy's move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DualRewardConfig:
    alpha: float = 0.6  # weight on env (win) reward
    beta: float = 0.4   # weight on behavioral cloning bonus

    def __post_init__(self) -> None:
        s = self.alpha + self.beta
        if s <= 0:
            raise ValueError("alpha + beta must be positive")
        self.alpha /= s
        self.beta /= s


def bc_bonus(action: int, reference_logits: np.ndarray) -> float:
    """Log-prob of `action` under the reference policy. Returns 0 if index is out of bounds."""
    if action < 0 or action >= len(reference_logits):
        return 0.0
    p = np.exp(reference_logits - reference_logits.max())
    p = p / p.sum()
    p = np.clip(p, 1e-6, 1.0)
    return float(np.log(p[action]))


def combine(env_reward: float, action: int, reference_logits: np.ndarray | None, cfg: DualRewardConfig) -> float:
    if reference_logits is None:
        return env_reward
    return cfg.alpha * env_reward + cfg.beta * bc_bonus(action, reference_logits)
