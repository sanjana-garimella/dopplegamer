"""Gymnasium wrapper that clamps illegal actions before env.step.

RPS+ (and other arena envs) raise on illegal moves. SB3 policies sample from
the full Discrete(n) space, so training needs this clamp.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class LegalActionWrapper(gym.Wrapper):
    """If action is illegal, replace with the first legal move (or 0)."""

    def _legal(self) -> list[int]:
        legal_actions = getattr(self.env, "legal_actions", None)
        if callable(legal_actions):
            return [int(m) for m in legal_actions()]
        legal_fn = getattr(self.env, "legal_moves", None)
        if callable(legal_fn):
            try:
                return [int(m) for m in legal_fn()]
            except TypeError:
                return [int(m) for m in legal_fn(True)]
        return []

    def step(self, action: Any):
        action = int(action)
        legal = self._legal()
        if legal and action not in legal:
            action = legal[0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(info, dict):
            info = {**info, "action": action}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
