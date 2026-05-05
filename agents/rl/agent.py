"""Inference-time wrappers for trained PPO / BC+RL checkpoints.

PPOAgent      — pure win-rate policy loaded from a Stable-Baselines3 checkpoint.
BCRLAgent     — BC+RL checkpoint that distinguishes itself from PPO by running
                the dual-reward policy: it tracks recent win/loss history and
                adaptively blends an energy-conservative BC prior with the
                aggressive RL objective, mirroring the alpha/beta training signal.

Both fall back to rule-based heuristics when no checkpoint path is provided,
so they can run in benchmark mode without a trained model file.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from agents.base import Agent
from agents.checkpoints import resolve_checkpoint
from agents.rl.rewards import DualRewardConfig, bc_bonus
from environments.rps_plus import Move, N_MOVES


# ---------------------------------------------------------------------------
# PPO Agent — pure competitive policy
# ---------------------------------------------------------------------------

class PPOAgent(Agent):
    """Win-rate-first policy loaded from a Stable-Baselines3 PPO checkpoint.

    If no checkpoint is provided, uses a rule-based approximation:
    prefer POWER when energy >= 2, otherwise pick an aggressive base move.
    """

    name = "ppo"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._model = None
        self.rng = np.random.default_rng(seed)
        self.checkpoint_path = resolve_checkpoint("ppo") if checkpoint is None else Path(checkpoint)

        if self.checkpoint_path is not None:
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(str(self.checkpoint_path))
            except Exception:
                # Graceful fallback to heuristic if checkpoint unavailable.
                self._model = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = set(info.get("legal_moves") or list(range(N_MOVES)))

        if self._model is not None:
            try:
                action, _ = self._model.predict(obs, deterministic=True)
                a = int(action)
                return a if a in legal else int(self.rng.choice(sorted(legal)))
            except Exception:
                pass

        # Heuristic fallback: POWER when energy allows, else aggressive base moves.
        agent_energy = int(round(obs[0] * 5))
        if agent_energy >= 2 and int(Move.POWER) in legal:
            return int(Move.POWER)
        aggressive = [m for m in [Move.ROCK, Move.PAPER, Move.SCISSORS] if int(m) in legal]
        if aggressive:
            return int(self.rng.choice([int(m) for m in aggressive]))
        return int(self.rng.choice(sorted(legal)))


# ---------------------------------------------------------------------------
# BC+RL Agent — blended behavioral-cloning + competitive policy
# ---------------------------------------------------------------------------


class BCRLAgent(Agent):
    """BC+RL policy that blends competitive reward with behavioral-cloning fidelity.

    Key differences from PPOAgent:
    - Tracks recent win/loss history to dynamically adjust the alpha/beta blend.
    - When winning: leans more heavily on the competitive (RL) objective.
    - When losing: leans toward the stable BC prior to recover composure.
    - Uses an energy-aware BC prior that mirrors human-style energy management.
    """

    name = "bc_rl"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        history_window: int = 10,
        bc_prior: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        self.base_cfg = DualRewardConfig(alpha=alpha, beta=beta)
        self.rng = np.random.default_rng(seed)
        self._recent_outcomes: deque[int] = deque(maxlen=history_window)
        self._last_opp: int | None = None
        self._model = None
        self.bc_prior = bc_prior if bc_prior is not None else np.array([
            0.20,   # ROCK
            0.20,   # PAPER
            0.20,   # SCISSORS
            0.10,   # LIZARD
            0.15,   # POWER
            0.15,   # RECHARGE
        ], dtype=np.float64)
        self.bc_prior /= self.bc_prior.sum()
        self.checkpoint_path = resolve_checkpoint("bc_rl") if checkpoint is None else Path(checkpoint)

        if self.checkpoint_path is not None:
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(str(self.checkpoint_path))
            except Exception:
                self._model = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._recent_outcomes.clear()
        self._last_opp = None

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self._recent_outcomes.append(int(outcome))
        self._last_opp = int(opponent_move)

    def _effective_cfg(self) -> DualRewardConfig:
        """Dynamically adjust alpha/beta based on recent performance."""
        if not self._recent_outcomes:
            return self.base_cfg
        recent = list(self._recent_outcomes)
        win_rate = sum(1 for o in recent if o > 0) / len(recent)
        # If winning: more RL weight (exploit). If losing: more BC (stabilise).
        alpha = 0.5 + 0.3 * (win_rate - 0.5)   # range [0.35, 0.65]
        beta  = 1.0 - alpha
        return DualRewardConfig(alpha=alpha, beta=beta)

    def _bc_logits(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """BC prior: energy-aware human-style reference logits."""
        agent_energy = int(round(obs[0] * 5))
        logits = np.log(self.bc_prior + 1e-8)
        # Human-style: recharge more when low, power more when high
        if agent_energy <= 1:
            logits[int(Move.RECHARGE)] += 1.5
            logits[int(Move.POWER)]    -= 2.0
        elif agent_energy >= 4:
            logits[int(Move.POWER)]    += 1.0
            logits[int(Move.RECHARGE)] -= 1.0
        return logits

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = set(info.get("legal_moves") or list(range(N_MOVES)))

        if self._model is not None:
            try:
                action, _ = self._model.predict(obs, deterministic=True)
                a = int(action)
                return a if a in legal else int(self.rng.choice(sorted(legal)))
            except Exception:
                pass

        # Heuristic blended policy
        cfg = self._effective_cfg()
        bc_logits = self._bc_logits(obs, info)

        # Score each legal action with the blended objective
        scores: dict[int, float] = {}
        for a in legal:
            # RL component: prefer counter-moves or power
            rl_score = 0.5
            if self._last_opp is not None:
                from environments.rps_plus import BEST_COUNTER
                try:
                    best_counter = int(BEST_COUNTER.get(Move(self._last_opp), Move.ROCK))
                except ValueError:
                    best_counter = int(Move.ROCK)
                if a == best_counter:
                    rl_score = 1.0
                elif a == int(Move.POWER) and obs[0] >= 0.4:
                    rl_score = 0.8
                elif a == int(Move.RECHARGE) and obs[0] < 0.2:
                    rl_score = 0.4  # penalise recharging mid-fight unless necessary

            # BC component: log-prob under human-style prior
            bc_score = float(bc_bonus(a, bc_logits))
            # Normalise bc_score to [0,1] range roughly
            bc_score_norm = 1.0 / (1.0 + np.exp(-bc_score))

            scores[a] = cfg.alpha * rl_score + cfg.beta * float(bc_score_norm)

        best = max(scores, key=lambda a: scores[a])
        return best
