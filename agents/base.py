"""Common Agent interface so the evaluator can treat every architecture the same."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from environments.rps_plus import BEST_COUNTER, Move


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        """Choose an action given observation and info.

        Args:
            obs: Observation array (game-specific).
            info: Additional info dict, must contain 'legal_moves' key.

        Returns:
            Action index.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Basic validation
        if not isinstance(obs, np.ndarray):
            raise ValueError(f"obs must be numpy array, got {type(obs)}")
        if not isinstance(info, dict):
            raise ValueError(f"info must be dict, got {type(info)}")
        if 'legal_moves' not in info:
            raise ValueError("info must contain 'legal_moves'")
        legal = info['legal_moves']
        if not isinstance(legal, (list, set)) or not all(isinstance(m, int) for m in legal):
            raise ValueError("legal_moves must be list/set of ints")
        if not legal:
            raise ValueError("legal_moves cannot be empty")

    def reset(self, seed: int | None = None) -> None:
        return None


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        from environments.rps_plus import N_MOVES
        self.N_MOVES = N_MOVES
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        super().act(obs, info)  # Validation
        legal = info.get("legal_moves") or list(range(self.N_MOVES))
        return int(self.rng.choice(legal))


class HeuristicAgent(Agent):
    """Counter-frequency baseline: pick the move that beats the opponent's most-used base move,
    falling back to RECHARGE when energy is low."""

    name = "heuristic"

    # Canonical best-response table (POWER ties POWER; RECHARGE never wins).
    COUNTER = BEST_COUNTER

    def __init__(self, seed: int | None = None) -> None:
        from environments.rps_plus import N_MOVES
        self.N_MOVES = N_MOVES
        self.rng = np.random.default_rng(seed)
        self._opponent_history: list[int] = []

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._opponent_history.clear()

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        super().act(obs, info)  # Validation
        legal = set(info.get("legal_moves") or list(range(self.N_MOVES)))
        try:
            from environments.rps_plus import Move
            agent_energy = int(round(obs[0] * 5))
            if agent_energy <= 1 and int(Move.RECHARGE) in legal:
                return int(Move.RECHARGE)
            if not self._opponent_history:
                return int(self.rng.choice(list(legal)))
            counts = np.bincount(self._opponent_history, minlength=self.N_MOVES)
            favorite = Move(int(counts.argmax()))
            counter = self.COUNTER[favorite]
            if int(counter) in legal:
                return int(counter)
        except (ValueError, KeyError, ImportError, IndexError):
            pass
            
        if not self._opponent_history:
            return int(self.rng.choice(list(legal)))
        return int(self.rng.choice(list(legal)))


    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self._opponent_history.append(int(opponent_move))


class OptimalAgent(Agent):
    """Lag-1 best-response agent: counters the opponent's previous move.

    Not a game-theoretic optimum; registered as both `optimal` and `lag1_counter`.
    """

    name = "lag1_counter"

    COUNTER = BEST_COUNTER

    def __init__(self, seed: int | None = None) -> None:
        from environments.rps_plus import N_MOVES
        self.N_MOVES = N_MOVES
        self.rng = np.random.default_rng(seed)
        self._last_opp: int | None = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._last_opp = None

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        super().act(obs, info)
        legal = set(info.get("legal_moves") or [])
        if not legal: # Fallback for safety
            return 0
            
        try:
            from environments.rps_plus import Move
            # Only try RPS logic if energy/moves align with RPS+ expectations
            if len(obs) > 0 and self._last_opp is not None:
                agent_energy = int(round(obs[0] * 5))
                counter = int(self.COUNTER[Move(self._last_opp)])
                if counter in legal:
                    return counter
                    
                # Default to POWER if energy allows
                if agent_energy >= 2 and int(Move.POWER) in legal:
                    return int(Move.POWER)
        except (ValueError, KeyError, ImportError, IndexError):
            pass
            
        return int(self.rng.choice(list(legal)))

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self._last_opp = int(opponent_move)


AGENT_REGISTRY: dict[str, type[Agent]] = {}


def register_agent(name: str, cls: type[Agent]) -> None:
    """Add an agent class to the global registry."""
    AGENT_REGISTRY[name] = cls


# Register core agents
register_agent("random", RandomAgent)
register_agent("heuristic", HeuristicAgent)
register_agent("lag1_counter", OptimalAgent)
register_agent("optimal", OptimalAgent)  # alias
