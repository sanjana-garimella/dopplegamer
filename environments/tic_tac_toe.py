"""Tic-Tac-Toe environment for interactive dashboard play."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback for minimal runtime
    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def contains(self, x: int) -> bool:
            return isinstance(x, int) and 0 <= x < self.n

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Env:
        metadata: dict[str, list[str]] = {}

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = type("_Spaces", (), {"Discrete": _Discrete, "Box": _Box})()

WIN_PATTERNS = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


@dataclass
class RoundResult:
    turn: int
    agent_move: int
    opponent_move: int
    outcome: int


@dataclass
class EnvState:
    turn: int = 0
    agent_score: int = 0
    opponent_score: int = 0
    history: list[RoundResult] = field(default_factory=list)


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_moves: int = 9, seed: int | None = None) -> None:
        super().__init__()
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.state = EnvState()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.state = EnvState()
        return self._obs(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        if self.board[action] != 0:
            raise ValueError("Move must be placed on an empty square.")

        self.board[action] = 1
        self.state.turn += 1
        if self._check_winner(self.board, 1):
            self.state.agent_score = 1
            outcome = 1
            terminated = True
        else:
            opponent_action = self._choose_opponent_move()
            if opponent_action is not None:
                self.board[opponent_action] = -1
                self.state.turn += 1
            if self._check_winner(self.board, -1):
                self.state.opponent_score = 1
                outcome = -1
                terminated = True
            elif self.state.turn >= self.max_moves or not self._has_empty():
                outcome = 0
                terminated = True
            else:
                outcome = 0
                terminated = False

        self.state.history.append(
            RoundResult(
                turn=self.state.turn,
                agent_move=action,
                opponent_move=-1 if self.state.agent_score == 1 else (opponent_action if opponent_action is not None else -1),
                outcome=outcome,
            )
        )

        truncated = self.state.turn >= self.max_moves
        return self._obs(), float(outcome), terminated, truncated, self._info()

    def render(self):
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(symbols[x] for x in self.board[i * 3:(i + 1) * 3]) for i in range(3)]
        return "\n".join(rows)

    def legal_moves(self) -> list[int]:
        return [i for i, x in enumerate(self.board) if x == 0]

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def _choose_opponent_move(self) -> int | None:
        if hasattr(self, "_opponent_policy"):
            return self._opponent_policy()
        empties = self.legal_moves()
        if not empties:
            return None
        return int(self._rng.choice(empties))

    def _has_empty(self) -> bool:
        return any(x == 0 for x in self.board)

    def _check_winner(self, board: np.ndarray, symbol: int) -> bool:
        return any(all(board[i] == symbol for i in pattern) for pattern in WIN_PATTERNS)

    def _obs(self) -> np.ndarray:
        return self.board.astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "legal_moves": self.legal_moves(),
            "n_legal": len(self.legal_moves()),
        }
