"""Connect Four environment for interactive dashboard play."""

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

ROWS = 6
COLS = 7

WIN_PATTERNS = []
for r in range(ROWS):
    for c in range(COLS - 3):
        WIN_PATTERNS.append([(r, c + i) for i in range(4)])
for c in range(COLS):
    for r in range(ROWS - 3):
        WIN_PATTERNS.append([(r + i, c) for i in range(4)])
for r in range(ROWS - 3):
    for c in range(COLS - 3):
        WIN_PATTERNS.append([(r + i, c + i) for i in range(4)])
for r in range(ROWS - 3):
    for c in range(3, COLS):
        WIN_PATTERNS.append([(r + i, c - i) for i in range(4)])


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


class ConnectFourEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_moves: int = 42, seed: int | None = None) -> None:
        super().__init__()
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.state = EnvState()
        self.action_space = spaces.Discrete(COLS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(ROWS * COLS,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.state = EnvState()
        return self._obs(), self._info()

    def load_challenge(self, board: list[list[int]] | np.ndarray, turn: int = 0):
        self.board = np.array(board, dtype=np.int8).reshape((ROWS, COLS))
        self.state = EnvState(turn=int(turn))
        return self._obs(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        if action not in self.legal_moves():
            raise ValueError("Column is full or invalid.")

        self._drop_piece(action, 1)
        self.state.turn += 1
        if self._check_winner(1):
            self.state.agent_score = 1
            self.state.history.append(RoundResult(self.state.turn, action, -1, 1))
            return self._obs(), 1.0, True, False, self._info()

        opp_action = self._choose_opponent_move()
        if opp_action is not None:
            self._drop_piece(opp_action, -1)
            self.state.turn += 1

        if self._check_winner(-1):
            self.state.opponent_score = 1
            self.state.history.append(RoundResult(self.state.turn, action, opp_action, -1))
            return self._obs(), -1.0, True, False, self._info()

        self.state.history.append(RoundResult(self.state.turn, action, opp_action, 0))
        terminated = self.state.turn >= self.max_moves or not self.legal_moves()
        return self._obs(), 0.0, terminated, False, self._info()

    def render(self):
        rows = []
        for r in range(ROWS - 1, -1, -1):
            rows.append("|" + "|".join(self._symbol(self.board[r, c]) for c in range(COLS)) + "|")
        return "\n".join(rows)

    def legal_moves(self) -> list[int]:
        return [c for c in range(COLS) if self.board[-1, c] == 0]

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def _drop_piece(self, column: int, symbol: int) -> None:
        for row in range(ROWS):
            if self.board[row, column] == 0:
                self.board[row, column] = symbol
                return

    def _choose_opponent_move(self) -> int | None:
        if hasattr(self, "_opponent_policy"):
            return self._opponent_policy()
        legal = self.legal_moves()
        if not legal:
            return None
        return int(self._rng.choice(legal))

    def _check_winner(self, player: int) -> bool:
        for pattern in WIN_PATTERNS:
            if all(self.board[r, c] == player for r, c in pattern):
                return True
        return False

    def _symbol(self, value: int) -> str:
        return "." if value == 0 else "X" if value == 1 else "O"

    def _obs(self) -> np.ndarray:
        return self.board.flatten().astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {"legal_moves": self.legal_moves(), "n_legal": len(self.legal_moves())}
