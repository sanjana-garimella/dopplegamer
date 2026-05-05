"""Small-board Gomoku environment with the shared arena API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from environments.tic_tac_toe import EnvState, RoundResult


BOARD_SIZE = 9
WIN_LENGTH = 5


class GomokuEnv:
    """Five-in-a-row on a 9x9 board.

    Agent is 1 and moves first. Opponent is -1. Each env.step resolves the
    agent move and one opponent reply when the game is still active.
    """

    name = "gomoku"

    def __init__(self, max_moves: int | None = None, seed: int | None = None, board_size: int = BOARD_SIZE, win_length: int = WIN_LENGTH) -> None:
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.max_moves = max_moves if max_moves is not None else self.board_size * self.board_size
        self._rng = np.random.default_rng(seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.state = EnvState()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.state = EnvState()
        return self._obs(), self._info()

    def load_challenge(self, board: list[list[int]] | np.ndarray, turn: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
        arr = np.array(board, dtype=np.int8)
        if arr.shape != (self.board_size, self.board_size):
            raise ValueError("Challenge board shape does not match Gomoku board size")
        self.board = arr
        self.state = EnvState(turn=int(turn))
        return self._obs(), self._info()

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            raise ValueError(f"Illegal Gomoku move: {action}")

        self._place(action, 1)
        self.state.turn += 1
        if self._has_five(1):
            self.state.agent_score = 1
            self._record(action, -1, 1)
            return self._obs(), 1.0, True, False, self._info()

        if self.state.turn >= self.max_moves or not self.legal_moves():
            self._record(action, -1, 0)
            return self._obs(), 0.0, True, False, self._info()

        opp_action = self._choose_opponent_move()
        if opp_action is not None:
            self._place(opp_action, -1)
            self.state.turn += 1

        if self._has_five(-1):
            self.state.opponent_score = 1
            self._record(action, opp_action, -1)
            return self._obs(), -1.0, True, False, self._info()

        terminated = self.state.turn >= self.max_moves or not self.legal_moves()
        self._record(action, opp_action if opp_action is not None else -1, 0)
        return self._obs(), 0.0, terminated, False, self._info()

    def legal_moves(self) -> list[int]:
        return [idx for idx, value in enumerate(self.board.flatten()) if value == 0]

    def _place(self, action: int, player: int) -> None:
        r, c = divmod(action, self.board_size)
        self.board[r, c] = player

    def _choose_opponent_move(self) -> int | None:
        legal = self.legal_moves()
        if not legal:
            return None
        if hasattr(self, "_opponent_policy"):
            move = self._opponent_policy()
            if move not in legal:
                raise ValueError(f"Illegal opponent Gomoku move: {move}")
            return int(move)
        return int(self._rng.choice(legal))

    def _has_five(self, player: int) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != player:
                    continue
                for dr, dc in directions:
                    end_r = r + (self.win_length - 1) * dr
                    end_c = c + (self.win_length - 1) * dc
                    if not (0 <= end_r < self.board_size and 0 <= end_c < self.board_size):
                        continue
                    if all(self.board[r + i * dr, c + i * dc] == player for i in range(self.win_length)):
                        return True
        return False

    def _record(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self.state.history.append(
            RoundResult(
                turn=self.state.turn,
                agent_move=int(agent_move),
                opponent_move=int(opponent_move),
                outcome=int(outcome),
            )
        )

    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        import math
        n = int(math.isqrt(len(self.board.flatten())))
        rows = []
        for r in range(n):
            rows.append(" ".join(symbols.get(int(self.board[r, c]), "?") for c in range(n)))
        return "\n".join(rows)

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def _obs(self) -> np.ndarray:
        return self.board.flatten().astype(np.float32)

    def _info(self) -> dict[str, Any]:
        legal = self.legal_moves()
        return {"legal_moves": legal, "n_legal": len(legal)}
