"""Nim environment with shared arena API."""

from __future__ import annotations

from typing import Any

import numpy as np

from environments.tic_tac_toe import EnvState, RoundResult


MAX_REMOVE = 3


class NimEnv:
    """Three-pile normal-play Nim.

    Actions are encoded as ``pile_index * MAX_REMOVE + (remove_count - 1)``.
    Taking the final object wins immediately.
    """

    name = "nim"

    def __init__(self, max_moves: int = 30, seed: int | None = None, piles: list[int] | tuple[int, ...] | None = None) -> None:
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self.initial_piles = np.array(piles if piles is not None else [3, 4, 5], dtype=np.int8)
        self.piles = self.initial_piles.copy()
        self.state = EnvState()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.piles = self.initial_piles.copy()
        self.state = EnvState()
        return self._obs(), self._info()

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            raise ValueError(f"Illegal Nim move: {action}")

        self._apply(action)
        self.state.turn += 1
        if self._empty():
            self.state.agent_score = 1
            self._record(action, -1, 1)
            return self._obs(), 1.0, True, False, self._info()

        opp_action = self._choose_opponent_move()
        if opp_action is not None:
            self._apply(opp_action)
            self.state.turn += 1

        if self._empty():
            self.state.opponent_score = 1
            self._record(action, opp_action, -1)
            return self._obs(), -1.0, True, False, self._info()

        truncated = self.state.turn >= self.max_moves
        outcome = self._outcome() if truncated else 0
        self._record(action, opp_action if opp_action is not None else -1, outcome)
        return self._obs(), float(outcome), False, truncated, self._info()

    def legal_moves(self) -> list[int]:
        legal: list[int] = []
        for pile_idx, count in enumerate(self.piles):
            for remove in range(1, min(int(count), MAX_REMOVE) + 1):
                legal.append(pile_idx * MAX_REMOVE + (remove - 1))
        return legal

    def _apply(self, action: int) -> None:
        pile_idx, remove_idx = divmod(int(action), MAX_REMOVE)
        self.piles[pile_idx] -= remove_idx + 1

    def _choose_opponent_move(self) -> int | None:
        legal = self.legal_moves()
        if not legal:
            return None
        if hasattr(self, "_opponent_policy"):
            move = self._opponent_policy()
            if move not in legal:
                raise ValueError(f"Illegal opponent Nim move: {move}")
            return int(move)
        return int(self._rng.choice(legal))

    def _empty(self) -> bool:
        return bool(np.all(self.piles == 0))

    def _outcome(self) -> int:
        if self.state.agent_score > self.state.opponent_score:
            return 1
        if self.state.opponent_score > self.state.agent_score:
            return -1
        return 0

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
        return f"Nim: piles={list(self.piles)} score={self.state.agent_score}:{self.state.opponent_score}"

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def _obs(self) -> np.ndarray:
        return (self.piles.astype(np.float32) / 5.0)

    def _info(self) -> dict[str, Any]:
        legal = self.legal_moves()
        return {"legal_moves": legal, "n_legal": len(legal)}
