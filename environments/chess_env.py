"""Chess environment with Gymnasium-compatible API.

Wraps python-chess with the same reset()/step()/render() contract used by
RPSPlusEnv, so all AgentBench agents can be evaluated on Chess too.

python-chess is an optional dependency — import is lazy so this module
is always safe to import.

Action space  : Discrete(len(legal_moves)) — index into board.legal_moves list
Observation   : float32 array of shape (64,) — piece value per square
Reward        : +1.0 win, -1.0 loss, 0.0 draw / ongoing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Piece-type → normalised value (King excluded from material count)
_PIECE_VALUES = {1: 0.1, 2: 0.3, 3: 0.3, 4: 0.5, 5: 0.9, 6: 0.0}


@dataclass
class ChessState:
    turn: int = 0
    agent_score: int = 0
    opponent_score: int = 0
    result: str = "*"
    done: bool = False


class ChessEnv:
    """Minimal Chess environment backed by python-chess.

    The opponent plays a random legal move each turn.
    """

    name = "chess"

    def __init__(self, max_moves: int = 200, seed: int | None = None) -> None:
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self._board = None
        self.state = ChessState()

    # ---------------------------------------------------------------- Gymnasium API

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        try:
            import chess
        except ImportError as exc:
            raise ImportError(
                "python-chess is required for ChessEnv. Install with: pip install chess"
            ) from exc

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._board = chess.Board()
        self.state = ChessState()
        return self._encode(), self._info()

    def load_challenge(self, fen: str) -> tuple[np.ndarray, dict]:
        try:
            import chess
        except ImportError as exc:
            raise ImportError(
                "python-chess is required for ChessEnv. Install with: pip install chess"
            ) from exc
        self._board = chess.Board(fen)
        self.state = ChessState(turn=self._board.fullmove_number * 2 - (0 if self._board.turn else 1))
        return self._encode(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        legal = list(self._board.legal_moves)
        if not legal:
            return self._encode(), 0.0, True, False, self._info()

        # Agent move
        if not isinstance(action, int) or not 0 <= action < len(legal):
            raise ValueError(f"Illegal chess action index: {action}")
        self._board.push(legal[action])
        self.state.turn += 1

        reward, terminated = self._check_terminal()
        if terminated:
            return self._encode(), reward, True, False, self._info()

        # Opponent: random legal reply or agent policy
        opp_legal = list(self._board.legal_moves)
        if opp_legal:
            if hasattr(self, "_opponent_policy"):
                opp_move_idx = self._opponent_policy()
                if not isinstance(opp_move_idx, int) or not 0 <= opp_move_idx < len(opp_legal):
                    raise ValueError(f"Illegal opponent chess action index: {opp_move_idx}")
                opp_move = opp_legal[opp_move_idx]
            else:
                opp_move = opp_legal[int(self._rng.integers(len(opp_legal)))]
            self._board.push(opp_move)
            self.state.turn += 1

        reward, terminated = self._check_terminal()
        truncated = self.state.turn >= self.max_moves
        return self._encode(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        if self._board is None:
            return "(not initialized)"
        return (
            f"turn={self.state.turn} result={self.state.result}\n"
            + str(self._board)
        )

    def legal_moves(self) -> list[int]:
        if self._board is None:
            return []
        return list(range(len(list(self._board.legal_moves))))

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    # ---------------------------------------------------------------- helpers

    def _encode(self) -> np.ndarray:
        arr = np.zeros(64, dtype=np.float32)
        if self._board is None:
            return arr
        for sq in range(64):
            piece = self._board.piece_at(sq)
            if piece is not None:
                val = _PIECE_VALUES.get(piece.piece_type, 0.0)
                arr[sq] = val if piece.color else -val
        return arr

    def _info(self) -> dict:
        if self._board is None:
            return {"legal_moves": [], "n_legal": 0, "result": "*"}
        legal = list(self._board.legal_moves)
        legal_indices = list(range(len(legal)))
        return {
            "legal_moves": legal_indices,
            "n_legal": len(legal_indices),
            "result": self.state.result,
            "legal_uci": [move.uci() for move in legal],
        }

    def _check_terminal(self) -> tuple[float, bool]:
        if not self._board.is_game_over():
            return 0.0, False
        result = self._board.result()
        self.state.result = result
        self.state.done = True
        if result == "1-0":
            self.state.agent_score = 1
            return 1.0, True
        if result == "0-1":
            self.state.opponent_score = 1
            return -1.0, True
        return 0.0, True  # draw
