"""Othello (Reversi) environment with Gymnasium-compatible API.

Action space  : Discrete(64) — index into flat 8x8 board
Observation   : float32 array of shape (64,)
Reward        : +1.0 win, -1.0 loss, 0.0 draw / ongoing
"""

from __future__ import annotations

from typing import Any
import numpy as np

from environments.tic_tac_toe import EnvState, RoundResult


class OthelloEnv:
    """Othello (Reversi) Environment."""
    
    name = "othello"
    
    def __init__(self, max_moves: int = 120, seed: int | None = None) -> None:
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.state = EnvState()
        
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = np.zeros((8, 8), dtype=np.int8)
        
        # Standard starting position (Agent = 1 = Black, Opponent = -1 = White)
        self.board[3, 3] = -1
        self.board[4, 4] = -1
        self.board[3, 4] = 1
        self.board[4, 3] = 1
        
        self.state = EnvState()
        return self._obs(), self._info(1)

    def _get_flips(self, board: np.ndarray, r: int, c: int, player: int) -> list[tuple[int, int]]:
        if board[r, c] != 0:
            return []
            
        flips = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            current_flips = []
            
            while 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == -player:
                current_flips.append((nr, nc))
                nr += dr
                nc += dc
                
            if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == player:
                flips.extend(current_flips)
                
        return flips

    def _legal_moves_for_player(self, board: np.ndarray, player: int) -> list[int]:
        legal = []
        for r in range(8):
            for c in range(8):
                if board[r, c] == 0:
                    flips = self._get_flips(board, r, c, player)
                    if flips:
                        legal.append(r * 8 + c)
        return legal

    def legal_moves(self) -> list[int]:
        return self._legal_moves_for_player(self.board, 1)

    def _apply_move(self, board: np.ndarray, action: int, player: int) -> None:
        r, c = action // 8, action % 8
        flips = self._get_flips(board, r, c, player)
        board[r, c] = player
        for fr, fc in flips:
            board[fr, fc] = player

    def step(self, action: int):
        legal = self.legal_moves()
        
        # If Agent has no legal moves, they are forced to pass (action is ignored)
        # However, if they have legal moves, the action must be valid.
        agent_move_recorded = action
        if legal:
            if action not in legal:
                raise ValueError(f"Illegal Othello move: {action}")
            self._apply_move(self.board, action, 1)
            agent_move_recorded = action
        else:
            agent_move_recorded = -1 # Pass
            
        self.state.turn += 1
        
        # Check if game over before opponent moves
        agent_legal_next = self._legal_moves_for_player(self.board, 1)
        opp_legal = self._legal_moves_for_player(self.board, -1)
        
        if not agent_legal_next and not opp_legal:
            outcome = self._get_outcome()
            self._record_history(agent_move_recorded, -1, outcome)
            return self._obs(), float(outcome), True, False, self._info(1)

        # Opponent Move
        opponent_action = self._choose_opponent_move(opp_legal)
        if opponent_action is not None:
            self._apply_move(self.board, opponent_action, -1)
            self.state.turn += 1
            
        # Check game over after opponent moves
        agent_legal_next = self._legal_moves_for_player(self.board, 1)
        opp_legal = self._legal_moves_for_player(self.board, -1)
        
        terminated = False
        outcome = 0
        if not agent_legal_next and not opp_legal:
            terminated = True
            outcome = self._get_outcome()
            
        truncated = self.state.turn >= self.max_moves
        if truncated and not terminated:
            outcome = self._get_outcome()
            
        self._record_history(agent_move_recorded, opponent_action if opponent_action is not None else -1, outcome)
        return self._obs(), float(outcome), terminated, truncated, self._info(1)

    def _choose_opponent_move(self, opp_legal: list[int]) -> int | None:
        if not opp_legal:
            return None
        if hasattr(self, "_opponent_policy"):
            move = self._opponent_policy()
            if move not in opp_legal:
                raise ValueError(f"Illegal opponent Othello move: {move}")
            return move
        return int(self._rng.choice(opp_legal))

    def _get_outcome(self) -> int:
        agent_count = np.sum(self.board == 1)
        opp_count = np.sum(self.board == -1)
        if agent_count > opp_count:
            self.state.agent_score = 1
            return 1
        elif opp_count > agent_count:
            self.state.opponent_score = 1
            return -1
        return 0

    def _record_history(self, agent_move: int, opp_move: int, outcome: int) -> None:
        self.state.history.append(
            RoundResult(
                turn=self.state.turn,
                agent_move=agent_move,
                opponent_move=opp_move,
                outcome=outcome,
            )
        )

    def render(self) -> str:
        symbols = {0: ".", 1: "B", -1: "W"}
        rows = ["  " + " ".join(str(c) for c in range(8))]
        for r in range(8):
            rows.append(f"{r} " + " ".join(symbols[int(self.board[r, c])] for c in range(8)))
        return "\n".join(rows)

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def _obs(self) -> np.ndarray:
        return self.board.flatten().astype(np.float32)

    def _info(self, player: int) -> dict[str, Any]:
        legal = self._legal_moves_for_player(self.board, player)
        return {
            "legal_moves": legal,
            "n_legal": len(legal),
        }
