"""Checkers environment with Gymnasium-compatible API.

Action space  : Discrete(4096) — index into (from_sq * 64 + to_sq)
Observation   : float32 array of shape (64,)
Reward        : +1.0 win, -1.0 loss, 0.0 draw / ongoing
"""

from __future__ import annotations

from typing import Any
import numpy as np

from environments.tic_tac_toe import EnvState, RoundResult


class CheckersEnv:
    """Checkers Environment."""
    
    name = "checkers"
    
    def __init__(self, max_moves: int = 200, seed: int | None = None, forced_jump: bool = True) -> None:
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self.board = np.zeros(64, dtype=np.int8)
        self.state = EnvState()
        self.forced_jump = forced_jump
        self.must_jump_piece = None # If a piece is mid-jump, it must continue jumping
        
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = np.zeros(64, dtype=np.int8)
        self.must_jump_piece = None
        
        # Agent = 1, Opponent = -1
        # Kings = 2, -2
        # Board is 8x8. Black squares only.
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r * 8 + c] = 1 # Agent starts at top (rows 0-2) moving down
                    
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self.board[r * 8 + c] = -1 # Opponent starts at bottom (rows 5-7) moving up
                    
        self.state = EnvState()
        return self._obs(), self._info(1)

    def load_challenge(self, board: list[int] | np.ndarray, turn: int = 0, must_jump_piece: int | None = None) -> tuple[np.ndarray, dict]:
        self.board = np.array(board, dtype=np.int8).reshape(64)
        self.must_jump_piece = must_jump_piece
        self.state = EnvState(turn=int(turn))
        return self._obs(), self._info(1)

    def _get_moves(self, board: np.ndarray, player: int, specific_piece: int | None = None) -> dict[int, tuple[int, int]]:
        # Returns dict: action_id -> (from_sq, to_sq, jump_sq)
        moves = {}
        jumps = {}
        
        direction = 1 if player == 1 else -1
        
        for i in range(64):
            if board[i] == 0:
                continue
            p = board[i]
            if p * player <= 0:
                continue # Not player's piece
                
            if specific_piece is not None and i != specific_piece:
                continue
                
            is_king = abs(p) == 2
            r, c = i // 8, i % 8
            
            # Normal moves & jumps
            dirs = []
            if is_king:
                dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            else:
                dirs = [(direction, -1), (direction, 1)]
                
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    idx = nr * 8 + nc
                    if board[idx] == 0:
                        # Normal move
                        action = i * 64 + idx
                        moves[action] = (i, idx, -1)
                    elif board[idx] * player < 0:
                        # Potential jump
                        jr, jc = nr + dr, nc + dc
                        if 0 <= jr < 8 and 0 <= jc < 8:
                            j_idx = jr * 8 + jc
                            if board[j_idx] == 0:
                                action = i * 64 + j_idx
                                jumps[action] = (i, j_idx, idx)
                                
        # Forced jumps
        if jumps and self.forced_jump:
            return jumps
        if specific_piece is not None:
            return {} # Mid-jump but no more jumps available
        return {**moves, **jumps}

    def legal_moves(self) -> list[int]:
        moves = self._get_moves(self.board, 1, self.must_jump_piece)
        return list(moves.keys())

    def _apply_move(self, board: np.ndarray, action: int, moves_dict: dict, player: int) -> bool:
        # Returns True if the turn continues (multiple jump)
        from_sq, to_sq, jump_sq = moves_dict[action]
        piece = board[from_sq]
        board[from_sq] = 0
        board[to_sq] = piece
        
        if jump_sq != -1:
            board[jump_sq] = 0 # Capture
            
        # King promotion
        r = to_sq // 8
        promoted = False
        if player == 1 and r == 7 and piece == 1:
            board[to_sq] = 2
            promoted = True
        elif player == -1 and r == 0 and piece == -1:
            board[to_sq] = -2
            promoted = True
            
        if jump_sq != -1 and not promoted:
            # Check for multi-jumps
            next_moves = self._get_moves(board, player, specific_piece=to_sq)
            if next_moves:
                return True # Turn continues
        return False

    def step(self, action: int):
        moves_dict = self._get_moves(self.board, 1, self.must_jump_piece)
        legal = list(moves_dict.keys())
        
        agent_move_recorded = action
        if legal:
            if action not in legal:
                raise ValueError(f"Illegal checkers move: {action}")
            continues = self._apply_move(self.board, action, moves_dict, 1)
            agent_move_recorded = action
            if continues:
                self.must_jump_piece = moves_dict[action][1]
                return self._obs(), 0.0, False, False, self._info(1)
            else:
                self.must_jump_piece = None
        else:
            self.state.opponent_score = 1
            self._record_history(-1, -1, -1)
            return self._obs(), -1.0, True, False, self._info(1)

        self.state.turn += 1

        outcome = self._terminal_outcome(next_player=-1)
        if outcome is not None:
            self._record_history(agent_move_recorded, -1, outcome)
            return self._obs(), float(outcome), True, False, self._info(1)

        # Opponent Move Loop
        opp_continues = True
        opp_must_jump = None
        opponent_action = None
        
        while opp_continues:
            opp_moves_dict = self._get_moves(self.board, -1, opp_must_jump)
            opp_legal = list(opp_moves_dict.keys())
            
            if not opp_legal:
                break
                
            opponent_action = self._choose_opponent_move(opp_legal)
            if opponent_action is None:
                break
                
            opp_continues = self._apply_move(self.board, opponent_action, opp_moves_dict, -1)
            if opp_continues:
                opp_must_jump = opp_moves_dict[opponent_action][1]
            else:
                opp_must_jump = None
                self.state.turn += 1
                
        post_outcome = self._terminal_outcome(next_player=1)
        terminated = post_outcome is not None
        truncated = self.state.turn >= self.max_moves
        outcome = post_outcome if terminated else (self._get_outcome() if truncated else 0)
        
        self._record_history(agent_move_recorded, opponent_action if opponent_action is not None else -1, outcome)
        return self._obs(), float(outcome), terminated, truncated, self._info(1)

    def _choose_opponent_move(self, opp_legal: list[int]) -> int | None:
        if not opp_legal:
            return None
        if hasattr(self, "_opponent_policy"):
            move = self._opponent_policy()
            if move not in opp_legal:
                raise ValueError(f"Illegal opponent checkers move: {move}")
            return move
        return int(self._rng.choice(opp_legal))

    def _check_terminal(self) -> bool:
        return self._terminal_outcome(next_player=1) is not None

    def _terminal_outcome(self, next_player: int) -> int | None:
        agent_pieces = np.sum(self.board > 0)
        opp_pieces = np.sum(self.board < 0)
        if agent_pieces == 0 or opp_pieces == 0:
            return self._get_outcome()

        moves = self._get_moves(self.board, next_player, self.must_jump_piece if next_player == 1 else None)
        if not moves:
            if next_player == 1:
                self.state.opponent_score = 1
                return -1
            self.state.agent_score = 1
            return 1
        return None

    def _get_outcome(self) -> int:
        agent_pieces = np.sum(self.board > 0)
        opp_pieces = np.sum(self.board < 0)
        if agent_pieces > opp_pieces:
            self.state.agent_score = 1
            return 1
        elif opp_pieces > agent_pieces:
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

    def _obs(self) -> np.ndarray:
        return self.board.astype(np.float32)

    def render(self) -> str:
        symbols = {0: ".", 1: "B", -1: "W", 2: "K", -2: "k"}
        rows = ["  " + " ".join(str(c) for c in range(8))]
        for r in range(8):
            rows.append(f"{r} " + " ".join(symbols.get(int(self.board[r * 8 + c]), "?") for c in range(8)))
        return "\n".join(rows)

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def _info(self, player: int) -> dict[str, Any]:
        legal = list(self._get_moves(self.board, player, self.must_jump_piece).keys())
        return {
            "legal_moves": legal,
            "n_legal": len(legal),
        }
