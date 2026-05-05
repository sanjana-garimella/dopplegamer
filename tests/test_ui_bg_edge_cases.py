"""Edge case tests for Dashboard UI logic and Background Data Tasks."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Mock dependencies
from collections import namedtuple

# UI Logic
from dashboard.pages.player_profile import _get_game_winner, _format_chess_move
from agents.rl.rewards import bc_bonus

# Background Tasks
from impostor.player_profiles import PlayerProfileManager
from data.schemas import init_db
from environments.utils import history_to_records


class MockState:
    def __init__(self, agent_score, opponent_score):
        self.agent_score = agent_score
        self.opponent_score = opponent_score


class MockEnv:
    def __init__(self, agent_score, opponent_score):
        self.state = MockState(agent_score, opponent_score)


class MockChessBoard:
    def __init__(self, is_checkmate=False, is_stalemate=False, turn=True):
        self._is_checkmate = is_checkmate
        self._is_stalemate = is_stalemate
        self.turn = turn  # True for White, False for Black

    def is_checkmate(self):
        return self._is_checkmate

    def is_stalemate(self):
        return self._is_stalemate


class MockChessEnv:
    def __init__(self, board):
        self._board = board


# ═══════════════════════════════════════════════════════════════════════════════
# UI EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestUILogicEdgeCases:
    
    def test_get_game_winner_rps_tie(self):
        """Tie state in RPS+ should return 'tie'."""
        env = MockEnv(5, 5)
        assert _get_game_winner(env, "RPS+") == "tie"

    def test_get_game_winner_ttt_agent_win(self):
        """Agent win in Tic-Tac-Toe should return 'player'."""
        # Note: UI logic assigns 'player' when agent_score > opponent_score
        env = MockEnv(10, -10)
        assert _get_game_winner(env, "Tic-Tac-Toe") == "player"

    def test_get_game_winner_chess_checkmate(self):
        """Chess checkmate edge cases."""
        # AI is black (False). If white just moved and caused checkmate (turn is now False) -> player wins
        # Wait, the UI logic: "ai" if env._board.turn else "player". 
        # If it's black's turn (False) and checkmate, white (player) won.
        board = MockChessBoard(is_checkmate=True, turn=False)
        env = MockChessEnv(board)
        assert _get_game_winner(env, "Chess") == "player"
        
        # If it's white's turn (True) and checkmate, black (ai) won.
        board2 = MockChessBoard(is_checkmate=True, turn=True)
        env2 = MockChessEnv(board2)
        assert _get_game_winner(env2, "Chess") == "ai"

    def test_get_game_winner_chess_stalemate(self):
        board = MockChessBoard(is_checkmate=False, is_stalemate=True)
        env = MockChessEnv(board)
        assert _get_game_winner(env, "Chess") == "tie"

    def test_get_game_winner_unknown_game(self):
        """Unknown game type should default to 'tie' gracefully."""
        env = MockEnv(1, 0)
        assert _get_game_winner(env, "Unknown") == "tie"

    def test_bc_bonus_out_of_bounds(self):
        """bc_bonus should return 0.0 safely when action index exceeds model knowledge."""
        logits = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) # Length 6
        
        # Valid actions
        assert bc_bonus(0, logits) != 0.0
        assert bc_bonus(5, logits) != 0.0
        
        # Out of bounds actions (e.g. from Tic-Tac-Toe)
        assert bc_bonus(6, logits) == 0.0
        assert bc_bonus(8, logits) == 0.0
        assert bc_bonus(-1, logits) == 0.0

    def test_format_chess_move_fallback(self):
        """If chess package fails or invalid move, should fallback to uci string."""
        class MockMove:
            def uci(self): return "e2e4"
            
        class MockBoard:
            def san(self, move):
                raise ValueError("Invalid move for SAN")
                
        # Should catch error and return uci()
        assert _format_chess_move(MockMove(), MockBoard()) == "e2e4"


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND TASK EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackgroundTasksEdgeCases:
    
    def _tmp_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        return f.name
        
    def test_history_to_records_empty(self):
        """Converting empty history should yield empty list safely."""
        assert history_to_records([], "game_123") == []
        
    def test_update_signature_empty_lists(self):
        """Updating signature with no gameplay data shouldn't crash."""
        db = self._tmp_db()
        try:
            init_db(db)
            mgr = PlayerProfileManager(db)
            p = mgr.create("Test Player")
            
            # Pass empty lists
            mgr.update_signature(p.player_id, [], [], [])
            
            # Signature should still be retrieved as mostly zeros
            profile = mgr.get(p.player_id)
            sig = profile.behavioral_signature
            assert np.all(np.array(sig.move_distribution) == 0.0)
            assert sig.energy_aggression == 0.0
        finally:
            os.unlink(db)

    def test_update_signature_mismatched_lengths(self):
        """If logging logic gets corrupted and returns mismatched lists, should it crash?"""
        db = self._tmp_db()
        try:
            init_db(db)
            mgr = PlayerProfileManager(db)
            p = mgr.create("Corrupted Player")
            
            moves = [[0, 1, 2]]
            outcomes = [[1, -1]] # Shorter
            opps = [[0, 1, 2]]
            
            # The metrics module (action_distribution) might slice or throw.
            # We ensure it doesn't hard-crash the DB or leaves it in a bad state.
            try:
                mgr.update_signature(p.player_id, moves, outcomes, opps)
            except Exception:
                pass # Expected to potentially throw due to index errors in metrics, but DB should be fine
            profile = mgr.get(p.player_id)
            sig = profile.behavioral_signature
            assert sig is not None
        finally:
            os.unlink(db)

    def test_update_signature_huge_data(self):
        """Updating signature with a massive number of rounds."""
        db = self._tmp_db()
        try:
            init_db(db)
            mgr = PlayerProfileManager(db)
            p = mgr.create("Big Data Player")
            
            moves = [list(np.random.randint(0, 6, size=5000))]
            outcomes = [list(np.random.choice([-1, 0, 1], size=5000))]
            opps = [list(np.random.randint(0, 6, size=5000))]
            
            # Should compute without timing out or overflowing
            mgr.update_signature(p.player_id, moves, outcomes, opps)
            profile = mgr.get(p.player_id)
            sig = profile.behavioral_signature
            assert profile.total_rounds == 5000
            # Distribution should be approximately uniform
            assert np.allclose(sig.move_distribution, 1/6, atol=0.05)
        finally:
            os.unlink(db)

    def test_update_signature_accumulates_batches(self):
        """Profile counters should reflect all batches, not just the latest game."""
        db = self._tmp_db()
        try:
            init_db(db)
            mgr = PlayerProfileManager(db)
            p = mgr.create("Accumulating Player")

            mgr.update_signature(p.player_id, [[0, 0, 0]], [[1, 1, 1]], [[1, 1, 1]])
            mgr.update_signature(p.player_id, [[1]], [[-1]], [[0]])

            profile = mgr.get(p.player_id)
            sig = profile.behavioral_signature
            assert profile.games_played == 2
            assert profile.total_rounds == 4
            assert profile.win_rate == 0.75
            assert np.allclose(sig.move_distribution, [0.75, 0.25, 0.0, 0.0, 0.0, 0.0])
        finally:
            os.unlink(db)
