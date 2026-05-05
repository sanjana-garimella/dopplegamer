from __future__ import annotations

import pytest

from dashboard.pages.live_game import (
    PLAY_MODES,
    _agent_runtime_status,
    _agent_support_note,
    _chess_action_for_target,
    _coerce_action_to_legal,
    _game_rules,
    _opponent_legal_after_pending,
    _supported_agents_for_game,
)
from agents.profile_counter import ProfileCounterAgent
from environments.tic_tac_toe import TicTacToeEnv


def test_game_rules_include_playable_chess_instructions():
    rules = _game_rules("Chess")
    assert any("White" in rule for rule in rules)
    assert any("highlighted destination" in rule for rule in rules)


def test_chess_action_prefers_queen_promotion():
    chess = pytest.importorskip("chess")

    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    selected_square = chess.A7
    target_square = chess.A8

    move_idx = _chess_action_for_target(board, selected_square, target_square)

    legal_moves = list(board.legal_moves)
    assert move_idx is not None
    assert legal_moves[move_idx].uci() == "a7a8q"


def test_non_rps_games_only_offer_stable_agents():
    assert _supported_agents_for_game("Chess") == ["random", "profile_counter"]
    assert _supported_agents_for_game("Gomoku") == ["random", "profile_counter"]
    assert _supported_agents_for_game("Nim") == ["random", "profile_counter"]
    assert _supported_agents_for_game("War") == ["random", "profile_counter"]


def test_illegal_agent_action_falls_back_to_first_legal_move():
    assert _coerce_action_to_legal(99, [4, 7, 9]) == 4
    assert _coerce_action_to_legal("7", [4, 7, 9]) == 7


def test_agent_support_note_mentions_stable_bot_set():
    note = _agent_support_note("Chess", "random")
    assert "stable bot set" in note


def test_runtime_status_marks_random_as_rule_based():
    label, detail = _agent_runtime_status("random", object())
    assert label == "Rule-based baseline"
    assert "No training checkpoint" in detail


def test_runtime_status_marks_profile_counter_as_history_trained():
    agent = ProfileCounterAgent(player_id=None)
    label, detail = _agent_runtime_status("profile_counter", agent)
    assert label == "History-trained"
    assert "local gameplay database" in detail


def test_live_game_exposes_four_match_modes():
    assert list(PLAY_MODES) == [
        "Play a Bot",
        "Try Another Bot",
        "Two Players",
        "Watch Bots",
    ]


def test_human_two_legal_moves_exclude_pending_tic_tac_toe_square():
    env = TicTacToeEnv(seed=0)
    env.reset(seed=0)

    legal = _opponent_legal_after_pending(env, "Tic-Tac-Toe", 4)

    assert 4 not in legal
    assert len(legal) == 8
