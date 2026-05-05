import pytest

from environments.chess_env import ChessEnv
from environments.checkers import CheckersEnv
from environments.gomoku import GomokuEnv
from environments.nim import NimEnv
from environments.othello import OthelloEnv
from environments.rps_plus import Move, RPSPlusEnv, resolve


def test_resolve_recharge_loses_to_attack():
    assert resolve(Move.RECHARGE, Move.ROCK) == -1


def test_env_step_runs():
    env = RPSPlusEnv(max_turns=3, seed=0)
    obs, info = env.reset(seed=0)
    assert len(obs) == 3
    action = info["legal_moves"][0]
    obs2, reward, term, trunc, info2 = env.step(action)
    assert len(obs2) == 3
    assert isinstance(reward, float)
    assert term is False
    assert trunc is False
    assert "legal_moves" in info2


def test_chess_rejects_illegal_action_index():
    env = ChessEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(999)


def test_othello_rejects_illegal_move():
    env = OthelloEnv(seed=0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(0)


def test_checkers_player_with_no_legal_moves_loses():
    env = CheckersEnv(seed=0)
    env.reset(seed=0)
    env.board[:] = 0
    env.board[0] = 1
    env.board[9] = -1
    env.board[18] = -1
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert truncated is False
    assert reward == -1.0


def test_gomoku_five_in_a_row_wins_before_opponent_reply():
    env = GomokuEnv(seed=0)
    env.reset(seed=0)
    env.board[0, 0:4] = 1

    _, reward, terminated, truncated, _ = env.step(4)

    assert terminated is True
    assert truncated is False
    assert reward == 1.0
    assert env.state.agent_score == 1


def test_nim_taking_final_stone_wins():
    env = NimEnv(seed=0)
    env.reset(seed=0)
    env.piles[:] = [0, 0, 1]

    _, reward, terminated, truncated, _ = env.step(6)

    assert terminated is True
    assert truncated is False
    assert reward == 1.0
    assert env.state.agent_score == 1
