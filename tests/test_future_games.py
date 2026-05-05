import numpy as np
import pytest

from environments.future_games import FUTURE_GAME_ENVS


def test_future_game_envs_share_basic_contract():
    assert set(FUTURE_GAME_ENVS) == {
        "candy_crush",
        "2048",
        "wordle",
        "sudoku",
        "pacman",
        "minecraft",
        "among_us",
        "clash_royale",
        "flappy_bird",
        "ludo",
        "uno",
        "scrabble",
        "monopoly",
        "penalty_shootout",
        "cricket_strategy",
    }

    for name, env_cls in FUTURE_GAME_ENVS.items():
        env = env_cls(max_moves=8, seed=7)
        obs, info = env.reset(seed=7)

        assert isinstance(obs, np.ndarray), name
        assert obs.dtype == np.float32, name
        assert "legal_moves" in info, name
        assert info["n_legal"] == len(info["legal_moves"]), name

        for _ in range(8):
            legal = info["legal_moves"]
            if not legal:
                break
            obs, reward, terminated, truncated, info = env.step(int(legal[0]))
            assert isinstance(obs, np.ndarray), name
            assert isinstance(reward, float), name
            assert isinstance(terminated, bool), name
            assert isinstance(truncated, bool), name
            assert "legal_moves" in info, name
            if terminated or truncated:
                break

        assert env.state.history, name


def test_future_game_envs_are_seed_deterministic_on_reset():
    for name, env_cls in FUTURE_GAME_ENVS.items():
        env_a = env_cls(seed=11)
        env_b = env_cls(seed=11)
        obs_a, info_a = env_a.reset(seed=11)
        obs_b, info_b = env_b.reset(seed=11)

        assert np.array_equal(obs_a, obs_b), name
        assert info_a["legal_moves"] == info_b["legal_moves"], name


def test_pacman_progresses_and_stays_within_legal_moves():
    env_cls = FUTURE_GAME_ENVS["pacman"]
    env = env_cls(max_moves=12, seed=3)
    obs, info = env.reset(seed=3)

    assert obs.dtype == np.float32
    assert info["legal_moves"]

    player_start = env.player
    obs, reward, terminated, truncated, info = env.step(info["legal_moves"][0])

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.player != player_start or len(info["legal_moves"]) == 1


def test_wordle_duplicate_letter_feedback_matches_original_style_accounting():
    env_cls = FUTURE_GAME_ENVS["wordle"]
    env = env_cls(seed=0)
    env.reset(seed=0)
    assert env._score_guess("allay", "awake") == [2, 0, 0, 1, 0]


def test_wordle_allows_repeated_guesses():
    env_cls = FUTURE_GAME_ENVS["wordle"]
    env = env_cls(seed=0)
    env.reset(seed=0)
    idx = env.GUESSES.index("adieu")

    env.step(idx)
    env.step(idx)

    assert len(env.guess_history) == 2


def test_2048_rejects_non_effective_move():
    env_cls = FUTURE_GAME_ENVS["2048"]
    env = env_cls(seed=0)
    env.reset(seed=0)
    env.board = np.array(
        [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    with pytest.raises(ValueError):
        env.step(0)


def test_2048_returns_merge_reward():
    env_cls = FUTURE_GAME_ENVS["2048"]
    env = env_cls(seed=0)
    env.reset(seed=0)
    env.board = np.array(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    _, reward, _, _, _ = env.step(3)
    assert reward == 4.0


def test_sudoku_is_standard_9x9_and_rejects_illegal_action():
    env_cls = FUTURE_GAME_ENVS["sudoku"]
    env = env_cls(seed=0)
    obs, info = env.reset(seed=0)

    assert env.board.shape == (9, 9)
    assert obs.shape[0] == 81
    with pytest.raises(ValueError):
        env.step(0)


def test_sudoku_terminates_when_no_legal_moves_remain():
    env_cls = FUTURE_GAME_ENVS["sudoku"]
    env = env_cls(seed=0)
    env.reset(seed=0)

    env.board = np.array(
        [
            [5, 3, 2, 0, 7, 6, 1, 4, 8],
            [6, 7, 0, 1, 9, 5, 3, 2, 0],
            [1, 9, 8, 0, 3, 2, 5, 6, 7],
            [8, 0, 0, 0, 6, 4, 7, 9, 3],
            [4, 2, 9, 8, 5, 3, 0, 0, 1],
            [7, 0, 5, 9, 2, 0, 4, 0, 6],
            [9, 6, 1, 3, 0, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 0, 4, 5, 8, 0, 0, 7, 9],
        ],
        dtype=np.int8,
    )
    env.givens = env.board != 0
    action = 414

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert reward == -1.0
    assert info["legal_moves"] == []


def test_pacman_tunnel_wrap_and_power_mode():
    env_cls = FUTURE_GAME_ENVS["pacman"]
    env = env_cls(seed=0)
    env.reset(seed=0)
    env.player = (3, 0)
    wrapped = env._move_target(env.player, 3)
    assert wrapped == (3, env.grid.shape[1] - 1)

    env.player = (2, 1)
    env.ghost = (1, 1)
    env.power_pellets[1, 1] = True
    env.pellets[1, 1] = False
    _, reward, terminated, _, _ = env.step(0)
    assert terminated is False
    assert reward >= 10.0
    assert env.frightened_turns > 0
