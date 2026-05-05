from __future__ import annotations

import pandas as pd

from data.schemas import connect, init_db
from data.backfill import backfill_game_types, canonical_game_type, infer_game_type
from data.collector import collect
from dashboard.pages.player_profile import _filter_sessions, _load_game_sessions, _summarize_session_groups


def test_init_db_adds_game_type_column(tmp_path):
    db = tmp_path / "games.db"
    init_db(db)

    conn = connect(db)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    finally:
        conn.close()

    assert "game_type" in columns


def test_summarize_session_groups_computes_win_rate():
    sessions = pd.DataFrame(
        [
            {"game_id": "g1", "player_name": "A", "player_id": "a", "bot_name": "random", "game_type": "RPS+", "n_turns": 10, "agent_score": 3, "opponent_score": 1, "result": "Win"},
            {"game_id": "g2", "player_name": "A", "player_id": "a", "bot_name": "random", "game_type": "RPS+", "n_turns": 12, "agent_score": 1, "opponent_score": 2, "result": "Loss"},
            {"game_id": "g3", "player_name": "B", "player_id": "b", "bot_name": "heuristic", "game_type": "Chess", "n_turns": 20, "agent_score": 0, "opponent_score": 0, "result": "Tie"},
        ]
    )

    summary = _summarize_session_groups(sessions, ["player_name", "player_id"])

    a_row = summary[summary["player_id"] == "a"].iloc[0]
    assert a_row["games"] == 2
    assert a_row["wins"] == 1
    assert a_row["losses"] == 1
    assert a_row["ties"] == 0
    assert a_row["win_rate"] == 0.5


def test_filter_sessions_handles_player_bot_and_game_filters():
    sessions = pd.DataFrame(
        [
            {"game_id": "g1", "player_id": "guest_123", "bot_name": "random", "game_type": "RPS+"},
            {"game_id": "g2", "player_id": "user_a", "bot_name": "heuristic", "game_type": "Chess"},
            {"game_id": "g3", "player_id": "user_a", "bot_name": "random", "game_type": "Chess"},
        ]
    )

    filtered = _filter_sessions(
        sessions,
        players=["user_a"],
        bots=["random"],
        games=["Chess"],
    )

    assert filtered["game_id"].tolist() == ["g3"]


def test_init_db_migrates_legacy_games_table_without_losing_rows(tmp_path):
    db = tmp_path / "legacy.db"
    conn = connect(db)
    try:
        conn.execute(
            """CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                opponent_name TEXT NOT NULL,
                seed INTEGER,
                started_at TEXT NOT NULL,
                n_turns INTEGER NOT NULL,
                agent_score INTEGER NOT NULL,
                opponent_score INTEGER NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO games
               (game_id, agent_name, opponent_name, seed, started_at, n_turns, agent_score, opponent_score)
               VALUES ('g1', 'guest_abc', 'random', 1, '2026-01-01T00:00:00', 10, 3, 1)"""
        )
        conn.commit()
    finally:
        conn.close()

    init_db(db)

    conn = connect(db)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
        row = conn.execute("SELECT game_id, agent_name, opponent_name, game_type FROM games WHERE game_id = 'g1'").fetchone()
    finally:
        conn.close()

    assert "game_type" in cols
    assert row == ("g1", "guest_abc", "random", None)


def test_load_game_sessions_handles_guest_and_malicious_like_names(tmp_path):
    db = tmp_path / "analytics.db"
    init_db(db)
    conn = connect(db)
    try:
        conn.execute(
            """INSERT INTO games
               (game_id, agent_name, opponent_name, game_type, seed, started_at, n_turns, agent_score, opponent_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("g1", "guest_'; DROP TABLE games; --", "profile_counter", "Chess", None, "2026-01-01T00:00:00", 14, 1, 0),
        )
        conn.commit()
        sessions = _load_game_sessions(conn)
    finally:
        conn.close()

    assert len(sessions) == 1
    assert sessions.iloc[0]["player_id"] == "guest_'; DROP TABLE games; --"
    assert sessions.iloc[0]["player_name"] == "guest_'; DROP TABLE games; --"
    assert sessions.iloc[0]["game_type"] == "Chess"
    assert sessions.iloc[0]["result"] == "Win"


def test_canonical_game_type_normalizes_common_variants():
    assert canonical_game_type("chess") == "Chess"
    assert canonical_game_type("ChessEnv") == "Chess"
    assert canonical_game_type("connect_four") == "Connect Four"
    assert canonical_game_type("RPS+") == "RPS+"


def test_infer_game_type_handles_common_legacy_patterns():
    assert infer_game_type(["ROCK", "PAPER"], ["SCISSORS"], 10, "scripted") == "RPS+"
    assert infer_game_type(["0", "4", "8"], ["1", "2"], 5, "rl") == "Tic-Tac-Toe"
    assert infer_game_type(["0", "2", "6"], ["1", "5"], 12, "random") == "Connect Four"
    assert infer_game_type(["10", "23", "54"], ["5", "44"], 8, "bcrl") == "Othello"
    assert infer_game_type(["130", "194"], ["67"], 6, "heuristic") == "Checkers"


def test_backfill_game_types_updates_legacy_rows(tmp_path):
    db = tmp_path / "backfill.db"
    init_db(db)
    conn = connect(db)
    try:
        conn.execute(
            """INSERT INTO games
               (game_id, agent_name, opponent_name, game_type, seed, started_at, n_turns, agent_score, opponent_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("g1", "guest_1", "scripted", None, None, "2026-01-01T00:00:00", 20, 3, 1),
        )
        conn.executemany(
            """INSERT INTO rounds
               (game_id, turn, agent_move, agent_move_name, opponent_move, opponent_move_name, outcome, agent_energy_after, opponent_energy_after)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("g1", 1, 0, "ROCK", 1, "PAPER", -1, 3, 3),
                ("g1", 2, 5, "RECHARGE", 0, "ROCK", 1, 4, 3),
            ],
        )
        result = backfill_game_types(conn)
        row = conn.execute("SELECT game_type FROM games WHERE game_id = 'g1'").fetchone()
    finally:
        conn.close()

    assert result["updated"] >= 1
    assert row[0] == "RPS+"


def test_collect_supports_non_rps_game_types(tmp_path):
    db = tmp_path / "collector.db"
    written = collect(
        db_path=db,
        n_games=2,
        max_turns=5,
        policy_name="random",
        seed=7,
        agent_name="collector_test",
        game_type="Tic-Tac-Toe",
    )
    assert written == 2

    conn = connect(db)
    try:
        rows = conn.execute("SELECT DISTINCT game_type FROM games").fetchall()
    finally:
        conn.close()

    assert rows == [("Tic-Tac-Toe",)]
