from __future__ import annotations

from data.importers import game_balance_report, generate_tictactoe_games, import_chess_pgn, import_moves_csv
from data.schemas import connect


def test_import_chess_pgn_writes_chess_games(tmp_path):
    db = tmp_path / "games.db"
    pgn = tmp_path / "sample.pgn"
    pgn.write_text(
        '[Event "Sample"]\n'
        '[White "Alice"]\n'
        '[Black "Bob"]\n'
        '[Result "1-0"]\n\n'
        '1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n',
        encoding="utf-8",
    )

    imported = import_chess_pgn(db, pgn, max_games=1)
    assert imported == 1

    conn = connect(db)
    try:
        rows = conn.execute("SELECT game_type, agent_name, opponent_name FROM games").fetchall()
    finally:
        conn.close()

    assert rows == [("Chess", "Alice", "Bob")]


def test_import_moves_csv_writes_generic_games(tmp_path):
    db = tmp_path / "games.db"
    csv_path = tmp_path / "othello.csv"
    csv_path.write_text(
        "player_id,opponent_name,result,moves\n"
        "p1,bot_a,1-0,19 26 18 34\n"
        "p2,bot_b,0-1,20 29 21 37\n",
        encoding="utf-8",
    )

    imported = import_moves_csv(db, csv_path, game_type="Othello")
    assert imported == 2

    conn = connect(db)
    try:
        rows = conn.execute("SELECT DISTINCT game_type FROM games ORDER BY game_type").fetchall()
    finally:
        conn.close()

    assert rows == [("Othello",)]


def test_generate_tictactoe_games_and_report_balance(tmp_path):
    db = tmp_path / "games.db"
    generated = generate_tictactoe_games(db, limit=12)
    assert generated == 12

    report = game_balance_report(db)
    assert report[0]["game_type"] == "Tic-Tac-Toe"
    assert report[0]["n_games"] == 12
