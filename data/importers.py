"""Import external game datasets into the AgentBench SQLite schema."""

from __future__ import annotations

import datetime as dt
import itertools
import uuid
from pathlib import Path

from data.backfill import canonical_game_type
from data.collector import insert_game
from data.schemas import connect, init_db


def _parse_result_token(result: str | None) -> tuple[int, int]:
    token = (result or "").strip().lower()
    if token in {"1-0", "white", "player", "win", "x"}:
        return 1, 0
    if token in {"0-1", "black", "opponent", "loss", "o"}:
        return 0, 1
    return 0, 0


def _generic_round_records(game_id: str, moves: list[int], final_outcome: int = 0) -> list[dict]:
    records: list[dict] = []
    turn = 0
    for idx in range(0, len(moves), 2):
        turn += 1
        agent_move = int(moves[idx])
        opponent_move = int(moves[idx + 1]) if idx + 1 < len(moves) else -1
        outcome = final_outcome if idx + 2 >= len(moves) else 0
        records.append(
            {
                "game_id": game_id,
                "turn": turn,
                "agent_move": agent_move,
                "agent_move_name": str(agent_move),
                "opponent_move": opponent_move,
                "opponent_move_name": str(opponent_move),
                "outcome": outcome,
                "agent_energy_after": 0,
                "opponent_energy_after": 0,
            }
        )
    return records


def import_chess_pgn(db_path: str | Path, pgn_path: str | Path, max_games: int | None = None) -> int:
    try:
        import chess.pgn
    except ImportError as exc:
        raise ImportError("python-chess is required for chess PGN import") from exc

    init_db(db_path)
    imported = 0
    conn = connect(db_path)
    try:
        with open(pgn_path, "r", encoding="utf-8") as handle:
            while True:
                if max_games is not None and imported >= max_games:
                    break
                game = chess.pgn.read_game(handle)
                if game is None:
                    break

                board = game.board()
                moves = list(game.mainline_moves())
                round_records = []
                turn = 0
                idx = 0
                while idx < len(moves):
                    white_move = moves[idx]
                    legal_before = list(board.legal_moves)
                    white_idx = legal_before.index(white_move)
                    board.push(white_move)
                    black_idx = -1
                    if idx + 1 < len(moves):
                        black_move = moves[idx + 1]
                        legal_before_black = list(board.legal_moves)
                        black_idx = legal_before_black.index(black_move)
                        board.push(black_move)
                    turn += 1
                    round_records.append(
                        {
                            "game_id": "",
                            "turn": turn,
                            "agent_move": white_idx,
                            "agent_move_name": white_move.uci(),
                            "opponent_move": black_idx,
                            "opponent_move_name": moves[idx + 1].uci() if idx + 1 < len(moves) else "-1",
                            "outcome": 0,
                            "agent_energy_after": 0,
                            "opponent_energy_after": 0,
                        }
                    )
                    idx += 2

                result = game.headers.get("Result", "1/2-1/2")
                agent_score, opponent_score = _parse_result_token(result)
                final_outcome = 1 if agent_score > opponent_score else -1 if opponent_score > agent_score else 0
                if round_records:
                    round_records[-1]["outcome"] = final_outcome

                game_id = uuid.uuid4().hex
                for record in round_records:
                    record["game_id"] = game_id
                insert_game(
                    conn,
                    game_id=game_id,
                    agent_name=game.headers.get("White", "lichess_white"),
                    opponent_name=game.headers.get("Black", "lichess_black"),
                    game_type="Chess",
                    seed=None,
                    n_turns=max(1, len(moves)),
                    agent_score=agent_score,
                    opponent_score=opponent_score,
                    rounds=round_records,
                )
                imported += 1
        conn.commit()
    finally:
        conn.close()
    return imported


def import_moves_csv(
    db_path: str | Path,
    csv_path: str | Path,
    *,
    game_type: str,
    moves_column: str = "moves",
    result_column: str = "result",
    player_column: str = "player_id",
    bot_column: str = "opponent_name",
    delimiter: str = " ",
    max_games: int | None = None,
) -> int:
    import csv

    game_type = canonical_game_type(game_type) or game_type
    init_db(db_path)
    imported = 0
    conn = connect(db_path)
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if max_games is not None and imported >= max_games:
                    break
                moves_raw = (row.get(moves_column) or "").strip()
                if not moves_raw:
                    continue
                moves = [int(part) for part in moves_raw.split(delimiter) if part.strip()]
                agent_score, opponent_score = _parse_result_token(row.get(result_column))
                final_outcome = 1 if agent_score > opponent_score else -1 if opponent_score > agent_score else 0
                game_id = uuid.uuid4().hex
                rounds = _generic_round_records(game_id, moves, final_outcome=final_outcome)
                insert_game(
                    conn,
                    game_id=game_id,
                    agent_name=row.get(player_column) or f"{game_type.lower()}_player",
                    opponent_name=row.get(bot_column) or "dataset_opponent",
                    game_type=game_type,
                    seed=None,
                    n_turns=len(moves),
                    agent_score=agent_score,
                    opponent_score=opponent_score,
                    rounds=rounds,
                )
                imported += 1
        conn.commit()
    finally:
        conn.close()
    return imported


def generate_tictactoe_games(
    db_path: str | Path,
    *,
    limit: int | None = None,
    agent_name: str = "generated_ttt",
    opponent_name: str = "generated_ttt",
) -> int:
    init_db(db_path)
    imported = 0
    conn = connect(db_path)

    WIN_PATTERNS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]

    def winner(board: tuple[int, ...]) -> int:
        for a, b, c in WIN_PATTERNS:
            s = board[a] + board[b] + board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    def walk(board: list[int], player: int, moves: list[int]):
        nonlocal imported
        result = winner(tuple(board))
        empties = [idx for idx, value in enumerate(board) if value == 0]
        if result != 0 or not empties:
            game_id = uuid.uuid4().hex
            agent_score = 1 if result == 1 else 0
            opponent_score = 1 if result == -1 else 0
            rounds = _generic_round_records(game_id, moves, final_outcome=result)
            insert_game(
                conn,
                game_id=game_id,
                agent_name=agent_name,
                opponent_name=opponent_name,
                game_type="Tic-Tac-Toe",
                seed=None,
                n_turns=len(moves),
                agent_score=agent_score,
                opponent_score=opponent_score,
                rounds=rounds,
            )
            imported += 1
            return

        if limit is not None and imported >= limit:
            return

        for idx in empties:
            board[idx] = player
            moves.append(idx)
            walk(board, -player, moves)
            moves.pop()
            board[idx] = 0
            if limit is not None and imported >= limit:
                return

    try:
        walk([0] * 9, 1, [])
        conn.commit()
    finally:
        conn.close()
    return imported


def game_balance_report(db_path: str | Path) -> list[dict[str, object]]:
    init_db(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """SELECT COALESCE(NULLIF(game_type, ''), 'Unknown / Legacy') AS game_type,
                      COUNT(*) AS n_games
               FROM games
               GROUP BY 1
               ORDER BY n_games DESC"""
        ).fetchall()
    finally:
        conn.close()

    total = sum(int(row[1]) for row in rows) or 1
    return [
        {
            "game_type": row[0],
            "n_games": int(row[1]),
            "fraction": float(row[1]) / float(total),
        }
        for row in rows
    ]
