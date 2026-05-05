"""Backfill helpers for legacy game metadata."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable


RPS_MOVE_NAMES = {"ROCK", "PAPER", "SCISSORS", "LIZARD", "POWER", "RECHARGE"}


def canonical_game_type(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    mapping = {
        "rps+": "RPS+",
        "rpsplusenv": "RPS+",
        "rps_plus": "RPS+",
        "tic-tac-toe": "Tic-Tac-Toe",
        "tic_tac_toe": "Tic-Tac-Toe",
        "tictactoeenv": "Tic-Tac-Toe",
        "connect four": "Connect Four",
        "connect_four": "Connect Four",
        "connectfourenv": "Connect Four",
        "chess": "Chess",
        "chessenv": "Chess",
        "othello": "Othello",
        "othelloenv": "Othello",
        "checkers": "Checkers",
        "checkersenv": "Checkers",
        "war": "War",
        "warenv": "War",
    }
    return mapping.get(normalized, str(value))


def infer_game_type(
    agent_move_names: Iterable[str],
    opponent_move_names: Iterable[str],
    n_turns: int,
    opponent_name: str | None = None,
) -> str | None:
    names = {str(name) for name in agent_move_names if name is not None}
    opp_names = {str(name) for name in opponent_move_names if name is not None}
    combined = {name for name in names | opp_names if name != "-1"}

    if combined and combined.issubset(RPS_MOVE_NAMES):
        return "RPS+"

    numeric_values: list[int] = []
    for name in combined:
        try:
            numeric_values.append(int(name))
        except Exception:
            return None

    if not numeric_values:
        if opponent_name == "scripted":
            return "RPS+"
        return None

    max_value = max(numeric_values)
    min_value = min(numeric_values)
    if min_value < 0:
        return None

    if max_value <= 6:
        if n_turns > 9:
            return "Connect Four"
        return "Tic-Tac-Toe"
    if max_value <= 8:
        return "Tic-Tac-Toe"
    if max_value <= 63:
        return "Othello"
    return "Checkers"


def backfill_game_types(conn: sqlite3.Connection) -> dict[str, int]:
    updated = 0
    unchanged = 0
    unresolved = 0

    rows = conn.execute(
        """SELECT game_id, opponent_name, n_turns, game_type
           FROM games
           ORDER BY started_at"""
    ).fetchall()

    for game_id, opponent_name, n_turns, current_game_type in rows:
        round_rows = conn.execute(
            """SELECT agent_move_name, opponent_move_name
               FROM rounds
               WHERE game_id = ?""",
            (game_id,),
        ).fetchall()
        agent_names = [row[0] for row in round_rows]
        opp_names = [row[1] for row in round_rows]
        inferred = infer_game_type(agent_names, opp_names, int(n_turns or 0), opponent_name)
        current = canonical_game_type(current_game_type) if current_game_type else None

        if inferred is None:
            if current is None:
                unresolved += 1
            elif current != current_game_type:
                conn.execute("UPDATE games SET game_type = ? WHERE game_id = ?", (current, game_id))
                updated += 1
            else:
                unchanged += 1
            continue

        if current != inferred:
            conn.execute("UPDATE games SET game_type = ? WHERE game_id = ?", (inferred, game_id))
            updated += 1
        else:
            unchanged += 1

    conn.commit()
    return {"updated": updated, "unchanged": unchanged, "unresolved": unresolved}
