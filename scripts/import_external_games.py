from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from data.importers import game_balance_report, generate_tictactoe_games, import_chess_pgn, import_moves_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Import or generate external game datasets into AgentBench SQLite.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chess = subparsers.add_parser("chess-pgn")
    chess.add_argument("--db", default="data/game_data.db")
    chess.add_argument("--pgn", required=True)
    chess.add_argument("--max-games", type=int, default=None)

    moves_csv = subparsers.add_parser("moves-csv")
    moves_csv.add_argument("--db", default="data/game_data.db")
    moves_csv.add_argument("--csv", required=True)
    moves_csv.add_argument("--game", required=True)
    moves_csv.add_argument("--moves-column", default="moves")
    moves_csv.add_argument("--result-column", default="result")
    moves_csv.add_argument("--player-column", default="player_id")
    moves_csv.add_argument("--bot-column", default="opponent_name")
    moves_csv.add_argument("--delimiter", default=" ")
    moves_csv.add_argument("--max-games", type=int, default=None)

    ttt = subparsers.add_parser("tictactoe-generate")
    ttt.add_argument("--db", default="data/game_data.db")
    ttt.add_argument("--limit", type=int, default=None)

    report = subparsers.add_parser("report-balance")
    report.add_argument("--db", default="data/game_data.db")

    args = parser.parse_args()

    if args.command == "chess-pgn":
        n = import_chess_pgn(args.db, args.pgn, max_games=args.max_games)
        print(f"Imported {n} chess games from {args.pgn}")
    elif args.command == "moves-csv":
        n = import_moves_csv(
            args.db,
            args.csv,
            game_type=args.game,
            moves_column=args.moves_column,
            result_column=args.result_column,
            player_column=args.player_column,
            bot_column=args.bot_column,
            delimiter=args.delimiter,
            max_games=args.max_games,
        )
        print(f"Imported {n} {args.game} games from {args.csv}")
    elif args.command == "tictactoe-generate":
        n = generate_tictactoe_games(args.db, limit=args.limit)
        print(f"Generated {n} Tic-Tac-Toe games into {args.db}")
    elif args.command == "report-balance":
        for row in game_balance_report(args.db):
            print(f"{row['game_type']}: {row['n_games']} ({row['fraction']:.2%})")


if __name__ == "__main__":
    main()
