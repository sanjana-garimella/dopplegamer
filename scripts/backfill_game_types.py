from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from data.backfill import backfill_game_types
from data.schemas import connect, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill legacy game_type values in the games table.")
    parser.add_argument("--db", default="data/game_data.db")
    args = parser.parse_args()

    init_db(args.db)
    conn = connect(args.db)
    try:
        result = backfill_game_types(conn)
    finally:
        conn.close()

    print(
        f"Backfill complete: updated={result['updated']} "
        f"unchanged={result['unchanged']} unresolved={result['unresolved']}"
    )


if __name__ == "__main__":
    main()
