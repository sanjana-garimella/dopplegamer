"""Export benchmark tables from SQLite to CSV (and Parquet when pyarrow is available).

Usage:
    python scripts/export_results.py --db data/game_data.db --out results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data.schemas import connect


TABLES = (
    "inference_benchmarks",
    "agent_results",
    "games",
    "rounds",
)


def export(db_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    conn = connect(db_path)
    try:
        for table in TABLES:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            except Exception:
                continue
            if df.empty:
                continue
            csv_path = out_dir / f"{table}.csv"
            df.to_csv(csv_path, index=False)
            written.append(csv_path)
            try:
                parquet_path = out_dir / f"{table}.parquet"
                df.to_parquet(parquet_path, index=False)
                written.append(parquet_path)
            except Exception:
                pass
    finally:
        conn.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Doppelgamer SQLite tables.")
    parser.add_argument("--db", default="data/game_data.db")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    paths = export(Path(args.db), Path(args.out))
    if not paths:
        print("No rows exported.")
        return
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
