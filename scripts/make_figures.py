"""Regenerate simple paper figures from exported or live SQLite results.

Usage:
    python scripts/make_figures.py --db data/game_data.db --out figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _save_latency_bars(df: pd.DataFrame, out: Path) -> Path | None:
    if df.empty or "engine" not in df.columns:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return None

    agg = df.groupby("engine", as_index=False).agg(
        ttft_ms=("ttft_ms", "mean"),
        tpot_ms=("tpot_ms", "mean"),
        total_latency_ms=("total_latency_ms", "mean"),
        kv_cache_mb=("kv_cache_mb", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(agg["engine"], agg["total_latency_ms"], color="#c44e52")
    axes[0].set_title("Mean total latency (ms)")
    axes[0].set_xlabel("engine")
    axes[1].bar(agg["engine"], agg["kv_cache_mb"], color="#4c72b0")
    axes[1].set_title("Mean KV cache (MB)")
    axes[1].set_xlabel("engine")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make simple benchmark figures.")
    parser.add_argument("--db", default="data/game_data.db")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    from data.schemas import connect

    conn = connect(args.db)
    try:
        df = pd.read_sql_query("SELECT * FROM inference_benchmarks", conn)
    finally:
        conn.close()

    path = _save_latency_bars(df, Path(args.out) / "latency_kv.png")
    if path:
        print(path)
    else:
        print("No inference_benchmarks rows to plot.")


if __name__ == "__main__":
    main()
