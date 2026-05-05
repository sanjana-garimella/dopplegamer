from pathlib import Path

from evaluation.runner import run_benchmark
from data.schemas import connect


def test_run_benchmark_writes_rows(tmp_path: Path):
    db = tmp_path / "bench.db"
    result = run_benchmark(rounds=3, engines=["baseline"], agents=["sft", "agentic"], db_path=db)
    assert "run_id" in result
    assert db.exists()


def test_run_benchmark_returns_game_breakdown_for_multiple_games(tmp_path: Path):
    db = tmp_path / "bench_multi.db"
    result = run_benchmark(
        rounds=3,
        engines=["baseline"],
        agents=["random"],
        games=["RPS+", "Tic-Tac-Toe"],
        n_seeds=2,
        db_path=db,
    )
    breakdown = result["game_breakdown"]
    assert any(row["game_type"] == "RPS+" for row in breakdown)
    assert any(row["game_type"] == "Tic-Tac-Toe" for row in breakdown)
    assert all("win_rate_stats" in row for row in breakdown)


def test_run_benchmark_handles_rps_agents_on_non_rps_games(tmp_path: Path):
    db = tmp_path / "bench_non_rps_agents.db"
    result = run_benchmark(
        rounds=3,
        engines=[],
        agents=["sft", "bcrl", "adaptive_router"],
        games=["Tic-Tac-Toe"],
        db_path=db,
    )

    assert len(result["agents"]) == 3
    assert all(row["agent_name"].endswith("::Tic-Tac-Toe") for row in result["agents"])


def test_run_benchmark_honors_empty_agents_list(tmp_path: Path):
    db = tmp_path / "systems_only.db"
    result = run_benchmark(rounds=2, engines=["baseline"], agents=[], db_path=db)

    assert result["agents"] == []
    assert result["game_breakdown"] == []

    conn = connect(db)
    try:
        agent_rows = conn.execute("SELECT COUNT(*) FROM agent_results").fetchone()[0]
        inference_rows = conn.execute("SELECT COUNT(*) FROM inference_benchmarks").fetchone()[0]
    finally:
        conn.close()

    assert agent_rows == 0
    assert inference_rows == 2


def test_run_benchmark_honors_empty_engines_list(tmp_path: Path):
    db = tmp_path / "agents_only.db"
    result = run_benchmark(rounds=2, engines=[], agents=["random"], db_path=db)

    assert result["engines"] == []
    assert len(result["agents"]) == 1

    conn = connect(db)
    try:
        agent_rows = conn.execute("SELECT COUNT(*) FROM agent_results").fetchone()[0]
        inference_rows = conn.execute("SELECT COUNT(*) FROM inference_benchmarks").fetchone()[0]
    finally:
        conn.close()

    assert agent_rows == 2
    assert inference_rows == 0
