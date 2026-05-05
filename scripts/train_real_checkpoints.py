from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents import AGENT_REGISTRY, ImpostorTrainer
from data.schemas import connect, init_db
from evaluation.runner import run_benchmark


def _human_player_ids(db_path: Path, min_games: int, limit: int | None) -> list[tuple[str, int]]:
    bot_names = set(AGENT_REGISTRY.keys()) | {"random", "heuristic", "optimal", "agentic"}
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT agent_name, COUNT(*) AS n_games
            FROM games
            WHERE agent_name IS NOT NULL AND agent_name != ''
            GROUP BY agent_name
            ORDER BY n_games DESC
            """
        ).fetchall()
    finally:
        conn.close()

    filtered = [(name, int(n_games)) for name, n_games in rows if name not in bot_names and int(n_games) >= min_games]
    return filtered[:limit] if limit is not None else filtered


def _run_module(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local real checkpoints and verify them.")
    parser.add_argument("--db", default="data/game_data.db")
    parser.add_argument("--ppo-timesteps", type=int, default=15000)
    parser.add_argument("--bcrl-timesteps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--impostor-epochs", type=int, default=12)
    parser.add_argument("--min-player-games", type=int, default=2)
    parser.add_argument("--player-limit", type=int, default=5)
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-bcrl", action="store_true")
    parser.add_argument("--skip-impostor", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    init_db(db_path)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "db": str(db_path),
        "seed": args.seed,
        "trained": {},
    }

    if not args.skip_ppo:
        _run_module(
            [
                "-m",
                "agents.rl.train",
                "--timesteps",
                str(args.ppo_timesteps),
                "--output",
                "checkpoints/ppo_real",
                "--seed",
                str(args.seed),
                "--eval-freq",
                str(max(1000, args.ppo_timesteps // 2)),
                "--eval-episodes",
                "20",
            ]
        )
        summary["trained"]["ppo"] = {"checkpoint": "checkpoints/ppo_real.zip", "timesteps": args.ppo_timesteps}

    if not args.skip_bcrl:
        _run_module(
            [
                "-m",
                "agents.rl.bc_rl",
                "--timesteps",
                str(args.bcrl_timesteps),
                "--output",
                "checkpoints/bc_rl_real",
                "--seed",
                str(args.seed),
            ]
        )
        summary["trained"]["bc_rl"] = {"checkpoint": "checkpoints/bc_rl_real.zip", "timesteps": args.bcrl_timesteps}

    if not args.skip_impostor:
        trainer = ImpostorTrainer(db_path=db_path, checkpoint_dir=checkpoints_dir / "impostor")
        player_rows = _human_player_ids(db_path, min_games=args.min_player_games, limit=args.player_limit)
        impostor_results = []
        for player_id, n_games in player_rows:
            trained = trainer.train_all(player_id)
            impostor_results.append(
                {
                    "player_id": player_id,
                    "games": n_games,
                    "ngram": trained["ngram"]["result"].stats,
                    "lstm": trained["lstm"]["result"].stats,
                }
            )
        summary["trained"]["impostor"] = impostor_results

    if not args.skip_benchmark:
        result = run_benchmark(
            rounds=60,
            agents=["random", "heuristic", "rl", "bc_rl", "profile_counter", "adaptive_router"],
            n_seeds=2,
            games=["RPS+"],
            db_path=db_path,
        )
        summary["benchmark"] = {
            "run_id": result["run_id"],
            "scores": result["agents"],
            "game_breakdown": result["game_breakdown"],
        }

    summary_path = checkpoints_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
