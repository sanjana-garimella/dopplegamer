"""Game data collection: run episodes and persist them to SQLite.

Usage:
    python -m data.collector --rounds 500 --policy random --db data/game_data.db
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from data.schemas import init_db, connect
from data.backfill import canonical_game_type
from environments import CheckersEnv, ChessEnv, ConnectFourEnv, GomokuEnv, NimEnv, OthelloEnv, RPSPlusEnv, TicTacToeEnv, WarEnv
from environments.utils import history_to_records, random_policy, run_episode


PolicyFactory = Callable[[int | None], Callable[[np.ndarray, dict], int]]

POLICY_REGISTRY: dict[str, PolicyFactory] = {
    "random": random_policy,
}

ENV_REGISTRY = {
    "RPS+": RPSPlusEnv,
    "Tic-Tac-Toe": TicTacToeEnv,
    "Connect Four": ConnectFourEnv,
    "Chess": ChessEnv,
    "Othello": OthelloEnv,
    "Checkers": CheckersEnv,
    "Gomoku": GomokuEnv,
    "Nim": NimEnv,
    "War": WarEnv,
}


def _new_env(game_type: str, max_turns: int, seed: int | None):
    game_type = canonical_game_type(game_type) or game_type
    env_cls = ENV_REGISTRY[game_type]
    if game_type == "RPS+":
        return env_cls(max_turns=max_turns, seed=seed)
    return env_cls(max_moves=max_turns, seed=seed)


def _generic_history_to_records(history, game_id: str) -> list[dict]:
    records = []
    for idx, round_obj in enumerate(history):
        turn = int(getattr(round_obj, "turn", idx + 1))
        agent_move = int(getattr(round_obj, "agent_move", -1))
        opponent_move = int(getattr(round_obj, "opponent_move", -1))
        agent_move_name = getattr(getattr(round_obj, "agent_move", None), "name", str(agent_move))
        opponent_move_name = getattr(getattr(round_obj, "opponent_move", None), "name", str(opponent_move))
        records.append(
            {
                "game_id": game_id,
                "turn": turn,
                "agent_move": agent_move,
                "agent_move_name": str(agent_move_name),
                "opponent_move": opponent_move,
                "opponent_move_name": str(opponent_move_name),
                "outcome": int(getattr(round_obj, "outcome", 0)),
                "agent_energy_after": int(getattr(round_obj, "agent_energy_after", 0)),
                "opponent_energy_after": int(getattr(round_obj, "opponent_energy_after", 0)),
            }
        )
    return records


def insert_game(
    conn: sqlite3.Connection,
    game_id: str,
    agent_name: str,
    opponent_name: str,
    game_type: str | None,
    seed: int | None,
    n_turns: int,
    agent_score: int,
    opponent_score: int,
    rounds: list[dict],
    *,
    metadata: dict | None = None,
) -> None:
    metadata = metadata or {}
    conn.execute(
        """INSERT INTO games
           (game_id, agent_name, opponent_name, game_type, seed, started_at, n_turns, agent_score, opponent_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            game_id,
            agent_name,
            opponent_name,
            game_type,
            seed,
            dt.datetime.now(dt.timezone.utc).isoformat(),
            n_turns,
            agent_score,
            opponent_score,
        ),
    )
    conn.executemany(
        """INSERT INTO rounds
           (game_id, turn, agent_move, agent_move_name, opponent_move,
            opponent_move_name, outcome, agent_energy_after, opponent_energy_after)
           VALUES (:game_id, :turn, :agent_move, :agent_move_name, :opponent_move,
                   :opponent_move_name, :outcome, :agent_energy_after, :opponent_energy_after)""",
        rounds,
    )


def persist_counterfactual_summary(
    conn: sqlite3.Connection,
    *,
    replay_id: str,
    source_game_id: str,
    player_id: str,
    game_type: str,
    baseline_agent: str,
    clone_agent: str,
    source_result: str,
    replay_summary_json: str,
    created_at: str,
) -> None:
    conn.execute(
        """INSERT INTO counterfactual_replays
           (replay_id, source_game_id, player_id, game_type, baseline_agent, clone_agent,
            source_result, replay_summary_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            replay_id,
            source_game_id,
            player_id,
            game_type,
            baseline_agent,
            clone_agent,
            source_result,
            replay_summary_json,
            created_at,
        ),
    )


def collect(
    db_path: str | Path,
    n_games: int,
    max_turns: int,
    policy_name: str = "random",
    seed: int | None = None,
    agent_name: str | None = None,
    game_type: str = "RPS+",
) -> int:
    init_db(db_path)
    factory = POLICY_REGISTRY[policy_name]
    agent_name = agent_name or policy_name
    game_type = canonical_game_type(game_type) or game_type
    if game_type not in ENV_REGISTRY:
        raise ValueError(f"Unsupported game type: {game_type}. Available: {list(ENV_REGISTRY)}")
    conn = connect(db_path)
    try:
        rng = np.random.default_rng(seed)
        for i in range(n_games):
            game_seed = int(rng.integers(0, 2**31 - 1))
            env = _new_env(game_type, max_turns, game_seed)
            policy = factory(game_seed)
            history = run_episode(env, policy, seed=game_seed)
            game_id = uuid.uuid4().hex
            records = history_to_records(history, game_id) if game_type == "RPS+" else _generic_history_to_records(history, game_id)
            insert_game(
                conn,
                game_id=game_id,
                agent_name=agent_name,
                opponent_name="scripted",
                game_type=game_type,
                seed=game_seed,
                n_turns=env.state.turn,
                agent_score=env.state.agent_score,
                opponent_score=env.state.opponent_score,
                rounds=records,
            )
        conn.commit()
    finally:
        conn.close()
    return n_games


def main() -> None:
    p = argparse.ArgumentParser(description="Collect RPS+ gameplay data into SQLite.")
    p.add_argument("--db", default="data/game_data.db")
    p.add_argument("--rounds", type=int, default=500, help="number of games to play")
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--game", default="RPS+", choices=list(ENV_REGISTRY))
    p.add_argument("--policy", default="random", choices=list(POLICY_REGISTRY))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--agent-name", default=None)
    args = p.parse_args()

    n = collect(
        db_path=args.db,
        n_games=args.rounds,
        max_turns=args.max_turns,
        policy_name=args.policy,
        seed=args.seed,
        agent_name=args.agent_name,
        game_type=args.game,
    )
    print(f"Wrote {n} {args.game} games to {args.db}")


if __name__ == "__main__":
    main()
