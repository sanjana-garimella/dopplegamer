from __future__ import annotations

import numpy as np

from agents.adaptive_router import AdaptiveRouterAgent
from data.collector import insert_game
from data.schemas import init_db
from environments.rps_plus import Move


def test_adaptive_router_starts_with_sft_for_rps(tmp_path):
    agent = AdaptiveRouterAgent(player_id="player_a", db_path=tmp_path / "games.db", seed=0)

    action = agent.act(
        obs=np.array([0.6, 0.6, 0.0], dtype=np.float32),
        info={"legal_moves": list(range(6))},
    )

    assert action in range(6)
    assert agent.last_selected_expert == "sft"


def test_adaptive_router_switches_to_ppo_after_losses(tmp_path):
    agent = AdaptiveRouterAgent(player_id="player_a", db_path=tmp_path / "games.db", seed=0)
    for _ in range(4):
        agent.observe(int(Move.ROCK), int(Move.PAPER), -1)

    action = agent.act(
        obs=np.array([0.6, 0.6, 0.0], dtype=np.float32),
        info={"legal_moves": list(range(6))},
    )

    assert action in range(6)
    assert agent.last_selected_expert == "ppo"


def test_adaptive_router_uses_profile_counter_when_history_exists_and_results_drop(tmp_path):
    db = tmp_path / "games.db"
    init_db(db)
    records = [
        {
            "game_id": "g1",
            "turn": 1,
            "agent_move": int(Move.ROCK),
            "agent_move_name": "ROCK",
            "opponent_move": int(Move.SCISSORS),
            "opponent_move_name": "SCISSORS",
            "outcome": 1,
            "agent_energy_after": 3,
            "opponent_energy_after": 3,
        },
        {
            "game_id": "g1",
            "turn": 2,
            "agent_move": int(Move.ROCK),
            "agent_move_name": "ROCK",
            "opponent_move": int(Move.SCISSORS),
            "opponent_move_name": "SCISSORS",
            "outcome": 1,
            "agent_energy_after": 3,
            "opponent_energy_after": 3,
        },
    ]

    import sqlite3

    conn = sqlite3.connect(db)
    try:
        insert_game(
            conn,
            game_id="g1",
            agent_name="player_a",
            opponent_name="adaptive_router",
            game_type="RPS+",
            seed=None,
            n_turns=2,
            agent_score=2,
            opponent_score=0,
            rounds=records,
        )
        conn.commit()
    finally:
        conn.close()

    agent = AdaptiveRouterAgent(player_id="player_a", db_path=db, seed=0)
    for _ in range(3):
        agent.observe(int(Move.ROCK), int(Move.PAPER), -1)

    action = agent.act(
        obs=np.array([0.6, 0.6, 0.0], dtype=np.float32),
        info={"legal_moves": list(range(6))},
    )

    assert action in range(6)
    assert agent.last_selected_expert == "profile_counter"


def test_adaptive_router_prefers_history_aware_expert_for_non_rps_games(tmp_path):
    agent = AdaptiveRouterAgent(player_id="player_a", db_path=tmp_path / "games.db", seed=0)

    action = agent.act(
        obs=np.zeros(64, dtype=np.float32),
        info={"legal_moves": [19, 26, 37, 44]},
    )

    assert action in {19, 26, 37, 44}
    assert agent.last_selected_expert == "safe_fallback"
