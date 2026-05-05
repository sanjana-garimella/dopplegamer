from data.collector import insert_game
from data.schemas import init_db
from agents.profile_counter import ProfileCounterAgent
from environments.rps_plus import Move


def test_profile_counter_trains_from_saved_player_history(tmp_path):
    db = tmp_path / "games.db"
    init_db(db)
    records = [
        {
            "game_id": "g1",
            "turn": 1,
            "agent_move": int(Move.ROCK),
            "agent_move_name": "ROCK",
            "opponent_move": int(Move.PAPER),
            "opponent_move_name": "PAPER",
            "outcome": -1,
            "agent_energy_after": 3,
            "opponent_energy_after": 3,
        },
        {
            "game_id": "g1",
            "turn": 2,
            "agent_move": int(Move.ROCK),
            "agent_move_name": "ROCK",
            "opponent_move": int(Move.PAPER),
            "opponent_move_name": "PAPER",
            "outcome": -1,
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
            opponent_name="profile_counter",
            game_type="RPS+",
            seed=None,
            n_turns=2,
            agent_score=0,
            opponent_score=2,
            rounds=records,
        )
        conn.commit()
    finally:
        conn.close()

    agent = ProfileCounterAgent(player_id="player_a", db_path=db)
    action = agent.act(
        obs=__import__("numpy").array([0.6, 0.6, 0.1], dtype=__import__("numpy").float32),
        info={"legal_moves": list(range(6))},
    )

    assert action == int(Move.PAPER)


def test_profile_counter_uses_current_move_when_available(tmp_path):
    agent = ProfileCounterAgent(player_id="missing", db_path=tmp_path / "games.db")
    agent.set_current_player_move(int(Move.SCISSORS))

    action = agent.act(
        obs=__import__("numpy").array([0.6, 0.6, 0.1], dtype=__import__("numpy").float32),
        info={"legal_moves": list(range(6))},
    )

    assert action == int(Move.ROCK)
