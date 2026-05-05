"""SQLite schemas for game data + benchmark results."""

from __future__ import annotations

import sqlite3
from pathlib import Path

GAMES_TABLE = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    opponent_name TEXT NOT NULL,
    game_type TEXT,
    seed INTEGER,
    started_at TEXT NOT NULL,
    n_turns INTEGER NOT NULL,
    agent_score INTEGER NOT NULL,
    opponent_score INTEGER NOT NULL
)
"""

ROUNDS_TABLE = """
CREATE TABLE IF NOT EXISTS rounds (
    game_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    agent_move INTEGER NOT NULL,
    agent_move_name TEXT NOT NULL,
    opponent_move INTEGER NOT NULL,
    opponent_move_name TEXT NOT NULL,
    outcome INTEGER NOT NULL,
    agent_energy_after INTEGER NOT NULL,
    opponent_energy_after INTEGER NOT NULL,
    PRIMARY KEY (game_id, turn),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
)
"""

INFERENCE_BENCH_TABLE = """
CREATE TABLE IF NOT EXISTS inference_benchmarks (
    run_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    model TEXT NOT NULL,
    quantization TEXT,
    turn INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    ttft_ms REAL,
    tpot_ms REAL,
    total_latency_ms REAL,
    kv_cache_mb REAL,
    scheduling_overhead_ms REAL,
    PRIMARY KEY (run_id, engine, turn)
)
"""

AGENT_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_results (
    run_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    games_played INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    ties INTEGER NOT NULL,
    win_rate REAL NOT NULL,
    behavioral_fidelity REAL,
    action_kl REAL,
    avg_decision_ms REAL,
    PRIMARY KEY (run_id, agent_name)
)
"""

PLAYER_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS player_profiles (
    player_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    games_played INTEGER NOT NULL DEFAULT 0,
    total_rounds INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    behavioral_signature_json TEXT
)
"""

IMPOSTOR_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS impostor_results (
    run_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    impostor_type TEXT NOT NULL,
    game_type TEXT NOT NULL DEFAULT 'RPS+',
    n_training_rounds INTEGER NOT NULL,
    fidelity_score REAL,
    kl_divergence REAL,
    tvd REAL,
    fool_rate REAL,
    explanation_sample TEXT,
    embedding_json TEXT,
    trained_at TEXT NOT NULL,
    PRIMARY KEY (run_id, player_id, impostor_type, game_type)
)
"""

DETECTION_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS detection_sessions (
    session_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL,
    impostor_type TEXT NOT NULL,
    source_player_id TEXT,
    study_block_id TEXT,
    blind_label TEXT,
    surprisal_summary_json TEXT,
    game_id TEXT,
    detected_as_human INTEGER NOT NULL,
    confidence REAL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
)
"""

BEHAVIORAL_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS behavioral_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    games_played INTEGER NOT NULL DEFAULT 0,
    total_rounds INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    drift_score REAL NOT NULL DEFAULT 0.0,
    signature_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
)
"""

CLONE_AB_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS clone_ab_runs (
    run_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    clone_type TEXT NOT NULL,
    baseline_type TEXT NOT NULL,
    n_games INTEGER NOT NULL DEFAULT 0,
    clone_wins INTEGER NOT NULL DEFAULT 0,
    baseline_wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    clone_fidelity REAL,
    detection_rate REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, block_id)
)
"""

COUNTERFACTUAL_REPLAYS_TABLE = """
CREATE TABLE IF NOT EXISTS counterfactual_replays (
    replay_id TEXT PRIMARY KEY,
    source_game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    baseline_agent TEXT NOT NULL,
    clone_agent TEXT NOT NULL,
    source_result TEXT NOT NULL,
    replay_summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

SHAREABLE_REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS shareable_reports (
    report_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
)
"""

CLONE_LADDER_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS clone_ladder_runs (
    ladder_run_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    rung_index INTEGER NOT NULL DEFAULT 0,
    rung_agent TEXT NOT NULL,
    result TEXT NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    completed_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
)
"""

BLIND_STUDY_BLOCKS_TABLE = """
CREATE TABLE IF NOT EXISTS blind_study_blocks (
    block_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    schedule_json TEXT NOT NULL,
    current_index INTEGER NOT NULL DEFAULT 0,
    revealed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
)
"""

DATASET_SLICES_TABLE = """
CREATE TABLE IF NOT EXISTS dataset_slices (
    slice_id TEXT PRIMARY KEY,
    player_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    slice_name TEXT NOT NULL,
    filter_config_json TEXT NOT NULL,
    n_sessions INTEGER NOT NULL DEFAULT 0,
    n_rounds INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES player_profiles(player_id)
)
"""

ALL_TABLES = [
    GAMES_TABLE,
    ROUNDS_TABLE,
    INFERENCE_BENCH_TABLE,
    AGENT_RESULTS_TABLE,
    PLAYER_PROFILES_TABLE,
    IMPOSTOR_RESULTS_TABLE,
    DETECTION_SESSIONS_TABLE,
]

EXTENDED_TABLES = [
    BEHAVIORAL_SNAPSHOTS_TABLE,
    CLONE_AB_RUNS_TABLE,
    COUNTERFACTUAL_REPLAYS_TABLE,
    SHAREABLE_REPORTS_TABLE,
    CLONE_LADDER_RUNS_TABLE,
    BLIND_STUDY_BLOCKS_TABLE,
    DATASET_SLICES_TABLE,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_games_agent ON games(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_rounds_game ON rounds(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_games_started ON games(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_rounds_turn ON rounds(game_id, turn)",
    "CREATE INDEX IF NOT EXISTS idx_detect_player_type ON detection_sessions(player_id, impostor_type)",
]

EXTENDED_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_impostor_player_game ON impostor_results(player_id, game_type)",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_player_game ON behavioral_snapshots(player_id, game_type, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_ab_runs_player_game ON clone_ab_runs(player_id, game_type, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_counterfactual_player ON counterfactual_replays(player_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_ladder_runs_player_time ON clone_ladder_runs(player_id, completed_at)",
    "CREATE INDEX IF NOT EXISTS idx_blind_blocks_player ON blind_study_blocks(player_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_dataset_slices_player_game ON dataset_slices(player_id, game_type, created_at)",
]


class NamedRow(tuple):
    """Tuple row with sqlite.Row-style column lookup."""

    def __new__(cls, values, columns):
        obj = super().__new__(cls, values)
        obj._columns = columns
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(self._columns[key])
        return super().__getitem__(key)


def _named_row_factory(cursor: sqlite3.Cursor, row: tuple) -> NamedRow:
    columns = {col[0]: idx for idx, col in enumerate(cursor.description)}
    return NamedRow(row, columns)


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = _named_row_factory
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        for ddl in ALL_TABLES:
            conn.execute(ddl)
        for ddl in INDEXES:
            conn.execute(ddl)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()
        }
        if "game_type" not in columns:
            conn.execute("ALTER TABLE games ADD COLUMN game_type TEXT")
        impostor_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(impostor_results)").fetchall()
        }
        if "game_type" not in impostor_columns:
            conn.execute("ALTER TABLE impostor_results ADD COLUMN game_type TEXT NOT NULL DEFAULT 'RPS+'")
        if "fool_rate" not in impostor_columns:
            conn.execute("ALTER TABLE impostor_results ADD COLUMN fool_rate REAL")
        if "explanation_sample" not in impostor_columns:
            conn.execute("ALTER TABLE impostor_results ADD COLUMN explanation_sample TEXT")
        if "embedding_json" not in impostor_columns:
            conn.execute("ALTER TABLE impostor_results ADD COLUMN embedding_json TEXT")
        detection_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(detection_sessions)").fetchall()
        }
        if "source_player_id" not in detection_columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN source_player_id TEXT")
        if "study_block_id" not in detection_columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN study_block_id TEXT")
        if "blind_label" not in detection_columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN blind_label TEXT")
        if "surprisal_summary_json" not in detection_columns:
            conn.execute("ALTER TABLE detection_sessions ADD COLUMN surprisal_summary_json TEXT")
        conn.commit()
    finally:
        conn.close()


def init_extended_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        for ddl in EXTENDED_TABLES:
            conn.execute(ddl)
        for ddl in EXTENDED_INDEXES:
            conn.execute(ddl)
        conn.commit()
    finally:
        conn.close()
