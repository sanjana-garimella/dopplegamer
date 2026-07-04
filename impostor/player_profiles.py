"""Player profile management for Impostor Effect.

Each human player gets a PlayerProfile that stores their behavioral signature —
a compact fingerprint of their play style used to measure how well an Impostor
reproduces them.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from data.schemas import connect, init_db, init_extended_db
from environments.rps_plus import BEST_COUNTER, N_MOVES, Move
from impostor.metrics import retraining_trigger


# ─────────────────────────────────────────────────────── behavioral signature

@dataclass
class BehavioralSignature:
    """Compact representation of a player's stylistic tendencies."""
    move_distribution: list[float]   # empirical P(move) for each of N_MOVES moves
    energy_aggression: float         # P(POWER) across all rounds
    recharge_rate: float             # P(RECHARGE) across all rounds
    counter_rate: float              # P(correct counter to prev opponent move)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BehavioralSignature":
        return cls(**d)

    def similarity(self, other: "BehavioralSignature") -> float:
        """Cosine similarity of the two move distributions."""
        a = np.array(self.move_distribution)
        b = np.array(other.move_distribution)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


# ─────────────────────────────────────────────────────────── player profile

@dataclass
class PlayerProfile:
    player_id: str
    display_name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    games_played: int = 0
    total_rounds: int = 0
    win_rate: float = 0.0
    behavioral_signature: BehavioralSignature | None = None

    def to_dict(self) -> dict:
        d = {
            "player_id": self.player_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "games_played": self.games_played,
            "total_rounds": self.total_rounds,
            "win_rate": self.win_rate,
            "behavioral_signature": (
                self.behavioral_signature.to_dict()
                if self.behavioral_signature else None
            ),
        }
        return d


# ─────────────────────────────────────────────────────── signature computation

_COUNTER: dict[int, int] = {int(k): int(v) for k, v in BEST_COUNTER.items()}


def compute_signature(
    move_sequences: list[list[int]],
    outcome_sequences: list[list[int]],
    opp_sequences: list[list[int]] | None = None,
) -> BehavioralSignature:
    """Derive a BehavioralSignature from raw game data."""
    all_moves = [m for seq in move_sequences for m in seq]
    total = max(len(all_moves), 1)

    dist = np.zeros(N_MOVES)
    for m in all_moves:
        dist[m] += 1
    if dist.sum() > 0:
        dist /= dist.sum()

    power = int(Move.POWER)
    recharge = int(Move.RECHARGE)
    energy_aggression = sum(1 for m in all_moves if m == power) / total
    recharge_rate = sum(1 for m in all_moves if m == recharge) / total

    counter_hits = counter_opps = 0
    if opp_sequences:
        for move_seq, opp_seq in zip(move_sequences, opp_sequences):
            for i in range(1, min(len(move_seq), len(opp_seq))):
                expected = _COUNTER.get(opp_seq[i - 1])
                if expected is not None:
                    counter_opps += 1
                    if move_seq[i] == expected:
                        counter_hits += 1
    counter_rate = counter_hits / max(counter_opps, 1)

    return BehavioralSignature(
        move_distribution=dist.tolist(),
        energy_aggression=energy_aggression,
        recharge_rate=recharge_rate,
        counter_rate=counter_rate,
    )


def _blend_signatures(
    previous: BehavioralSignature | None,
    previous_rounds: int,
    current: BehavioralSignature,
    current_rounds: int,
) -> BehavioralSignature:
    """Combine a saved profile fingerprint with a new batch by round count."""
    if previous is None or previous_rounds <= 0:
        return current
    if current_rounds <= 0:
        return previous

    total = previous_rounds + current_rounds
    prev_weight = previous_rounds / total
    cur_weight = current_rounds / total
    prev_dist = np.array(previous.move_distribution, dtype=np.float64)
    cur_dist = np.array(current.move_distribution, dtype=np.float64)
    return BehavioralSignature(
        move_distribution=(prev_dist * prev_weight + cur_dist * cur_weight).tolist(),
        energy_aggression=(previous.energy_aggression * prev_weight) + (current.energy_aggression * cur_weight),
        recharge_rate=(previous.recharge_rate * prev_weight) + (current.recharge_rate * cur_weight),
        counter_rate=(previous.counter_rate * prev_weight) + (current.counter_rate * cur_weight),
    )


# ─────────────────────────────────────────────────────── profile manager

class PlayerProfileManager:
    """CRUD interface for player profiles stored in SQLite."""

    def __init__(self, db_path: str | Path = "data/game_data.db") -> None:
        self.db_path = Path(db_path)
        # `player_profiles` lives in the base schema; a fresh DB file (e.g. a new
        # clone/instance) has no tables at all until this runs, so every method
        # below would hit `no such table: player_profiles`.
        init_db(self.db_path)

    def create(self, display_name: str, player_id: str | None = None) -> PlayerProfile:
        player_id = player_id or uuid.uuid4().hex[:8]
        profile = PlayerProfile(player_id=player_id, display_name=display_name)
        conn = connect(self.db_path)
        try:
            conn.execute(
                """INSERT OR IGNORE INTO player_profiles
                   (player_id, display_name, created_at, games_played, total_rounds,
                    win_rate, behavioral_signature_json)
                   VALUES (?, ?, ?, 0, 0, 0.0, NULL)""",
                (profile.player_id, profile.display_name, profile.created_at),
            )
            conn.commit()
        finally:
            conn.close()
        return profile

    def get(self, player_id: str) -> PlayerProfile | None:
        conn = connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT player_id, display_name, created_at, games_played, "
                "total_rounds, win_rate, behavioral_signature_json "
                "FROM player_profiles WHERE player_id = ?",
                (player_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_profile(row)

    def list_all(self) -> list[PlayerProfile]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT player_id, display_name, created_at, games_played, "
                "total_rounds, win_rate, behavioral_signature_json FROM player_profiles"
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_profile(r) for r in rows]

    def clear_clone_artifacts(self, player_id: str) -> dict[str, int]:
        """Delete clone outputs while keeping the underlying player profile and gameplay."""
        init_extended_db(self.db_path)
        tables = [
            "impostor_results",
            "detection_sessions",
            "clone_ab_runs",
            "counterfactual_replays",
            "shareable_reports",
            "blind_study_blocks",
            "dataset_slices",
        ]
        deleted: dict[str, int] = {}
        conn = connect(self.db_path)
        try:
            for table in tables:
                cursor = conn.execute(f"DELETE FROM {table} WHERE player_id = ?", (player_id,))
                deleted[table] = int(cursor.rowcount or 0)
            conn.commit()
        finally:
            conn.close()
        return deleted

    def clear_gameplay_data(self, player_id: str) -> dict[str, int]:
        """Delete saved gameplay plus all derived clone artifacts for one player."""
        init_extended_db(self.db_path)
        deleted = self.clear_clone_artifacts(player_id)
        conn = connect(self.db_path)
        try:
            game_rows = conn.execute(
                "SELECT game_id FROM games WHERE agent_name = ?",
                (player_id,),
            ).fetchall()
            game_ids = [str(row["game_id"]) for row in game_rows]
            rounds_deleted = 0
            if game_ids:
                placeholders = ",".join("?" for _ in game_ids)
                rounds_deleted = int(
                    conn.execute(
                        f"DELETE FROM rounds WHERE game_id IN ({placeholders})",
                        game_ids,
                    ).rowcount
                    or 0
                )
            games_deleted = int(
                conn.execute("DELETE FROM games WHERE agent_name = ?", (player_id,)).rowcount
                or 0
            )
            snapshots_deleted = int(
                conn.execute("DELETE FROM behavioral_snapshots WHERE player_id = ?", (player_id,)).rowcount
                or 0
            )
            ladder_deleted = int(
                conn.execute("DELETE FROM clone_ladder_runs WHERE player_id = ?", (player_id,)).rowcount
                or 0
            )
            conn.execute(
                """UPDATE player_profiles
                   SET games_played = 0,
                       total_rounds = 0,
                       win_rate = 0.0,
                       behavioral_signature_json = NULL
                   WHERE player_id = ?""",
                (player_id,),
            )
            conn.commit()
        finally:
            conn.close()
        deleted.update(
            {
                "games": games_deleted,
                "rounds": rounds_deleted,
                "behavioral_snapshots": snapshots_deleted,
                "clone_ladder_runs": ladder_deleted,
            }
        )
        return deleted

    def update_signature(
        self,
        player_id: str,
        move_seqs: list[list[int]],
        outcome_seqs: list[list[int]],
        opp_seqs: list[list[int]] | None = None,
        game_type: str = "RPS+",
    ) -> BehavioralSignature:
        current_sig = compute_signature(move_seqs, outcome_seqs, opp_seqs)
        batch_rounds = sum(len(s) for s in move_seqs)
        batch_wins = sum(1 for seq in outcome_seqs for o in seq if o > 0)
        existing = self.get(player_id)
        prev_games = existing.games_played if existing else 0
        prev_rounds = existing.total_rounds if existing else 0
        prev_wins = (existing.win_rate * prev_rounds) if existing else 0.0
        sig = _blend_signatures(
            existing.behavioral_signature if existing else None,
            prev_rounds,
            current_sig,
            batch_rounds,
        )
        total_rounds = prev_rounds + batch_rounds
        win_rate = (prev_wins + batch_wins) / max(total_rounds, 1)
        conn = connect(self.db_path)
        try:
            conn.execute(
                """UPDATE player_profiles
                   SET behavioral_signature_json = ?,
                       games_played = ?,
                       total_rounds = ?,
                       win_rate = ?
                   WHERE player_id = ?""",
                (
                    json.dumps(sig.to_dict()),
                    prev_games + len(move_seqs),
                    total_rounds,
                    win_rate,
                    player_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        self.snapshot_signature(player_id, sig, game_type=game_type)
        return sig

    def snapshot_signature(
        self,
        player_id: str,
        signature: BehavioralSignature,
        *,
        game_type: str = "RPS+",
        drift_score: float | None = None,
    ) -> str:
        profile = self.get(player_id)
        if profile is None:
            raise ValueError(f"Unknown player_id `{player_id}`")
        prev_signature = profile.behavioral_signature
        if drift_score is None:
            drift_score = 0.0 if prev_signature is None else max(
                0.0,
                1.0 - prev_signature.similarity(signature),
            )
        snapshot_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        init_extended_db(self.db_path)
        conn = connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO behavioral_snapshots
                   (snapshot_id, player_id, game_type, games_played, total_rounds,
                    win_rate, drift_score, signature_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    player_id,
                    game_type,
                    int(profile.games_played),
                    int(profile.total_rounds),
                    float(profile.win_rate),
                    float(drift_score),
                    json.dumps(signature.to_dict()),
                    created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return snapshot_id

    def load_behavioral_snapshots(self, player_id: str) -> list[dict[str, Any]]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                """SELECT snapshot_id, player_id, game_type, games_played, total_rounds,
                          win_rate, drift_score, signature_json, created_at
                   FROM behavioral_snapshots
                   WHERE player_id = ?
                   ORDER BY created_at""",
                (player_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "snapshot_id": row["snapshot_id"],
                "player_id": row["player_id"],
                "game_type": row["game_type"],
                "games_played": int(row["games_played"]),
                "total_rounds": int(row["total_rounds"]),
                "win_rate": float(row["win_rate"]),
                "drift_score": float(row["drift_score"]),
                "signature": json.loads(row["signature_json"]) if row["signature_json"] else None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def retraining_status(self, player_id: str, *, recent_window: int = 3, threshold: float = 0.18):
        return retraining_trigger(
            self.load_behavioral_snapshots(player_id),
            recent_window=recent_window,
            threshold=threshold,
        )

    def load_per_game_profiles(self, player_id: str) -> dict[str, dict[str, Any]]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT g.game_type,
                       COUNT(DISTINCT g.game_id) AS games_played,
                       COALESCE(SUM(g.n_turns), 0) AS total_rounds,
                       COALESCE(AVG(CASE WHEN g.agent_score > g.opponent_score THEN 1.0 ELSE 0.0 END), 0.0) AS win_rate
                FROM games g
                WHERE g.agent_name = ?
                GROUP BY g.game_type
                ORDER BY games_played DESC
                """,
                (player_id,),
            ).fetchall()
        finally:
            conn.close()
        return {
            row["game_type"] or "Unknown": {
                "games_played": int(row["games_played"]),
                "total_rounds": int(row["total_rounds"]),
                "win_rate": float(row["win_rate"]),
            }
            for row in rows
        }

    def record_detection_session(
        self,
        player_id: str,
        impostor_type: str,
        detected_as_human: bool,
        game_id: str | None = None,
        confidence: float | None = None,
        session_id: str | None = None,
        source_player_id: str | None = None,
        study_block_id: str | None = None,
        blind_label: str | None = None,
        surprisal_summary_json: str | None = None,
    ) -> str:
        """Record one Turing-test session to detection_sessions. Returns session_id."""
        sid = session_id or uuid.uuid4().hex
        recorded_at = datetime.now(timezone.utc).isoformat()
        conn = connect(self.db_path)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO detection_sessions
                   (session_id, player_id, impostor_type, source_player_id, study_block_id, blind_label, surprisal_summary_json, game_id,
                    detected_as_human, confidence, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sid,
                    player_id,
                    impostor_type,
                    source_player_id,
                    study_block_id,
                    blind_label,
                    surprisal_summary_json,
                    game_id,
                    int(detected_as_human),
                    confidence,
                    recorded_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return sid

    @staticmethod
    def _row_to_profile(row: tuple) -> PlayerProfile:
        pid, name, created, games, rounds, wr, sig_json = row
        sig = None
        if sig_json:
            try:
                sig = BehavioralSignature.from_dict(json.loads(sig_json))
            except Exception:
                pass
        return PlayerProfile(
            player_id=pid,
            display_name=name,
            created_at=created,
            games_played=games,
            total_rounds=rounds,
            win_rate=wr,
            behavioral_signature=sig,
        )
