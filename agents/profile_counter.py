"""Profile-trained counter agent.

This is a lightweight online-training opponent: it reads saved gameplay for the
active player, predicts likely player actions, and chooses legal moves that are
more likely to beat that player. It is intentionally fast enough to refresh at
game start and after completed games.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from agents.base import Agent
from data.schemas import connect, init_db
from environments.rps_plus import BEST_COUNTER, Move, N_MOVES


class ProfileCounterAgent(Agent):
    name = "profile_counter"

    def __init__(
        self,
        player_id: str | None = None,
        db_path: str | Path = "data/game_data.db",
        seed: int | None = None,
    ) -> None:
        self.player_id = player_id
        self.db_path = Path(db_path)
        self.rng = np.random.default_rng(seed)
        self.player_counts: Counter[int] = Counter()
        self.context_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.successful_opponent_moves: Counter[int] = Counter()
        self._last_player_move: int | None = None
        self._current_player_move: int | None = None
        self.train_from_db()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._last_player_move = None
        self._current_player_move = None
        self.train_from_db()

    def set_current_player_move(self, move: int) -> None:
        self._current_player_move = int(move)

    def train_from_db(self) -> None:
        self.player_counts.clear()
        self.context_counts.clear()
        self.successful_opponent_moves.clear()
        if not self.player_id:
            return

        init_db(self.db_path)
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                """SELECT r.agent_move, r.opponent_move, r.outcome
                   FROM rounds r
                   JOIN games g USING (game_id)
                   WHERE g.agent_name = ?
                   ORDER BY g.started_at, r.turn""",
                (self.player_id,),
            ).fetchall()
        finally:
            conn.close()

        previous: int | None = None
        for player_move, opponent_move, outcome in rows:
            player_move = int(player_move)
            opponent_move = int(opponent_move)
            outcome = int(outcome)
            self.player_counts[player_move] += 1
            if previous is not None:
                self.context_counts[previous][player_move] += 1
            if outcome < 0:
                self.successful_opponent_moves[opponent_move] += 1
            previous = player_move

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        player_move = int(opponent_move)
        self.player_counts[player_move] += 1
        if self._last_player_move is not None:
            self.context_counts[self._last_player_move][player_move] += 1
        if int(outcome) > 0:
            self.successful_opponent_moves[int(agent_move)] += 1
        self._last_player_move = player_move
        self._current_player_move = None

    def _predict_player_move(self) -> int | None:
        if self._current_player_move is not None:
            return self._current_player_move
        if self._last_player_move is not None:
            contextual = self.context_counts.get(self._last_player_move)
            if contextual:
                return int(contextual.most_common(1)[0][0])
        if self.player_counts:
            return int(self.player_counts.most_common(1)[0][0])
        return None

    def _rps_counter(self, predicted: int, legal: set[int]) -> int | None:
        if predicted not in range(N_MOVES):
            return None
        try:
            counter = int(BEST_COUNTER[Move(predicted)])
        except ValueError:
            return None
        if counter in legal:
            return counter
        if int(Move.POWER) in legal and predicted in {
            int(Move.ROCK),
            int(Move.PAPER),
            int(Move.SCISSORS),
            int(Move.LIZARD),
        }:
            return int(Move.POWER)
        return None

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = set(info.get("legal_moves") or [])
        if not legal:
            return 0

        predicted = self._predict_player_move()
        if predicted is not None:
            counter = self._rps_counter(predicted, legal)
            if counter is not None:
                return counter

        successful = {
            move: count
            for move, count in self.successful_opponent_moves.items()
            if move in legal
        }
        if successful:
            return int(max(successful, key=successful.get))

        return int(self.rng.choice(sorted(legal)))
