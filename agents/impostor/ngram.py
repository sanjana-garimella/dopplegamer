"""N-gram frequency Impostor — predicts next move from last N moves of game history."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from agents.base import Agent
from environments.rps_plus import N_MOVES


class NGramImpostor(Agent):
    """Personalized opponent model using N-gram transition frequencies.

    Learns which moves a specific player tends to play after observing their
    last N-1 moves, then reproduces that conditional distribution at play time.
    """

    name = "ngram"

    def __init__(self, n: int = 2, seed: int | None = None) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self._history: list[int] = []
        self.player_embedding: np.ndarray | None = None
        self.opponent_profile: dict[str, Any] | None = None
        # context tuple -> Counter of next moves
        self._transitions: dict[tuple, Counter] = defaultdict(Counter)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._history.clear()

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        """Record agent_move into the N-gram model and update transition counts."""
        m = int(agent_move)
        if len(self._history) >= self.n - 1 and self.n > 1:
            context = tuple(self._history[-(self.n - 1):])
            self._transitions[context][m] += 1
        elif self.n == 1:
            self._transitions[()][m] += 1
        self._history.append(m)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = set(info.get("legal_moves") or list(range(N_MOVES)))

        # Build context from recent history
        if self.n > 1 and len(self._history) >= self.n - 1:
            context = tuple(self._history[-(self.n - 1):])
        else:
            context = ()

        if context in self._transitions:
            counts = {m: c for m, c in self._transitions[context].items() if m in legal}
            if counts:
                total = sum(counts.values())
                moves = sorted(counts)
                probs = np.array([counts[m] / total for m in moves])
                return int(self.rng.choice(moves, p=probs))

        # Backoff to unigram if no context match
        unigram = self._transitions.get((), Counter())
        if unigram:
            legal_counts = {m: c for m, c in unigram.items() if m in legal}
            if legal_counts:
                total = sum(legal_counts.values())
                moves = sorted(legal_counts)
                probs = np.array([legal_counts[m] / total for m in moves])
                return int(self.rng.choice(moves, p=probs))

        return int(self.rng.choice(sorted(legal)))

    def train(self, move_sequence: list[int]) -> None:
        """Batch-learn from a recorded move sequence (offline training).

        Updates transition counts only.  Does NOT touch ``self._history``
        so that inference-time context (populated by ``observe()``) is not
        corrupted when training and playing are interleaved.
        """
        for i, m in enumerate(move_sequence):
            if self.n > 1 and i >= self.n - 1:
                context = tuple(move_sequence[i - (self.n - 1):i])
                self._transitions[context][m] += 1
            elif self.n == 1:
                self._transitions[()][m] += 1

    def online_update(self, move_sequence: list[int]) -> None:
        self.train(move_sequence)

    def set_player_embedding(self, embedding: np.ndarray | list[float] | None) -> None:
        if embedding is None:
            self.player_embedding = None
            return
        self.player_embedding = np.array(embedding, dtype=np.float64)

    def set_opponent_profile(self, profile: dict[str, Any] | None) -> None:
        self.opponent_profile = profile

    def predict_proba(
        self,
        history: list[int] | None = None,
        legal: list[int] | None = None,
    ) -> np.ndarray:
        """Return a probability distribution over all N_MOVES actions.

        Illegal moves receive probability 0. The result sums to 1 over legal moves,
        or is uniform over legal moves when no training context matches.
        """
        legal_set = set(legal) if legal is not None else set(range(N_MOVES))
        ctx_history = history if history is not None else self._history
        probs = np.zeros(N_MOVES, dtype=np.float64)

        if self.n > 1 and len(ctx_history) >= self.n - 1:
            context = tuple(ctx_history[-(self.n - 1):])
        else:
            context = ()

        if context in self._transitions:
            counts = {m: c for m, c in self._transitions[context].items() if m in legal_set}
            if counts:
                total = sum(counts.values())
                for m, c in counts.items():
                    probs[m] = c / total
                return probs

        unigram = self._transitions.get((), Counter())
        if unigram:
            legal_counts = {m: c for m, c in unigram.items() if m in legal_set}
            if legal_counts:
                total = sum(legal_counts.values())
                for m, c in legal_counts.items():
                    probs[m] = c / total
                return probs

        # Uniform over legal moves (cold start)
        for m in legal_set:
            probs[m] = 1.0 / len(legal_set)
        return probs

    def predict(self, history: list[int] | None = None, legal: list[int] | None = None) -> int:
        """Sample next move from the predicted distribution (does not modify self._history)."""
        probs = self.predict_proba(history, legal)
        legal_set = set(legal) if legal is not None else set(range(N_MOVES))
        candidates = [m for m in range(N_MOVES) if m in legal_set]
        candidate_probs = np.array([probs[m] for m in candidates])
        if candidate_probs.sum() < 1e-9:
            return int(self.rng.choice(candidates))
        candidate_probs /= candidate_probs.sum()
        return int(self.rng.choice(candidates, p=candidate_probs))

    def select_action(
        self,
        context: list[int] | None = None,
        legal: list[int] | None = None,
    ) -> int:
        """Sample an action from the predicted distribution. Alias for predict()."""
        return self.predict(history=context, legal=legal)

    def fidelity_score(self, reference_sequences: list[list[int]]) -> float:
        """Return 1 - TVD between the model's predicted distribution and reference sequences.

        Builds context correctly by advancing the observed history one move at a time,
        using the actual reference move (not the prediction) as the next context token.
        """
        all_ref = [m for seq in reference_sequences for m in seq]
        if not all_ref:
            return 0.0
        ref_dist = np.zeros(N_MOVES, dtype=np.float64)
        for m in all_ref:
            if 0 <= m < N_MOVES:
                ref_dist[m] += 1
        if ref_dist.sum() == 0:
            return 0.0
        ref_dist /= ref_dist.sum()

        pred_moves: list[int] = []
        for seq in reference_sequences:
            h: list[int] = []
            for m in seq:                                # advance context with actual move
                pred_moves.append(self.predict(h, list(range(N_MOVES))))
                h.append(m)
        if not pred_moves:
            return 0.0
        pred_dist = np.zeros(N_MOVES, dtype=np.float64)
        for m in pred_moves:
            if 0 <= m < N_MOVES:
                pred_dist[m] += 1
        if pred_dist.sum() > 0:
            pred_dist /= pred_dist.sum()
        tvd = float(0.5 * np.abs(ref_dist - pred_dist).sum())
        return max(0.0, 1.0 - tvd)

    def explain_prediction(
        self,
        history: list[int] | None = None,
        legal: list[int] | None = None,
    ) -> str:
        ctx_history = history if history is not None else self._history
        if self.n > 1 and len(ctx_history) >= self.n - 1:
            context = tuple(ctx_history[-(self.n - 1):])
        else:
            context = ()
        probs = self.predict_proba(history=ctx_history, legal=legal)
        top_move = int(np.argmax(probs)) if probs.sum() else -1
        if context in self._transitions and self._transitions[context]:
            count = int(sum(self._transitions[context].values()))
            opponent_note = ""
            if self.opponent_profile and self.opponent_profile.get("opponent_name"):
                opponent_note = f" Against {self.opponent_profile['opponent_name']}, that opponent profile appears {self.opponent_profile.get('games', 0)} times."
            return (
                f"Predicted move {top_move} from context {context} "
                f"seen {count} times in your history.{opponent_note}"
            )
        return f"Predicted move {top_move} from fallback unigram frequencies."

    def uncertainty(self, history: list[int] | None = None, legal: list[int] | None = None) -> dict[str, float]:
        probs = self.predict_proba(history=history, legal=legal)
        support = probs[probs > 0]
        if support.size == 0:
            return {"confidence": 0.0, "entropy": 0.0}
        entropy = float(-(support * np.log2(support)).sum())
        max_entropy = float(np.log2(max(len(support), 1)))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return {"confidence": float(np.clip(confidence, 0.0, 1.0)), "entropy": entropy}

    def latency_estimate_ms(self) -> float:
        return float(max(0.05, 0.08 * self.n + 0.002 * len(self._transitions)))

    def surprisal(
        self,
        realized_move: int,
        history: list[int] | None = None,
        legal: list[int] | None = None,
    ) -> dict[str, float | int]:
        probs = self.predict_proba(history=history, legal=legal)
        move = int(realized_move)
        prob = float(probs[move]) if 0 <= move < len(probs) else 0.0
        stats = self.uncertainty(history=history, legal=legal)
        return {
            "predicted_prob": prob,
            "surprisal": float(-np.log2(max(prob, 1e-9))),
            "confidence": float(stats.get("confidence", 0.0)),
            "entropy": float(stats.get("entropy", 0.0)),
            "expected_action": int(np.argmax(probs)) if probs.size else -1,
        }
