"""Per-player Impostor training pipeline.

Given a player_id, loads their recorded game data from SQLite and trains
all Impostor types (NGram, LSTM). Results are returned and optionally
persisted so the evaluation harness can reuse trained models.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import numpy as np

from agents.impostor.lstm import LSTMImpostor
from agents.impostor.ngram import NGramImpostor
from data.features import (
    CONTROLLED_REARING_SLICES,
    apply_session_slice,
    cross_game_embedding,
    opponent_style_summary,
    recency_weights,
    session_ordered_sequences,
)
from data.schemas import connect


@dataclass
class TrainingResult:
    player_id: str
    impostor_type: str
    n_sequences: int
    n_moves: int
    stats: dict


class MixtureImpostor:
    """Hybrid clone that blends local habit memory and sequence priors."""

    name = "mixture"

    def __init__(self, ngram: NGramImpostor, lstm: LSTMImpostor, heuristic_prior: list[float] | None = None) -> None:
        self.ngram = ngram
        self.lstm = lstm
        self.heuristic_prior = heuristic_prior
        self.player_embedding = getattr(ngram, "player_embedding", None)
        self.opponent_profile = getattr(ngram, "opponent_profile", None)

    def reset(self, seed: int | None = None) -> None:
        self.ngram.reset(seed=seed)
        self.lstm.reset(seed=seed)

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self.ngram.observe(agent_move, opponent_move, outcome)
        self.lstm.observe(agent_move, opponent_move, outcome)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = info.get("legal_moves") or list(range(6))
        probs = self.predict_proba(history=None, legal=legal)
        if probs.sum() <= 0:
            return int(legal[0])
        return int(np.argmax(probs))

    def predict_proba(self, history: list[int] | None = None, legal: list[int] | None = None) -> np.ndarray:
        probs = (0.55 * self.ngram.predict_proba(history=history, legal=legal)) + (
            0.35 * self.lstm.predict_proba(history=history, legal=legal)
        )
        if self.heuristic_prior is not None:
            prior = np.array(self.heuristic_prior, dtype=np.float64)
            if prior.sum() > 0:
                probs += 0.10 * (prior / prior.sum())
        if probs.sum() > 0:
            probs /= probs.sum()
        return probs

    def predict(self, history: list[int] | None = None, legal: list[int] | None = None) -> int:
        probs = self.predict_proba(history=history, legal=legal)
        if probs.sum() <= 0:
            legal_moves = legal or list(range(len(probs)))
            return int(legal_moves[0])
        return int(np.argmax(probs))

    def select_action(self, context: list[int] | None = None, legal: list[int] | None = None) -> int:
        return self.predict(history=context, legal=legal)

    def uncertainty(self, history: list[int] | None = None, legal: list[int] | None = None) -> dict[str, float]:
        probs = self.predict_proba(history=history, legal=legal)
        support = probs[probs > 0]
        if support.size == 0:
            return {"confidence": 0.0, "entropy": 0.0}
        entropy = float(-(support * np.log2(support)).sum())
        max_entropy = float(np.log2(max(len(support), 1)))
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return {"confidence": float(confidence), "entropy": entropy}

    def latency_estimate_ms(self) -> float:
        ngram_ms = getattr(self.ngram, "latency_estimate_ms", lambda: 0.1)()
        lstm_ms = getattr(self.lstm, "latency_estimate_ms", lambda: 1.0)()
        return float(0.55 * ngram_ms + 0.35 * lstm_ms + 0.10)

    def explain_prediction(self, history: list[int] | None = None, legal: list[int] | None = None) -> str:
        ngram_note = self.ngram.explain_prediction(history=history, legal=legal)
        lstm_note = self.lstm.explain_prediction(history=history, legal=legal)
        return f"Mixture clone blended short-context habit memory with sequence priors. {ngram_note} {lstm_note}"

    def surprisal(self, realized_move: int, history: list[int] | None = None, legal: list[int] | None = None) -> dict[str, float | int]:
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


class ImpostorTrainer:
    """Trains personalized Impostor agents from a player's SQLite game history."""

    def __init__(
        self,
        db_path: str | Path = "data/game_data.db",
        checkpoint_dir: str | Path = "checkpoints/impostor",
    ) -> None:
        self.db_path = Path(db_path)
        self.checkpoint_dir = Path(checkpoint_dir)

    # ---------------------------------------------------------------- data loading

    def load_player_sequences(self, player_id: str) -> list[list[int]]:
        """Return one list-of-moves per game for player_id."""
        return self.load_player_sequences_for_game(player_id, "RPS+")

    def load_player_sequences_for_game(self, player_id: str, game_type: str) -> list[list[int]]:
        """Return one list-of-moves per game for player_id and game_type."""
        conn = connect(self.db_path)
        try:
            game_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT game_id FROM games WHERE agent_name = ? AND game_type = ? ORDER BY started_at",
                    (player_id, game_type),
                ).fetchall()
            ]
            sequences = []
            for gid in game_ids:
                moves = [
                    r[0]
                    for r in conn.execute(
                        "SELECT agent_move FROM rounds WHERE game_id = ? ORDER BY turn",
                        (gid,),
                    ).fetchall()
                ]
                if moves:
                    sequences.append(moves)
        finally:
            conn.close()
        return sequences

    def load_player_sequences_with_context(self, player_id: str, game_type: str = "RPS+") -> list[dict[str, Any]]:
        return session_ordered_sequences(self.db_path, player_id, game_type=game_type)

    def load_filtered_sequences(
        self,
        player_id: str,
        *,
        game_type: str = "RPS+",
        slice_name: str | None = None,
    ) -> list[dict[str, Any]]:
        contexts = self.load_player_sequences_with_context(player_id, game_type=game_type)
        if not slice_name:
            return contexts
        return apply_session_slice(contexts, slice_name, dominant_game_type=game_type)

    def _contexts_to_sequences(
        self,
        contexts: list[dict[str, Any]],
        *,
        suppress_recharge: bool = False,
    ) -> list[list[int]]:
        sequences: list[list[int]] = []
        for ctx in contexts:
            seq = [int(m) for m in ctx.get("moves", [])]
            if suppress_recharge:
                seq = [0 if move == 5 else move for move in seq]
            if seq:
                sequences.append(seq)
        return sequences

    def _build_opponent_profile(
        self,
        player_id: str,
        *,
        game_type: str = "RPS+",
        include_opponent_conditioning: bool = True,
    ) -> dict[str, Any] | None:
        if not include_opponent_conditioning:
            return None
        opp_summary = opponent_style_summary(self.db_path, player_id, game_type=game_type)
        if opp_summary.empty:
            return None
        return opp_summary.iloc[0].to_dict()

    def load_player_outcomes(self, player_id: str) -> list[list[int]]:
        """Return one list-of-outcomes per game for player_id."""
        conn = connect(self.db_path)
        try:
            game_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT game_id FROM games WHERE agent_name = ? AND game_type = 'RPS+' ORDER BY started_at",
                    (player_id,),
                ).fetchall()
            ]
            outcome_seqs = []
            for gid in game_ids:
                outcomes = [
                    r[0]
                    for r in conn.execute(
                        "SELECT outcome FROM rounds WHERE game_id = ? ORDER BY turn",
                        (gid,),
                    ).fetchall()
                ]
                if outcomes:
                    outcome_seqs.append(outcomes)
        finally:
            conn.close()
        return outcome_seqs

    # ---------------------------------------------------------------- training

    def train_ngram(
        self,
        player_id: str,
        n: int = 2,
        *,
        game_type: str = "RPS+",
        slice_name: str | None = None,
        suppress_recharge: bool = False,
        include_opponent_conditioning: bool = True,
    ) -> tuple[NGramImpostor, TrainingResult]:
        contexts = self.load_filtered_sequences(player_id, game_type=game_type, slice_name=slice_name)
        sequences = self._contexts_to_sequences(contexts, suppress_recharge=suppress_recharge)
        agent = NGramImpostor(n=n)
        for seq in sequences:
            agent.train(seq)
        agent.set_player_embedding(cross_game_embedding(self.db_path, player_id))
        agent.set_opponent_profile(
            self._build_opponent_profile(
                player_id,
                game_type=game_type,
                include_opponent_conditioning=include_opponent_conditioning,
            )
        )
        result = TrainingResult(
            player_id=player_id,
            impostor_type=f"ngram_{n}" if not slice_name else f"ngram_{n}::{slice_name}",
            n_sequences=len(sequences),
            n_moves=sum(len(s) for s in sequences),
            stats={
                "n": n,
                "n_contexts": len(agent._transitions),
                "slice_name": slice_name,
                "game_type": game_type,
                "suppress_recharge": suppress_recharge,
                "opponent_conditioning": include_opponent_conditioning,
            },
        )
        return agent, result

    def train_lstm(
        self,
        player_id: str,
        epochs: int = 20,
        save: bool = True,
        *,
        game_type: str = "RPS+",
        slice_name: str | None = None,
        suppress_recharge: bool = False,
        include_opponent_conditioning: bool = True,
        hidden_size: int = 64,
        num_layers: int = 2,
        variant_name: str = "lstm",
        quantization_mode: str = "fp32",
    ) -> tuple[LSTMImpostor, TrainingResult]:
        contexts = self.load_filtered_sequences(player_id, game_type=game_type, slice_name=slice_name)
        sequences = self._contexts_to_sequences(contexts, suppress_recharge=suppress_recharge)
        agent = LSTMImpostor(hidden_size=hidden_size, num_layers=num_layers)
        agent.quantization_mode = quantization_mode
        agent.variant_name = variant_name
        agent.set_player_embedding(cross_game_embedding(self.db_path, player_id))
        agent.set_opponent_profile(
            self._build_opponent_profile(
                player_id,
                game_type=game_type,
                include_opponent_conditioning=include_opponent_conditioning,
            )
        )
        stats = agent.train(sequences, epochs=epochs)
        if save and "error" not in stats:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            agent.save(self.checkpoint_dir / f"{player_id}_{variant_name}.pt")
        result = TrainingResult(
            player_id=player_id,
            impostor_type=variant_name if not slice_name else f"{variant_name}::{slice_name}",
            n_sequences=len(sequences),
            n_moves=sum(len(s) for s in sequences),
            stats={
                **stats,
                "slice_name": slice_name,
                "game_type": game_type,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "quantization_mode": quantization_mode,
                "suppress_recharge": suppress_recharge,
                "opponent_conditioning": include_opponent_conditioning,
            },
        )
        return agent, result

    def persist_training_result(
        self,
        result: TrainingResult,
        fidelity_score: float | None = None,
        kl_divergence: float | None = None,
        tvd: float | None = None,
        run_id: str | None = None,
    ) -> str:
        """Write a TrainingResult to the impostor_results table. Returns run_id."""
        rid = run_id or uuid.uuid4().hex[:12]
        trained_at = datetime.now(timezone.utc).isoformat()
        embedding = cross_game_embedding(self.db_path, result.player_id).tolist()
        conn = connect(self.db_path)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO impostor_results
                   (run_id, player_id, impostor_type, game_type, n_training_rounds,
                    fidelity_score, kl_divergence, tvd, fool_rate, explanation_sample,
                    embedding_json, trained_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rid,
                    result.player_id,
                    result.impostor_type,
                    "RPS+",
                    result.n_moves,
                    fidelity_score,
                    kl_divergence,
                    tvd,
                    None,
                    result.stats.get("explanation_sample"),
                    json.dumps(embedding),
                    trained_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return rid

    def online_update(self, player_id: str, *, game_type: str = "RPS+") -> dict[str, Any]:
        sequences = self.load_player_sequences_for_game(player_id, game_type)
        if not sequences:
            return {"updated": False, "reason": "no sequences"}
        latest = sequences[-1]
        ngram, _ = self.train_ngram(player_id)
        ngram.online_update(latest)
        lstm, _ = self.train_lstm(player_id, epochs=5, save=False)
        lstm.online_update([latest], epochs=3)
        return {
            "updated": True,
            "game_type": game_type,
            "latest_sequence_len": len(latest),
        }

    def recency_weighted_online_update(self, player_id: str, *, game_type: str = "RPS+", decay: float = 0.85) -> dict[str, Any]:
        contexts = self.load_player_sequences_with_context(player_id, game_type=game_type)
        if not contexts:
            return {"updated": False, "reason": "no sequences"}
        weights = recency_weights(len(contexts), decay=decay)
        ngram = NGramImpostor(n=2)
        weighted_sequences: list[list[int]] = []
        for ctx, weight in zip(contexts, weights):
            repeats = max(1, int(np.ceil(weight * len(contexts))))
            for _ in range(repeats):
                seq = list(ctx["moves"])
                ngram.train(seq)
                weighted_sequences.append(seq)
        lstm = LSTMImpostor()
        lstm.train(weighted_sequences, epochs=5)
        return {
            "updated": True,
            "game_type": game_type,
            "weighted_sequences": len(weighted_sequences),
            "decay": decay,
        }

    def train_lstm_variants(self, player_id: str, *, game_type: str = "RPS+") -> dict[str, tuple[LSTMImpostor, TrainingResult]]:
        variants = {
            "lstm": dict(hidden_size=64, num_layers=2, quantization_mode="fp32"),
            "lstm_tiny": dict(hidden_size=32, num_layers=1, quantization_mode="fp16"),
            "lstm_int8": dict(hidden_size=48, num_layers=1, quantization_mode="int8"),
        }
        outputs = {}
        for name, cfg in variants.items():
            outputs[name] = self.train_lstm(
                player_id,
                epochs=5,
                save=False,
                game_type=game_type,
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                variant_name=name,
                quantization_mode=cfg["quantization_mode"],
            )
        return outputs

    def train_mixture(
        self,
        player_id: str,
        *,
        game_type: str = "RPS+",
        slice_name: str | None = None,
        suppress_recharge: bool = False,
        include_opponent_conditioning: bool = True,
    ) -> tuple[MixtureImpostor, TrainingResult]:
        ngram, ngram_result = self.train_ngram(
            player_id,
            game_type=game_type,
            slice_name=slice_name,
            suppress_recharge=suppress_recharge,
            include_opponent_conditioning=include_opponent_conditioning,
        )
        lstm, lstm_result = self.train_lstm(
            player_id,
            epochs=5,
            save=False,
            game_type=game_type,
            slice_name=slice_name,
            suppress_recharge=suppress_recharge,
            include_opponent_conditioning=include_opponent_conditioning,
        )
        flat_moves = [m for seq in self._contexts_to_sequences(self.load_filtered_sequences(player_id, game_type=game_type, slice_name=slice_name), suppress_recharge=suppress_recharge) for m in seq]
        prior = np.zeros(6, dtype=np.float64)
        for move in flat_moves:
            if 0 <= int(move) < 6:
                prior[int(move)] += 1
        mixture = MixtureImpostor(ngram, lstm, prior.tolist())
        result = TrainingResult(
            player_id=player_id,
            impostor_type="mixture" if not slice_name else f"mixture::{slice_name}",
            n_sequences=max(ngram_result.n_sequences, lstm_result.n_sequences),
            n_moves=max(ngram_result.n_moves, lstm_result.n_moves),
            stats={
                "components": ["ngram", "lstm", "heuristic_prior"],
                "slice_name": slice_name,
                "game_type": game_type,
                "suppress_recharge": suppress_recharge,
                "opponent_conditioning": include_opponent_conditioning,
            },
        )
        return mixture, result

    def train_with_interventions(
        self,
        player_id: str,
        *,
        game_type: str = "RPS+",
        suppress_recency: bool = False,
        suppress_recharge: bool = False,
        include_opponent_conditioning: bool = True,
    ) -> dict[str, tuple[Any, TrainingResult]]:
        slice_name = "recent_sessions_v1" if suppress_recency else None
        return {
            "ngram": self.train_ngram(
                player_id,
                game_type=game_type,
                slice_name=slice_name,
                suppress_recharge=suppress_recharge,
                include_opponent_conditioning=include_opponent_conditioning,
            ),
            "lstm": self.train_lstm(
                player_id,
                epochs=5,
                save=False,
                game_type=game_type,
                slice_name=slice_name,
                suppress_recharge=suppress_recharge,
                include_opponent_conditioning=include_opponent_conditioning,
                variant_name="lstm_intervened",
            ),
            "mixture": self.train_mixture(
                player_id,
                game_type=game_type,
                slice_name=slice_name,
                suppress_recharge=suppress_recharge,
                include_opponent_conditioning=include_opponent_conditioning,
            ),
        }

    def train_all(self, player_id: str) -> dict[str, Any]:
        """Train all Impostor types and return agents + results keyed by type."""
        ngram_agent, ngram_result = self.train_ngram(player_id)
        lstm_agent, lstm_result = self.train_lstm(player_id)
        mixture_agent, mixture_result = self.train_mixture(player_id)
        return {
            "ngram": {"agent": ngram_agent, "result": ngram_result},
            "lstm":  {"agent": lstm_agent,  "result": lstm_result},
            "mixture": {"agent": mixture_agent, "result": mixture_result},
        }
