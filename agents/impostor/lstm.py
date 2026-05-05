"""LSTM Impostor Agent — learns sequential behavioral patterns from game history.

PyTorch is an optional dependency. Without it, the agent falls back to a
uniform random policy so the module always imports cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agents.base import Agent
from environments.rps_plus import N_MOVES

# Sequence window fed into the LSTM each step.
_SEQ_LEN = 10
# Special token for "no history yet"
_PAD = N_MOVES


def _build_model(hidden_size: int, num_layers: int):
    """Return an LSTM model or None if PyTorch is unavailable."""
    try:
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                vocab = N_MOVES + 1  # moves + padding token
                self.embed = nn.Embedding(vocab, hidden_size, padding_idx=_PAD)
                self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, N_MOVES)

            def forward(self, x):
                x = self.embed(x)
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        return _Model()
    except ImportError:
        return None


class LSTMImpostor(Agent):
    """LSTM-based behavioral clone trained on a specific player's move sequences.

    The model takes the last _SEQ_LEN moves as input and predicts the next move,
    reproducing the player's sequential decision patterns.
    """

    name = "lstm"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        checkpoint: str | Path | None = None,
        seed: int | None = None,
        quantization_mode: str = "fp32",
        variant_name: str = "lstm",
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.quantization_mode = quantization_mode
        self.variant_name = variant_name
        self.rng = np.random.default_rng(seed)
        self._history: list[int] = []
        self._model = None
        self.player_embedding: np.ndarray | None = None
        self.opponent_profile: dict[str, Any] | None = None

        if checkpoint is not None:
            self._load(Path(checkpoint))

    # ---------------------------------------------------------------- Agent API

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._history.clear()

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self._history.append(int(agent_move))

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = list(info.get("legal_moves") or range(N_MOVES))

        if self._model is not None and self._history:
            try:
                import torch
                window = self._history[-_SEQ_LEN:]
                padded = [_PAD] * (_SEQ_LEN - len(window)) + window
                x = torch.tensor([padded], dtype=torch.long)
                with torch.no_grad():
                    logits = self._model(x)[0]
                probs = torch.softmax(logits, dim=-1).numpy()
                masked = np.array([probs[m] if m in legal else 0.0 for m in range(N_MOVES)])
                if masked.sum() > 1e-9:
                    masked /= masked.sum()
                    # Sample from the distribution to reproduce the player's stochastic
                    # behaviour — argmax would always pick the same move and fail the
                    # Turing-test criterion the Impostor is designed to pass.
                    return int(self.rng.choice(len(masked), p=masked))
            except Exception:
                pass

        return int(self.rng.choice(legal))

    # ---------------------------------------------------------------- Training

    def train(
        self,
        sequences: list[list[int]],
        epochs: int = 20,
        lr: float = 1e-3,
    ) -> dict:
        """Train on a list of per-game move sequences. Returns training stats."""
        try:
            import torch
            import torch.nn as nn
            from torch.optim import Adam
        except ImportError:
            return {"error": "PyTorch not installed — pip install torch"}

        self._model = _build_model(self.hidden_size, self.num_layers)
        if self._model is None:
            return {"error": "model build failed"}

        optimizer = Adam(self._model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self._model.train()

        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for seq in sequences:
                if len(seq) < 2:
                    continue
                for i in range(1, len(seq)):
                    window = seq[max(0, i - _SEQ_LEN):i]
                    padded = [_PAD] * (_SEQ_LEN - len(window)) + window
                    x = torch.tensor([padded], dtype=torch.long)
                    y = torch.tensor([seq[i]], dtype=torch.long)
                    logits = self._model(x)
                    loss = criterion(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))

        self._model.eval()
        return {"epochs": epochs, "final_loss": losses[-1] if losses else 0.0, "loss_curve": losses}

    def online_update(self, sequences: list[list[int]], epochs: int = 3) -> dict:
        return self.train(sequences, epochs=epochs)

    def set_player_embedding(self, embedding: np.ndarray | list[float] | None) -> None:
        if embedding is None:
            self.player_embedding = None
            return
        self.player_embedding = np.array(embedding, dtype=np.float64)

    def set_opponent_profile(self, profile: dict[str, Any] | None) -> None:
        self.opponent_profile = profile

    # ---------------------------------------------------------------- Prediction

    def predict_proba(
        self,
        history: list[int] | None = None,
        legal: list[int] | None = None,
    ) -> np.ndarray:
        """Return a probability distribution over all N_MOVES actions.

        Illegal moves receive probability 0. The result sums to 1 over legal moves,
        or is uniform over legal moves when no model is loaded or history is empty.
        """
        legal_list = legal if legal is not None else list(range(N_MOVES))
        hist = history if history is not None else self._history
        probs = np.zeros(N_MOVES, dtype=np.float64)

        if self._model is not None and hist:
            try:
                import torch
                window = hist[-_SEQ_LEN:]
                padded = [_PAD] * (_SEQ_LEN - len(window)) + window
                x = torch.tensor([padded], dtype=torch.long)
                with torch.no_grad():
                    logits = self._model(x)[0]
                raw = torch.softmax(logits, dim=-1).numpy().astype(np.float64)
                for m in range(N_MOVES):
                    probs[m] = raw[m] if m in legal_list else 0.0
                if probs.sum() > 1e-9:
                    probs /= probs.sum()
                    return probs
            except Exception:
                pass

        # Uniform fallback over legal moves
        for m in legal_list:
            probs[m] = 1.0 / max(len(legal_list), 1)
        return probs

    def predict(self, history: list[int] | None = None, legal: list[int] | None = None) -> int:
        """Sample next move from the predicted distribution (does not modify self._history)."""
        legal_list = legal if legal is not None else list(range(N_MOVES))
        probs = self.predict_proba(history, legal)
        candidates = [m for m in range(N_MOVES) if m in legal_list]
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
        """Return 1 - TVD between model's predicted distribution and reference sequences.

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
            for m in seq:                               # advance context with actual move
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
        hist = history if history is not None else self._history
        probs = self.predict_proba(history=hist, legal=legal)
        top_move = int(np.argmax(probs)) if probs.sum() else -1
        if not hist:
            return f"Predicted move {top_move} from cold-start distribution."
        opponent_note = ""
        if self.opponent_profile and self.opponent_profile.get("opponent_name"):
            opponent_note = f" Opponent context: {self.opponent_profile['opponent_name']}."
        return (
            f"Predicted move {top_move} from the last {min(len(hist), _SEQ_LEN)} "
            f"moves in your sequence window.{opponent_note}"
        )

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
        quant_factor = {"fp32": 1.0, "fp16": 0.72, "int8": 0.55, "int4": 0.42}.get(self.quantization_mode, 1.0)
        return float(max(0.2, (self.hidden_size * self.num_layers) / 18.0 * quant_factor))

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

    # ---------------------------------------------------------------- I/O

    def save(self, path: str | Path) -> None:
        if self._model is not None:
            try:
                import torch
                torch.save(self._model.state_dict(), str(path))
            except Exception:
                pass

    def _load(self, path: Path) -> None:
        try:
            import torch
            self._model = _build_model(self.hidden_size, self.num_layers)
            if self._model is not None and path.exists():
                self._model.load_state_dict(torch.load(str(path), map_location="cpu"))
                self._model.eval()
        except Exception:
            self._model = None
