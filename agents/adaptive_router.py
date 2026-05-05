"""Adaptive router agent that switches between expert policies.

The goal is to keep the live arena responsive while still letting a higher
level controller adapt over time. This router uses cheap local signals:

- saved player history (if available)
- recent win/loss trend
- current game type inferred from the move space

It can route among SFT, PPO/RL, BC+RL, and profile_counter. For non-RPS games
it falls back to the history-aware expert or a safe random-like legal choice.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

from agents.base import Agent
from agents.profile_counter import ProfileCounterAgent
from agents.rl import BCRLAgent, PPOAgent
from agents.sft.model import SFTAgent


class AdaptiveRouterAgent(Agent):
    name = "adaptive_router"

    def __init__(
        self,
        player_id: str | None = None,
        db_path: str | Path = "data/game_data.db",
        seed: int | None = None,
    ) -> None:
        self.player_id = player_id
        self.db_path = Path(db_path)
        self.rng = np.random.default_rng(seed)
        self.experts: dict[str, Agent] = {
            "sft": SFTAgent(seed=seed),
            "ppo": PPOAgent(seed=seed),
            "bc_rl": BCRLAgent(seed=seed),
            "profile_counter": ProfileCounterAgent(player_id=player_id, db_path=self.db_path, seed=seed),
        }
        self.selection_counts: Counter[str] = Counter()
        self.recent_outcomes: deque[int] = deque(maxlen=8)
        self.last_selected_expert: str | None = None
        self.last_route_reason: str = "Awaiting first move."
        self.persisted_history_examples = 0
        self._refresh_persisted_history_examples()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.recent_outcomes.clear()
        self.last_selected_expert = None
        self.last_route_reason = "Awaiting first move."
        self.selection_counts.clear()
        for expert in self.experts.values():
            expert.reset(seed=seed)
        self._refresh_persisted_history_examples()

    def set_current_player_move(self, move: int) -> None:
        setter = getattr(self.experts["profile_counter"], "set_current_player_move", None)
        if callable(setter):
            setter(move)

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self.recent_outcomes.append(int(outcome))
        for expert in self.experts.values():
            observer = getattr(expert, "observe", None)
            if callable(observer):
                try:
                    observer(agent_move, opponent_move, outcome)
                except (ValueError, TypeError):
                    continue

    def _has_profile_history(self) -> bool:
        return self.persisted_history_examples > 0

    def _refresh_persisted_history_examples(self) -> None:
        profile_agent = self.experts["profile_counter"]
        counts = getattr(profile_agent, "player_counts", None)
        self.persisted_history_examples = int(sum(counts.values())) if counts and hasattr(counts, "values") else 0

    def _recent_win_rate(self) -> float | None:
        if not self.recent_outcomes:
            return None
        wins = sum(1 for outcome in self.recent_outcomes if outcome > 0)
        return wins / len(self.recent_outcomes)

    def _route_expert(self, obs: np.ndarray, info: dict[str, Any]) -> str | None:
        legal = info.get("legal_moves") or []
        is_rps = len(legal) <= 6 and max(legal, default=0) <= 5
        recent_win_rate = self._recent_win_rate()
        has_history = self._has_profile_history()

        if not is_rps:
            if has_history:
                self.last_route_reason = "Using profile_counter because this game is outside the main RPS+ expert family and player history is available."
                return "profile_counter"
            self.last_route_reason = "Using a safe legal-move fallback because this game is outside the main RPS+ expert family and there is no player history yet."
            return None

        if has_history and recent_win_rate is not None and recent_win_rate < 0.45:
            self.last_route_reason = "Switching to profile_counter because saved player history exists and recent performance dipped."
            return "profile_counter"

        if recent_win_rate is None:
            self.last_route_reason = "Opening with sft to imitate learned play style before enough match feedback accumulates."
            return "sft"

        if recent_win_rate < 0.35:
            self.last_route_reason = "Switching to ppo because recent results are weak and the router wants a more competitive policy."
            return "ppo"

        if recent_win_rate > 0.7:
            self.last_route_reason = "Switching to sft because the router is comfortably ahead and can afford the more behavior-shaped expert."
            return "sft"

        self.last_route_reason = "Using bc_rl as the middle ground between fidelity and competitiveness for the current match state."
        return "bc_rl"

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = list(info.get("legal_moves") or [])
        if not legal:
            self.last_selected_expert = None
            self.last_route_reason = "No legal moves available."
            return 0

        expert_name = self._route_expert(obs, info)
        if expert_name is None:
            self.last_selected_expert = "safe_fallback"
            self.selection_counts["safe_fallback"] += 1
            return int(self.rng.choice(legal))
        expert = self.experts[expert_name]
        action = expert.act(obs, info)
        try:
            action = int(action)
        except Exception:
            action = legal[0]
        if action not in legal:
            action = legal[0]

        self.last_selected_expert = expert_name
        self.selection_counts[expert_name] += 1
        return action
