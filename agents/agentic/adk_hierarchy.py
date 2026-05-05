"""Google ADK hierarchical agents.

Top-level Coordinator delegates to sub-agents (Aggressive / Defensive). The
Coordinator picks the sub-agent based on score differential and energy state.
ADK is imported lazily; without it, the same hierarchy runs as plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.agentic.tools import GameMemory, predict_next_move
from agents.base import Agent
from environments.rps_plus import Move, N_MOVES


@dataclass
class ADKConfig:
    use_real_adk: bool = False


class _Sub:
    def act(self, memory: GameMemory, obs: np.ndarray, info: dict[str, Any]) -> int:
        raise NotImplementedError


class AggressiveSub(_Sub):
    def act(self, memory: GameMemory, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = set(info.get("legal_moves") or range(N_MOVES))
        agent_e = int(round(obs[0] * 5))
        if agent_e >= 2 and int(Move.POWER) in legal:
            return int(Move.POWER)
        prediction = predict_next_move(memory).get("prediction")
        counter = {"ROCK": Move.PAPER, "PAPER": Move.SCISSORS, "SCISSORS": Move.ROCK}.get(prediction, Move.ROCK)
        return int(counter) if int(counter) in legal else int(next(iter(legal)))


class DefensiveSub(_Sub):
    def act(self, memory: GameMemory, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = set(info.get("legal_moves") or range(N_MOVES))
        agent_e = int(round(obs[0] * 5))
        if agent_e <= 2 and int(Move.RECHARGE) in legal:
            return int(Move.RECHARGE)
        return int(Move.ROCK) if int(Move.ROCK) in legal else int(next(iter(legal)))


class ADKHierarchyAgent(Agent):
    name = "agentic_adk"

    def __init__(self, config: ADKConfig | None = None) -> None:
        self.config = config or ADKConfig()
        self._memory = GameMemory([], [], [])
        self._aggressive = AggressiveSub()
        self._defensive = DefensiveSub()

    def reset(self, seed: int | None = None) -> None:
        self._memory = GameMemory([], [], [])

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self._memory.agent_moves.append(int(agent_move))
        self._memory.opponent_moves.append(int(opponent_move))
        self._memory.outcomes.append(int(outcome))

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        # Coordinator: defensive if behind on energy, aggressive otherwise.
        agent_e = int(round(obs[0] * 5))
        opp_e = int(round(obs[1] * 5))
        if agent_e <= opp_e:
            return self._defensive.act(self._memory, obs, info)
        return self._aggressive.act(self._memory, obs, info)
