"""Unified Agentic LLM wrapper.

Exposes one interface while allowing strategy selection across:
- ReAct loop
- Crew-style orchestration
- Hierarchical delegation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from agents.agentic.adk_hierarchy import ADKHierarchyAgent
from agents.agentic.crew import CrewAgent
from agents.agentic.react_agent import ReActAgent
from agents.base import Agent

AgenticMode = Literal["react", "crew", "adk"]


@dataclass
class AgenticConfig:
    mode: AgenticMode = "react"


class AgenticLLM(Agent):
    name = "agentic"

    def __init__(self, config: AgenticConfig | None = None) -> None:
        self.config = config or AgenticConfig()
        if self.config.mode == "crew":
            self.impl: Agent = CrewAgent()
        elif self.config.mode == "adk":
            self.impl = ADKHierarchyAgent()
        else:
            self.impl = ReActAgent()

    def reset(self, seed: int | None = None) -> None:
        self.impl.reset(seed=seed)

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        obs_fn = getattr(self.impl, "observe", None)
        if callable(obs_fn):
            obs_fn(agent_move, opponent_move, outcome)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        try:
            return self.impl.act(obs, info)
        except Exception as e:
            # Fallback to random if LLM fails
            import random
            legal = info.get("legal_moves", list(range(6)))  # Assume 6 moves
            return random.choice(legal) if legal else 0
