"""CrewAI multi-agent orchestration: Analyst → Strategist → Executor.

The crew is constructed lazily so this module can be imported without crewai
installed. The fallback path uses the deterministic logic shared with
ReActAgent so tests still cover the orchestration shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.agentic.tools import GameMemory, analyze_opponent, predict_next_move
from agents.base import Agent
from environments.rps_plus import Move, N_MOVES


@dataclass
class CrewConfig:
    llm_model: str = "gpt-4o-mini"
    use_real_crewai: bool = False


class CrewAgent(Agent):
    """Three-role crew. With crewai installed, each role is an LLM agent.
    Without it, falls back to a Python pipeline that mirrors the same flow.
    """

    name = "agentic_crew"

    def __init__(self, config: CrewConfig | None = None) -> None:
        self.config = config or CrewConfig()
        self._memory = GameMemory([], [], [])

    def reset(self, seed: int | None = None) -> None:
        self._memory = GameMemory([], [], [])

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self._memory.agent_moves.append(int(agent_move))
        self._memory.opponent_moves.append(int(opponent_move))
        self._memory.outcomes.append(int(outcome))

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        if self.config.use_real_crewai:
            return self._act_with_crewai(obs, info)
        return self._act_fallback(obs, info)

    # ------------------------------------------------------------- fallback path

    def _act_fallback(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        analysis = analyze_opponent(self._memory)
        prediction = predict_next_move(self._memory)
        legal = set(info.get("legal_moves") or range(N_MOVES))
        target = prediction.get("prediction") or analysis.get("most_frequent")
        counter = {
            "ROCK": Move.PAPER,
            "PAPER": Move.SCISSORS,
            "SCISSORS": Move.ROCK,
            "LIZARD": Move.ROCK,
            "POWER": Move.POWER,
            "RECHARGE": Move.POWER,
        }
        move = counter.get(target, Move.ROCK)
        if int(move) in legal:
            return int(move)
        return int(next(iter(legal)))

    # --------------------------------------------------------------- crewai path

    def _act_with_crewai(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        from crewai import Agent as CAgent, Crew, Task

        analyst = CAgent(role="Analyst", goal="Detect opponent patterns", backstory="Game theorist.")
        strategist = CAgent(role="Strategist", goal="Choose counter strategy", backstory="Tactician.")
        executor = CAgent(role="Executor", goal="Output a single move name", backstory="Player.")
        history = analyze_opponent(self._memory)
        prediction = predict_next_move(self._memory)
        ctx = f"history={history}\nprediction={prediction}\nlegal={info.get('legal_moves')}"
        tasks = [
            Task(description=f"Summarize patterns. {ctx}", agent=analyst, expected_output="bullet list"),
            Task(description="Pick a counter move name from legal moves.", agent=strategist, expected_output="move name"),
            Task(description="Output ONLY the move name.", agent=executor, expected_output="MOVE_NAME"),
        ]
        crew = Crew(agents=[analyst, strategist, executor], tasks=tasks)
        result = str(crew.kickoff()).strip().upper()
        legal = info.get("legal_moves") or list(range(N_MOVES))
        for m in Move:
            if m.name in result and int(m) in legal:
                return int(m)
        return int(legal[0])
