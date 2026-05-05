"""Tool definitions for the agentic LLM. Pure-Python — no LLM dependency.

These are kept framework-agnostic; LangChain / CrewAI wrappers are added
in `react_agent.py` and `crew.py`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from environments.rps_plus import Move, N_MOVES


@dataclass
class GameMemory:
    agent_moves: list[int]
    opponent_moves: list[int]
    outcomes: list[int]


def get_game_history(memory: GameMemory, last_n: int = 5) -> list[dict]:
    """Return the last N rounds as structured records."""
    n = min(last_n, len(memory.agent_moves))
    return [
        {
            "turn": len(memory.agent_moves) - n + i + 1,
            "you": Move(memory.agent_moves[-n + i]).name,
            "opponent": Move(memory.opponent_moves[-n + i]).name,
            "outcome": memory.outcomes[-n + i],
        }
        for i in range(n)
    ]


def analyze_opponent(memory: GameMemory) -> dict:
    """Detect simple play patterns: most-frequent move, recent streak, energy bias."""
    if not memory.opponent_moves:
        return {"summary": "no data", "most_frequent": None, "streak": 0}
    counts = Counter(memory.opponent_moves)
    most_common, freq = counts.most_common(1)[0]
    last = memory.opponent_moves[-1]
    streak = 0
    for m in reversed(memory.opponent_moves):
        if m == last:
            streak += 1
        else:
            break
    return {
        "most_frequent": Move(most_common).name,
        "most_frequent_pct": freq / len(memory.opponent_moves),
        "current_streak_move": Move(last).name,
        "current_streak_len": streak,
        "n_observed": len(memory.opponent_moves),
    }


def predict_next_move(memory: GameMemory) -> dict:
    """Predict the opponent's next move via a first-order Markov estimate."""
    if len(memory.opponent_moves) < 2:
        return {"prediction": None, "confidence": 0.0}
    transitions = np.zeros((N_MOVES, N_MOVES), dtype=np.float64)
    for a, b in zip(memory.opponent_moves[:-1], memory.opponent_moves[1:]):
        transitions[a, b] += 1
    last = memory.opponent_moves[-1]
    row = transitions[last]
    if row.sum() == 0:
        return {"prediction": None, "confidence": 0.0}
    probs = row / row.sum()
    best = int(probs.argmax())
    return {
        "prediction": Move(best).name,
        "confidence": float(probs[best]),
        "distribution": {Move(i).name: float(probs[i]) for i in range(N_MOVES)},
    }


TOOLS = {
    "get_game_history": get_game_history,
    "analyze_opponent": analyze_opponent,
    "predict_next_move": predict_next_move,
}
