"""Self-reflection module: critique a recent decision and adjust strategy."""

from __future__ import annotations

from dataclasses import dataclass

from agents.agentic.tools import GameMemory


@dataclass
class Reflection:
    summary: str
    suggested_bias: str  # "aggressive" | "defensive" | "neutral"


def reflect(memory: GameMemory, last_n: int = 10) -> Reflection:
    if not memory.outcomes:
        return Reflection(summary="no rounds yet", suggested_bias="neutral")
    recent = memory.outcomes[-last_n:]
    wins = sum(1 for o in recent if o > 0)
    losses = sum(1 for o in recent if o < 0)
    if losses > wins + 1:
        return Reflection(
            summary=f"losing the last {len(recent)} rounds ({wins}W/{losses}L); switch to defensive play",
            suggested_bias="defensive",
        )
    if wins > losses + 1:
        return Reflection(
            summary=f"winning the last {len(recent)} rounds ({wins}W/{losses}L); press the advantage",
            suggested_bias="aggressive",
        )
    return Reflection(
        summary=f"even split on last {len(recent)} rounds ({wins}W/{losses}L); hold strategy",
        suggested_bias="neutral",
    )
