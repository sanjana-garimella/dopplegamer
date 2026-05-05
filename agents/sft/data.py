"""Convert SQLite gameplay rounds into prompt/completion pairs for SFT.

Each round becomes a training example whose prompt encodes the recent game
context and whose completion is the human (reference) move name. The format
is intentionally LLM-friendly so the same prompts can be replayed at inference
time through any of the serving engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from data.features import cross_game_embedding, opponent_style_summary
from data.schemas import connect
from environments.rps_plus import Move


SYSTEM = (
    "You play RPS+ (rock-paper-scissors plus). Moves: ROCK, PAPER, SCISSORS, "
    "LIZARD, POWER, RECHARGE. Power costs 2 energy and beats any base move. "
    "Recharge gains 1 energy but loses if opponent attacks. Pick the next move."
)


@dataclass
class SFTExample:
    prompt: str
    completion: str


def format_prompt(
    agent_energy: int,
    opponent_energy: int,
    history: list[tuple[Move, Move, int]],
    history_window: int = 5,
    player_embedding: list[float] | None = None,
    opponent_context: str | None = None,
) -> str:
    recent = history[-history_window:]
    if recent:
        lines = [
            f"  turn {-(len(recent) - i)}: you played {a.name}, opp played {o.name} ({outcome:+d})"
            for i, (a, o, outcome) in enumerate(recent)
        ]
        history_str = "\n".join(lines)
    else:
        history_str = "  (no prior turns)"
    embedding_line = ""
    if player_embedding:
        embedding_line = f"Player style embedding: {player_embedding[:8]}\n"
    opponent_line = f"Opponent context: {opponent_context}\n" if opponent_context else ""
    return (
        f"{SYSTEM}\n\n"
        f"{embedding_line}"
        f"{opponent_line}"
        f"Your energy: {agent_energy}/5\n"
        f"Opponent energy: {opponent_energy}/5\n"
        f"Recent rounds (most recent last):\n{history_str}\n\n"
        f"Next move:"
    )


def iter_examples(db_path: str | Path, history_window: int = 5) -> Iterator[SFTExample]:
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "SELECT g.agent_name, g.opponent_name, r.game_id, r.turn, r.agent_move, r.opponent_move, "
            "r.agent_energy_after, r.opponent_energy_after FROM rounds r "
            "JOIN games g USING (game_id) ORDER BY r.game_id, r.turn"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    by_game: dict[str, tuple[str, str, list[tuple[int, int, int, int, int]]]] = {}
    for agent_name, opponent_name, game_id, turn, am, om, ae, oe in rows:
        if game_id not in by_game:
            by_game[game_id] = (agent_name, opponent_name, [])
        by_game[game_id][2].append((turn, am, om, ae, oe))

    for player_id, opponent_name, game_rows in by_game.values():
        history: list[tuple[Move, Move, int]] = []
        agent_energy = 3
        opponent_energy = 3
        embedding = cross_game_embedding(db_path, player_id).tolist()
        opponent_stats = opponent_style_summary(db_path, player_id)
        opponent_context = None
        if not opponent_stats.empty:
            matching = opponent_stats.loc[opponent_stats["opponent_name"].astype(str) == str(opponent_name)]
            if matching.empty:
                matching = opponent_stats.head(1)
            row = matching.iloc[0]
            opponent_context = (
                f"{row['opponent_name']} · seen {int(row['games'])} games · "
                f"player win rate {float(row['player_win_rate']):.0%}"
            )
        for _, am, om, ae, oe in sorted(game_rows):
            prompt = format_prompt(
                agent_energy,
                opponent_energy,
                history,
                history_window,
                embedding,
                opponent_context,
            )
            yield SFTExample(prompt=prompt, completion=Move(am).name)
            _r = resolve_quick(am, om)
            outcome_sign = 1 if _r > 0 else (-1 if _r < 0 else 0)
            history.append((Move(am), Move(om), outcome_sign))
            agent_energy = ae
            opponent_energy = oe


def resolve_quick(a: int, b: int) -> int:
    from environments.rps_plus import resolve

    return resolve(Move(a), Move(b))
