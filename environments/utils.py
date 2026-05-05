"""Utilities for running RPS+ episodes and rendering history."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from environments.rps_plus import Move, RPSPlusEnv, RoundResult


Policy = Callable[[np.ndarray, dict], int]


def run_episode(
    env: RPSPlusEnv,
    policy: Policy,
    seed: int | None = None,
) -> list[RoundResult]:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action = policy(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return list(env.state.history)


def history_to_records(history: Iterable[RoundResult], game_id: str) -> list[dict]:
    return [
        {
            "game_id": game_id,
            "turn": r.turn,
            "agent_move": int(r.agent_move),
            "agent_move_name": r.agent_move.name,
            "opponent_move": int(r.opponent_move),
            "opponent_move_name": r.opponent_move.name,
            "outcome": r.outcome,
            "agent_energy_after": r.agent_energy_after,
            "opponent_energy_after": r.opponent_energy_after,
        }
        for r in history
    ]


def random_policy(seed: int | None = None) -> Policy:
    rng = np.random.default_rng(seed)

    def policy(obs: np.ndarray, info: dict) -> int:
        legal = info.get("legal_moves") or list(range(len(Move)))
        return int(rng.choice(legal))

    return policy
