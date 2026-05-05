"""RPS+ Gymnasium environment.

RPS+ has six moves:
- ROCK, PAPER, SCISSORS, LIZARD (base moves; no energy cost)
- POWER (costs 2 energy; beats every base move)
- RECHARGE (gains 1 energy; vulnerable to attacks)

Observation is compact and normalized for simple policy nets:
    [agent_energy/5, opponent_energy/5, turn/max_turns]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback for minimal runtime
    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def contains(self, x: int) -> bool:
            return isinstance(x, int) and 0 <= x < self.n

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _Env:
        metadata: dict[str, list[str]] = {}

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()

N_MOVES = 6
MAX_ENERGY = 5


class Move(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    LIZARD = 3
    POWER = 4
    RECHARGE = 5


BASE_WINS = {
    (Move.ROCK, Move.SCISSORS),
    (Move.ROCK, Move.LIZARD),
    (Move.PAPER, Move.ROCK),
    (Move.SCISSORS, Move.PAPER),
    (Move.SCISSORS, Move.LIZARD),
    (Move.LIZARD, Move.PAPER),
}


BEST_COUNTER = {
    Move.ROCK: Move.PAPER,
    Move.PAPER: Move.SCISSORS,
    Move.SCISSORS: Move.ROCK,
    Move.LIZARD: Move.ROCK,
    Move.POWER: Move.RECHARGE,
    Move.RECHARGE: Move.POWER,
}


@dataclass
class RoundResult:
    turn: int
    agent_move: Move
    opponent_move: Move
    outcome: int
    agent_energy_after: int
    opponent_energy_after: int


@dataclass
class EnvState:
    turn: int = 0
    agent_energy: int = 3
    opponent_energy: int = 3
    agent_score: int = 0
    opponent_score: int = 0
    history: list[RoundResult] = field(default_factory=list)


def resolve(agent: Move, opponent: Move) -> int:
    """Resolve one round outcome.

    Returns:
      +1 if agent wins, -1 if opponent wins, 0 tie.
    """
    if agent == opponent:
        return 0
    if agent == Move.RECHARGE and opponent != Move.RECHARGE:
        return -1
    if opponent == Move.RECHARGE and agent != Move.RECHARGE:
        return 1
    if agent == Move.POWER and opponent in (Move.ROCK, Move.PAPER, Move.SCISSORS, Move.LIZARD):
        return 1
    if opponent == Move.POWER and agent in (Move.ROCK, Move.PAPER, Move.SCISSORS, Move.LIZARD):
        return -1
    if (agent, opponent) in BASE_WINS:
        return 1
    if (opponent, agent) in BASE_WINS:
        return -1
    return 0


class RPSPlusEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_turns: int = 50, seed: int | None = None, starting_energy: int = 3) -> None:
        super().__init__()
        self.max_turns = max_turns
        self.starting_energy = max(0, min(MAX_ENERGY, int(starting_energy)))
        self.action_space = spaces.Discrete(N_MOVES)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.state = EnvState(agent_energy=self.starting_energy, opponent_energy=self.starting_energy)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = EnvState(agent_energy=self.starting_energy, opponent_energy=self.starting_energy)
        return self._obs(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        action_m = Move(action)
        legal = self.legal_moves(agent=True)
        if action_m not in legal:
            action_m = Move.RECHARGE if Move.RECHARGE in legal else legal[0]

        opponent_m = self._opponent_policy()
        outcome = resolve(action_m, opponent_m)

        self._apply_energy(action_m, opponent_m)
        self.state.turn += 1
        if outcome > 0:
            self.state.agent_score += 1
        elif outcome < 0:
            self.state.opponent_score += 1

        self.state.history.append(
            RoundResult(
                turn=self.state.turn,
                agent_move=action_m,
                opponent_move=opponent_m,
                outcome=outcome,
                agent_energy_after=self.state.agent_energy,
                opponent_energy_after=self.state.opponent_energy,
            )
        )

        terminated = self.state.turn >= self.max_turns
        truncated = False
        return self._obs(), float(outcome), terminated, truncated, self._info()

    def render(self):
        last = self.state.history[-1] if self.state.history else None
        return (
            f"turn={self.state.turn} score={self.state.agent_score}:{self.state.opponent_score} "
            f"energy={self.state.agent_energy}/{self.state.opponent_energy} last={last}"
        )

    # --------------------------------------------------------------- internals

    def legal_moves(self, agent: bool) -> list[Move]:
        energy = self.state.agent_energy if agent else self.state.opponent_energy
        legal = [Move.ROCK, Move.PAPER, Move.SCISSORS, Move.LIZARD, Move.RECHARGE]
        if energy >= 2:
            legal.append(Move.POWER)
        return legal

    def legal_actions(self) -> list[int]:
        return [int(m) for m in self.legal_moves(agent=True)]

    def _opponent_policy(self) -> Move:
        legal = self.legal_moves(agent=False)
        # Slightly favor RECHARGE when low on energy to create realistic dynamics.
        if self.state.opponent_energy <= 1 and Move.RECHARGE in legal and self.rng.random() < 0.6:
            return Move.RECHARGE
        return Move(int(self.rng.choice([int(m) for m in legal])))

    def _apply_energy(self, agent_move: Move, opponent_move: Move) -> None:
        if agent_move == Move.POWER:
            self.state.agent_energy = max(0, self.state.agent_energy - 2)
        elif agent_move == Move.RECHARGE:
            self.state.agent_energy = min(MAX_ENERGY, self.state.agent_energy + 1)

        if opponent_move == Move.POWER:
            self.state.opponent_energy = max(0, self.state.opponent_energy - 2)
        elif opponent_move == Move.RECHARGE:
            self.state.opponent_energy = min(MAX_ENERGY, self.state.opponent_energy + 1)

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self.state.agent_energy / MAX_ENERGY,
                self.state.opponent_energy / MAX_ENERGY,
                self.state.turn / max(1, self.max_turns),
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        return {"legal_moves": [int(m) for m in self.legal_moves(agent=True)]}
