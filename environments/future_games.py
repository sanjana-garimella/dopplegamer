"""Lightweight environments for the future-game benchmark set.

These are compact simulations, not full commercial-game clones. They share the
same reset/step/legal_moves contract as the existing dashboard environments so
agents can be smoke-tested across a broader range of decision shapes without
pulling in heavyweight external simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from environments.tic_tac_toe import EnvState, RoundResult


@dataclass(frozen=True)
class _DiscreteSpace:
    n: int

    def contains(self, x: int) -> bool:
        return isinstance(x, int) and 0 <= x < self.n


class _SimpleEnv:
    name = "simple"
    action_n = 1

    def __init__(self, max_moves: int = 100, seed: int | None = None) -> None:
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self.state = EnvState()
        self.action_space = _DiscreteSpace(self.action_n)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = EnvState()
        self._reset_state()
        return self._obs(), self._info()

    def _reset_state(self) -> None:
        return None

    def legal_moves(self) -> list[int]:
        return list(range(self.action_n))

    def _record(self, action: int, opponent_action: int, outcome: int) -> None:
        self.state.history.append(
            RoundResult(
                turn=self.state.turn,
                agent_move=int(action),
                opponent_move=int(opponent_action),
                outcome=int(outcome),
            )
        )

    def _finish(self, action: int, opponent_action: int, outcome: int, terminated: bool):
        self.state.turn += 1
        truncated = self.state.turn >= self.max_moves and not terminated
        final_outcome = outcome
        if terminated or truncated:
            final_outcome = self._outcome()
        self._record(action, opponent_action, final_outcome)
        return self._obs(), float(final_outcome), terminated, truncated, self._info()

    def _outcome(self) -> int:
        return 0

    def _obs(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        legal = self.legal_moves()
        return {"legal_moves": legal, "n_legal": len(legal)}


class TwentyFortyEightEnv(_SimpleEnv):
    """2048 on a 4x4 board. Actions: up, right, down, left."""

    name = "2048"
    action_n = 4

    def __init__(self, max_moves: int = 100, seed: int | None = None, stop_at_win: bool = True) -> None:
        self.stop_at_win = stop_at_win
        super().__init__(max_moves=max_moves, seed=seed)

    def _reset_state(self) -> None:
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.won = False
        self._spawn()
        self._spawn()

    def _spawn(self) -> None:
        empties = list(zip(*np.where(self.board == 0), strict=False))
        if empties:
            r, c = empties[int(self._rng.integers(len(empties)))]
            self.board[r, c] = 4 if self._rng.random() < 0.1 else 2

    def _slide_line(self, line: np.ndarray) -> tuple[np.ndarray, int]:
        values = [int(x) for x in line if x]
        merged: list[int] = []
        reward = 0
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                val = values[i] * 2
                merged.append(val)
                reward += val
                i += 2
            else:
                merged.append(values[i])
                i += 1
        merged.extend([0] * (4 - len(merged)))
        return np.array(merged, dtype=np.int32), reward

    def _move(self, action: int) -> tuple[bool, int]:
        old = self.board.copy()
        reward = 0
        board = self.board
        if action in (0, 2):
            for c in range(4):
                line = board[:, c] if action == 0 else board[::-1, c]
                new, gained = self._slide_line(line)
                if action == 0:
                    board[:, c] = new
                else:
                    board[::-1, c] = new
                reward += gained
        else:
            for r in range(4):
                line = board[r, ::-1] if action == 1 else board[r, :]
                new, gained = self._slide_line(line)
                if action == 1:
                    board[r, ::-1] = new
                else:
                    board[r, :] = new
                reward += gained
        self.score += reward
        return not np.array_equal(old, self.board), reward

    def legal_moves(self) -> list[int]:
        legal = []
        for action in range(4):
            clone = self.board.copy()
            score = self.score
            changed, _ = self._move(action)
            self.board = clone
            self.score = score
            if changed:
                legal.append(action)
        return legal

    def step(self, action: int):
        legal = self.legal_moves()
        if legal and action not in legal:
            raise ValueError(f"Illegal 2048 move: {action}")
        changed, gained = (False, 0)
        if legal:
            changed, gained = self._move(action)
        if changed:
            self._spawn()
            if int(self.board.max()) >= 2048:
                self.won = True
        terminated = (self.won and self.stop_at_win) or not self.legal_moves()
        obs, _, _, truncated, info = self._finish(action, -1, 0, terminated)
        return obs, float(gained), terminated, truncated, info

    def _outcome(self) -> int:
        won = self.won or int(self.board.max()) >= 2048
        self.won = won
        if won:
            self.state.agent_score = 1
        return 1 if won else 0

    def _obs(self) -> np.ndarray:
        return np.log2(np.maximum(self.board, 1)).flatten().astype(np.float32) / 11.0


class WordleEnv(_SimpleEnv):
    """Wordle-style environment with duplicate-letter aware feedback."""

    name = "wordle"
    ANSWERS = ("cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade")
    GUESSES = ANSWERS + ("adieu", "crane", "slate", "stare", "tears", "canny", "allay", "boost")
    action_n = len(GUESSES)

    def __init__(self, max_moves: int = 6, seed: int | None = None) -> None:
        super().__init__(max_moves=max_moves, seed=seed)

    def _reset_state(self) -> None:
        self.target_idx = int(self._rng.integers(len(self.ANSWERS)))
        self.last_feedback = np.zeros(5, dtype=np.float32)
        self.guess_history: list[tuple[str, list[int]]] = []
        self.n_guesses = 0

    def legal_moves(self) -> list[int]:
        return list(range(len(self.GUESSES)))

    @property
    def WORDS(self):
        return self.GUESSES

    def _score_guess(self, guess: str, target: str) -> list[int]:
        feedback = [0] * 5
        remaining: dict[str, int] = {}
        for i, (g, t) in enumerate(zip(guess, target)):
            if g == t:
                feedback[i] = 2
            else:
                remaining[t] = remaining.get(t, 0) + 1
        for i, g in enumerate(guess):
            if feedback[i] == 2:
                continue
            if remaining.get(g, 0) > 0:
                feedback[i] = 1
                remaining[g] -= 1
        return feedback

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            raise ValueError(f"Illegal Wordle guess index: {action}")
        guess = self.GUESSES[action]
        target = self.ANSWERS[self.target_idx]
        feedback = self._score_guess(guess, target)
        self.last_feedback = np.array(feedback, dtype=np.float32)
        self.guess_history.append((guess, feedback))
        self.n_guesses += 1
        terminated = guess == target or self.n_guesses >= self.max_moves
        outcome = 1 if guess == target else 0
        return self._finish(action, self.target_idx, outcome, terminated)

    def _outcome(self) -> int:
        target = self.ANSWERS[self.target_idx]
        if any(guess == target for guess, _ in self.guess_history):
            self.state.agent_score = 1
            return 1
        self.state.opponent_score = 1
        return -1

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.last_feedback / 2.0,
                np.array([self.n_guesses / max(1, self.max_moves)], dtype=np.float32),
            ]
        )


class SudokuEnv(_SimpleEnv):
    """9x9 Sudoku. Action is cell * 9 + value_index."""

    name = "sudoku"
    action_n = 729

    def _reset_state(self) -> None:
        self.board = np.array(
            [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ],
            dtype=np.int8,
        )
        self.solution = np.array(
            [
                [5, 3, 4, 6, 7, 8, 9, 1, 2],
                [6, 7, 2, 1, 9, 5, 3, 4, 8],
                [1, 9, 8, 3, 4, 2, 5, 6, 7],
                [8, 5, 9, 7, 6, 1, 4, 2, 3],
                [4, 2, 6, 8, 5, 3, 7, 9, 1],
                [7, 1, 3, 9, 2, 4, 8, 5, 6],
                [9, 6, 1, 5, 3, 7, 2, 8, 4],
                [2, 8, 7, 4, 1, 9, 6, 3, 5],
                [3, 4, 5, 2, 8, 6, 1, 7, 9],
            ],
            dtype=np.int8,
        )
        self.givens = self.board != 0

    def legal_moves(self) -> list[int]:
        legal = []
        for idx, value in enumerate(self.board.flatten()):
            if value == 0:
                for digit in range(1, 10):
                    if self._is_valid(idx, digit):
                        legal.append(idx * 9 + (digit - 1))
        return legal

    def _is_valid(self, idx: int, digit: int) -> bool:
        r, c = divmod(idx, 9)
        if self.givens[r, c]:
            return False
        box_r, box_c = (r // 3) * 3, (c // 3) * 3
        return (
            digit not in self.board[r, :]
            and digit not in self.board[:, c]
            and digit not in self.board[box_r : box_r + 3, box_c : box_c + 3]
        )

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            raise ValueError(f"Illegal Sudoku action: {action}")
        cell, value_idx = divmod(action, 9)
        self.board[cell // 9, cell % 9] = value_idx + 1
        terminated = not np.any(self.board == 0) or not self.legal_moves()
        return self._finish(action, -1, 1 if terminated else 0, terminated)

    def _outcome(self) -> int:
        solved = bool(np.array_equal(self.board, self.solution))
        self.state.agent_score = int(solved)
        self.state.opponent_score = int(not solved)
        return 1 if solved else -1

    def _obs(self) -> np.ndarray:
        return self.board.flatten().astype(np.float32) / 9.0


class CandyCrushEnv(_SimpleEnv):
    """Tiny match-3 board. Action is cell * 4 + direction."""

    name = "candy_crush"
    action_n = 6 * 6 * 4

    def _reset_state(self) -> None:
        self.board = self._rng.integers(1, 6, size=(6, 6), dtype=np.int8)
        self.score = 0

    def _swap_target(self, action: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
        cell, direction = divmod(action, 4)
        r, c = divmod(cell, 6)
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
        nr, nc = r + dr, c + dc
        if 0 <= nr < 6 and 0 <= nc < 6:
            return (r, c), (nr, nc)
        return None

    def _matches(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for r in range(6):
            run = [(r, 0)]
            for c in range(1, 6):
                if self.board[r, c] == self.board[r, c - 1]:
                    run.append((r, c))
                else:
                    if len(run) >= 3:
                        cells.update(run)
                    run = [(r, c)]
            if len(run) >= 3:
                cells.update(run)
        for c in range(6):
            run = [(0, c)]
            for r in range(1, 6):
                if self.board[r, c] == self.board[r - 1, c]:
                    run.append((r, c))
                else:
                    if len(run) >= 3:
                        cells.update(run)
                    run = [(r, c)]
            if len(run) >= 3:
                cells.update(run)
        return cells

    def legal_moves(self) -> list[int]:
        legal = []
        for action in range(self.action_n):
            target = self._swap_target(action)
            if not target:
                continue
            a, b = target
            self.board[a], self.board[b] = self.board[b], self.board[a]
            if self._matches():
                legal.append(action)
            self.board[a], self.board[b] = self.board[b], self.board[a]
        return legal

    def step(self, action: int):
        legal = self.legal_moves()
        if legal and action not in legal:
            action = int(self._rng.choice(legal))
        target = self._swap_target(action) if legal else None
        if target:
            a, b = target
            self.board[a], self.board[b] = self.board[b], self.board[a]
            matches = self._matches()
            self.score += len(matches)
            for r, c in matches:
                self.board[r, c] = int(self._rng.integers(1, 6))
        terminated = self.state.turn + 1 >= self.max_moves
        return self._finish(action, -1, 1 if terminated and self.score >= 30 else 0, terminated)

    def _obs(self) -> np.ndarray:
        return self.board.flatten().astype(np.float32) / 5.0


class FlappyBirdEnv(_SimpleEnv):
    name = "flappy_bird"
    action_n = 2

    def _reset_state(self) -> None:
        self.y = 0.5
        self.velocity = 0.0
        self.pipe_x = 1.0
        self.gap_y = float(self._rng.uniform(0.25, 0.75))
        self.score = 0

    def step(self, action: int):
        if action not in (0, 1):
            action = 0
        self.velocity += -0.08 if action == 1 else 0.035
        self.y += self.velocity
        self.pipe_x -= 0.12
        terminated = self.y <= 0.0 or self.y >= 1.0
        if self.pipe_x <= 0.05:
            if abs(self.y - self.gap_y) > 0.18:
                terminated = True
            else:
                self.score += 1
                self.pipe_x = 1.0
                self.gap_y = float(self._rng.uniform(0.25, 0.75))
        return self._finish(action, -1, 0, terminated)

    def _outcome(self) -> int:
        return 1 if self.score >= 5 else -1

    def _obs(self) -> np.ndarray:
        return np.array([self.y, self.velocity, self.pipe_x, self.gap_y, self.score / 5], dtype=np.float32)


class PacManEnv(_SimpleEnv):
    name = "pacman"
    action_n = 4

    def _reset_state(self) -> None:
        self.grid = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 2, 1, 1, 1, 2, 0],
                [1, 1, 0, 1, 0, 1, 1],
                [1, 1, 1, 0, 1, 1, 1],
                [1, 1, 0, 1, 0, 1, 1],
                [0, 2, 1, 1, 1, 2, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        self.walls = self.grid == 0
        self.pellets = self.grid == 1
        self.power_pellets = self.grid == 2
        self.player_start = (3, 1)
        self.ghost_start = (3, 5)
        self.player = self.player_start
        self.ghost = self.ghost_start
        self.score = 0
        self.frightened_turns = 0
        self.ghost_respawns = 0
        self.pellets[self.player] = False
        self.power_pellets[self.player] = False

    def _move_target(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        nr, nc = pos[0] + dr, pos[1] + dc
        rows, cols = self.grid.shape
        if pos[0] == 3 and pos[1] == 0 and action == 3:
            return (3, cols - 1)
        if pos[0] == 3 and pos[1] == cols - 1 and action == 1:
            return (3, 0)
        if 0 <= nr < rows and 0 <= nc < cols and not self.walls[nr, nc]:
            return (nr, nc)
        return pos

    def legal_moves(self) -> list[int]:
        legal = []
        for action in range(self.action_n):
            if self._move_target(self.player, action) != self.player:
                legal.append(action)
        return legal or [0]

    def _ghost_action(self) -> int:
        legal = [a for a in range(self.action_n) if self._move_target(self.ghost, a) != self.ghost]
        if not legal:
            return 0
        best_action = None
        best_dist = None
        for action in legal:
            nr, nc = self._move_target(self.ghost, action)
            dist = abs(nr - self.player[0]) + abs(nc - self.player[1])
            better = dist > best_dist if self.frightened_turns > 0 and best_dist is not None else dist < best_dist if best_dist is not None else True
            if best_dist is None or better:
                best_dist = dist
                best_action = action
        if self._rng.random() < (0.45 if self.frightened_turns > 0 else 0.2):
            return int(self._rng.choice(legal))
        return int(best_action)

    def _resolve_collision(self) -> tuple[bool, int]:
        if self.player != self.ghost:
            return False, 0
        if self.frightened_turns > 0:
            self.score += 10
            self.ghost = self.ghost_start
            self.ghost_respawns += 1
            return False, 10
        return True, -1

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            raise ValueError(f"Illegal Pac-Man move: {action}")
        self.player = self._move_target(self.player, action)
        if self.pellets[self.player]:
            self.pellets[self.player] = False
            self.score += 1
        if self.power_pellets[self.player]:
            self.power_pellets[self.player] = False
            self.frightened_turns = 6
            self.score += 5

        caught, reward = self._resolve_collision()
        if caught:
            return self._finish(action, -1, reward, True)

        ghost_action = self._ghost_action()
        self.ghost = self._move_target(self.ghost, ghost_action)
        if self.frightened_turns > 0:
            self.frightened_turns -= 1

        caught, collision_reward = self._resolve_collision()
        cleared = not bool(np.any(self.pellets)) and not bool(np.any(self.power_pellets))
        terminated = caught or cleared
        outcome = collision_reward if caught else 1 if cleared else reward
        return self._finish(action, ghost_action, outcome, terminated)

    def _outcome(self) -> int:
        if self.player == self.ghost and self.frightened_turns == 0:
            self.state.opponent_score = 1
            return -1
        if not np.any(self.pellets) and not np.any(self.power_pellets):
            self.state.agent_score = 1
            return 1
        return 0

    def _obs(self) -> np.ndarray:
        rows, cols = self.grid.shape
        board = np.zeros((rows, cols), dtype=np.float32)
        board[self.walls] = -1.0
        board[self.pellets] = 0.5
        board[self.power_pellets] = 0.75
        board[self.player] = 1.0
        board[self.ghost] = -0.25 if self.frightened_turns > 0 else -0.5
        extras = np.array(
            [
                self.player[0] / max(1, rows - 1),
                self.player[1] / max(1, cols - 1),
                self.ghost[0] / max(1, rows - 1),
                self.ghost[1] / max(1, cols - 1),
                min(self.score / 50.0, 1.0),
                self.frightened_turns / 6.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([board.flatten(), extras]).astype(np.float32)


class PenaltyShootoutEnv(_SimpleEnv):
    name = "penalty_shootout"
    action_n = 5

    def _reset_state(self) -> None:
        self.goals = 0
        self.kicks = 0

    def step(self, action: int):
        if action not in self.legal_moves():
            action = int(self._rng.integers(self.action_n))
        goalie = int(self._rng.integers(self.action_n))
        scored = action != goalie and self._rng.random() > 0.12
        self.goals += int(scored)
        self.kicks += 1
        terminated = self.kicks >= min(5, self.max_moves)
        return self._finish(action, goalie, 1 if scored else -1, terminated)

    def _outcome(self) -> int:
        self.state.agent_score = self.goals
        self.state.opponent_score = min(5, self.max_moves) - self.goals
        return 1 if self.goals >= 3 else -1

    def _obs(self) -> np.ndarray:
        return np.array([self.goals / 5, self.kicks / 5], dtype=np.float32)


class CricketStrategyEnv(_SimpleEnv):
    name = "cricket_strategy"
    action_n = 4

    def _reset_state(self) -> None:
        self.runs = 0
        self.wickets = 0
        self.balls = 0
        self.target = 36

    def step(self, action: int):
        if action not in range(self.action_n):
            action = 0
        outcomes = {
            0: [(0, 0.15), (1, 0.70), (-1, 0.15)],
            1: [(1, 0.65), (2, 0.20), (-1, 0.15)],
            2: [(0, 0.20), (4, 0.55), (-1, 0.25)],
            3: [(0, 0.25), (6, 0.40), (-1, 0.35)],
        }[action]
        values, probs = zip(*outcomes, strict=False)
        result = int(self._rng.choice(values, p=probs))
        if result < 0:
            self.wickets += 1
        else:
            self.runs += result
        self.balls += 1
        terminated = self.runs >= self.target or self.wickets >= 3 or self.balls >= min(12, self.max_moves)
        return self._finish(action, result, 0, terminated)

    def _outcome(self) -> int:
        won = self.runs >= self.target
        self.state.agent_score = self.runs
        self.state.opponent_score = self.target
        return 1 if won else -1

    def _obs(self) -> np.ndarray:
        return np.array([self.runs / self.target, self.wickets / 3, self.balls / 12], dtype=np.float32)


class LudoEnv(_SimpleEnv):
    name = "ludo"
    action_n = 4

    def _reset_state(self) -> None:
        self.positions = np.full(4, -1, dtype=np.int16)
        self.opp_positions = np.full(4, -1, dtype=np.int16)
        self.dice = int(self._rng.integers(1, 7))

    def legal_moves(self) -> list[int]:
        legal = []
        for i, pos in enumerate(self.positions):
            if pos == -1 and self.dice == 6:
                legal.append(i)
            elif 0 <= pos < 56 and pos + self.dice <= 56:
                legal.append(i)
        return legal or [0]

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            action = legal[0]
        if self.positions[action] == -1 and self.dice == 6:
            self.positions[action] = 0
        elif self.positions[action] >= 0:
            self.positions[action] += self.dice
        opp_action = int(self._rng.integers(4))
        opp_dice = int(self._rng.integers(1, 7))
        if self.opp_positions[opp_action] == -1 and opp_dice == 6:
            self.opp_positions[opp_action] = 0
        elif self.opp_positions[opp_action] >= 0:
            self.opp_positions[opp_action] = min(56, self.opp_positions[opp_action] + opp_dice)
        self.dice = int(self._rng.integers(1, 7))
        terminated = bool(np.all(self.positions == 56) or np.all(self.opp_positions == 56))
        return self._finish(action, opp_action, 0, terminated)

    def _outcome(self) -> int:
        agent_done = int(np.sum(self.positions == 56))
        opp_done = int(np.sum(self.opp_positions == 56))
        self.state.agent_score = agent_done
        self.state.opponent_score = opp_done
        return 1 if agent_done >= opp_done else -1

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.positions, self.opp_positions, [self.dice]]).astype(np.float32) / 56.0


class UNOEnv(_SimpleEnv):
    name = "uno"
    action_n = 5

    def _reset_state(self) -> None:
        self.hand = np.array([2, 2, 2, 2], dtype=np.int16)
        self.opp_cards = 7
        self.current_color = int(self._rng.integers(4))

    def legal_moves(self) -> list[int]:
        legal = [i for i, count in enumerate(self.hand) if count > 0 and (i == self.current_color or self.hand[i] >= 2)]
        legal.append(4)
        return legal

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            action = legal[0]
        if action == 4:
            self.hand[int(self._rng.integers(4))] += 1
        else:
            self.hand[action] -= 1
            self.current_color = action
        if self.opp_cards > 0:
            self.opp_cards -= int(self._rng.random() < 0.65)
        terminated = bool(self.hand.sum() == 0 or self.opp_cards == 0)
        return self._finish(action, self.current_color, 0, terminated)

    def _outcome(self) -> int:
        won = self.hand.sum() <= self.opp_cards
        self.state.agent_score = int(7 - self.hand.sum())
        self.state.opponent_score = int(7 - self.opp_cards)
        return 1 if won else -1

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.hand, [self.opp_cards, self.current_color]]).astype(np.float32) / 7.0


class ScrabbleEnv(_SimpleEnv):
    name = "scrabble"
    WORDS = ("ai", "data", "agent", "cache", "model", "python", "vector", "policy")
    action_n = len(WORDS)
    LETTERS = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}

    def _reset_state(self) -> None:
        self.used: set[int] = set()
        self.score = 0
        self.opp_score = 0

    def legal_moves(self) -> list[int]:
        return [i for i in range(len(self.WORDS)) if i not in self.used]

    def _word_score(self, word: str) -> int:
        return sum(self.LETTERS[ch] for ch in word) // 4

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            action = legal[0] if legal else 0
        self.used.add(action)
        self.score += self._word_score(self.WORDS[action])
        opp_action = int(self._rng.integers(len(self.WORDS)))
        self.opp_score += max(1, self._word_score(self.WORDS[opp_action]) - 1)
        terminated = len(self.used) >= min(self.max_moves, len(self.WORDS))
        return self._finish(action, opp_action, 0, terminated)

    def _outcome(self) -> int:
        self.state.agent_score = self.score
        self.state.opponent_score = self.opp_score
        return 1 if self.score >= self.opp_score else -1

    def _obs(self) -> np.ndarray:
        return np.array([self.score / 100, self.opp_score / 100, len(self.used) / len(self.WORDS)], dtype=np.float32)


class MonopolyEnv(_SimpleEnv):
    name = "monopoly"
    action_n = 3

    def _reset_state(self) -> None:
        self.position = 0
        self.cash = 1500
        self.properties = 0
        self.opp_net = 1500

    def legal_moves(self) -> list[int]:
        legal = [0]
        if self.cash >= 100:
            legal.append(1)
        if self.properties:
            legal.append(2)
        return legal

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            action = legal[0]
        roll = int(self._rng.integers(2, 13))
        self.position = (self.position + roll) % 40
        self.cash += 50
        if action == 1:
            self.cash -= 100
            self.properties += 1
        elif action == 2:
            self.cash += 80
            self.properties -= 1
        self.opp_net += int(self._rng.integers(-40, 90))
        terminated = self.state.turn + 1 >= self.max_moves or self.cash <= 0
        return self._finish(action, roll, 0, terminated)

    def _outcome(self) -> int:
        net = self.cash + self.properties * 120
        self.state.agent_score = int(net)
        self.state.opponent_score = int(self.opp_net)
        return 1 if net >= self.opp_net else -1

    def _obs(self) -> np.ndarray:
        return np.array([self.position / 40, self.cash / 2000, self.properties / 20, self.opp_net / 2000], dtype=np.float32)


class AmongUsEnv(_SimpleEnv):
    name = "among_us"
    action_n = 6

    def _reset_state(self) -> None:
        self.impostor = int(self._rng.integers(4))
        self.tasks = 0
        self.suspicion = np.zeros(4, dtype=np.float32)

    def step(self, action: int):
        if action not in range(self.action_n):
            action = 4
        if 0 <= action < 4:
            terminated = True
            outcome = 1 if action == self.impostor else -1
        elif action == 4:
            self.tasks += 1
            terminated = self.tasks >= 5
            outcome = 1 if terminated else 0
        else:
            self.suspicion[self.impostor] += 0.4
            self.suspicion += self._rng.random(4) * 0.1
            terminated = False
            outcome = 0
        return self._finish(action, self.impostor, outcome, terminated)

    def _outcome(self) -> int:
        won = self.tasks >= 5
        self.state.agent_score = int(won)
        self.state.opponent_score = int(not won)
        return 1 if won else -1

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.suspicion, [self.tasks / 5]]).astype(np.float32)


class MinecraftEnv(_SimpleEnv):
    name = "minecraft"
    action_n = 6

    def _reset_state(self) -> None:
        self.wood = 0
        self.stone = 0
        self.food = 2
        self.tool = 0
        self.shelter = 0
        self.health = 5

    def legal_moves(self) -> list[int]:
        legal = [0, 1, 2, 5]
        if self.wood >= 2 and self.stone >= 1:
            legal.append(3)
        if self.wood >= 3 and self.stone >= 2:
            legal.append(4)
        return legal

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            action = legal[0]
        if action == 0:
            self.wood += 1 + self.tool
        elif action == 1:
            self.stone += 1 + self.tool
        elif action == 2:
            self.food += 1
        elif action == 3:
            self.wood -= 2
            self.stone -= 1
            self.tool = 1
        elif action == 4:
            self.wood -= 3
            self.stone -= 2
            self.shelter = 1
        elif action == 5:
            self.food -= 1
            self.health = min(5, self.health + 1)
        if self.food < 0:
            self.health -= 1
            self.food = 0
        terminated = self.shelter == 1 or self.health <= 0
        return self._finish(action, -1, 0, terminated)

    def _outcome(self) -> int:
        won = self.shelter == 1
        self.state.agent_score = int(won)
        self.state.opponent_score = int(not won)
        return 1 if won else -1

    def _obs(self) -> np.ndarray:
        return np.array([self.wood, self.stone, self.food, self.tool, self.shelter, self.health], dtype=np.float32) / 10


class ClashRoyaleEnv(_SimpleEnv):
    name = "clash_royale"
    action_n = 5

    def _reset_state(self) -> None:
        self.elixir = 5
        self.agent_tower = 10
        self.opp_tower = 10
        self.lane_pressure = 0

    def legal_moves(self) -> list[int]:
        costs = [0, 2, 3, 4, 1]
        return [i for i, cost in enumerate(costs) if self.elixir >= cost]

    def step(self, action: int):
        legal = self.legal_moves()
        if action not in legal:
            action = 0
        costs = [0, 2, 3, 4, 1]
        damage = [0, 1, 2, 3, 0]
        self.elixir -= costs[action]
        self.elixir = min(10, self.elixir + 2)
        self.opp_tower -= damage[action]
        if action == 4:
            self.agent_tower += 1
        opp_damage = int(self._rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20]))
        self.agent_tower -= opp_damage
        terminated = self.opp_tower <= 0 or self.agent_tower <= 0 or self.state.turn + 1 >= self.max_moves
        return self._finish(action, opp_damage, 0, terminated)

    def _outcome(self) -> int:
        self.state.agent_score = max(0, self.opp_tower)
        self.state.opponent_score = max(0, self.agent_tower)
        return 1 if self.opp_tower <= self.agent_tower else -1

    def _obs(self) -> np.ndarray:
        return np.array([self.elixir / 10, self.agent_tower / 10, self.opp_tower / 10], dtype=np.float32)


FUTURE_GAME_ENVS = {
    "candy_crush": CandyCrushEnv,
    "2048": TwentyFortyEightEnv,
    "wordle": WordleEnv,
    "sudoku": SudokuEnv,
    "pacman": PacManEnv,
    "minecraft": MinecraftEnv,
    "among_us": AmongUsEnv,
    "clash_royale": ClashRoyaleEnv,
    "flappy_bird": FlappyBirdEnv,
    "ludo": LudoEnv,
    "uno": UNOEnv,
    "scrabble": ScrabbleEnv,
    "monopoly": MonopolyEnv,
    "penalty_shootout": PenaltyShootoutEnv,
    "cricket_strategy": CricketStrategyEnv,
}
