"""War (Card Game) environment with Gymnasium-compatible API.

This game is deterministic after shuffle and requires no strategy. 
It is purely for systems throughput benchmarking and basic agent functionality tests.

Action space  : Discrete(1) — Only 1 action (Play top card)
Observation   : float32 array of shape (4,) — [agent_deck_size, opp_deck_size, agent_last_card, opp_last_card]
Reward        : +1.0 win, -1.0 loss, 0.0 draw / ongoing
"""

from __future__ import annotations

from typing import Any
import numpy as np

from environments.tic_tac_toe import EnvState, RoundResult


class WarEnv:
    """War Card Game Environment."""
    
    name = "war"
    
    def __init__(self, max_moves: int = 2000, seed: int | None = None) -> None:
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        self.state = EnvState()
        self.agent_deck = []
        self.opp_deck = []
        self.agent_last_card = 0
        self.opp_last_card = 0
        
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        # 52 cards: values 2 to 14, 4 suits
        deck = [val for val in range(2, 15) for _ in range(4)]
        self._rng.shuffle(deck)
        
        self.agent_deck = deck[:26]
        self.opp_deck = deck[26:]
        self.agent_last_card = 0
        self.opp_last_card = 0
        
        self.state = EnvState()
        return self._obs(), self._info()

    def legal_moves(self) -> list[int]:
        if not self.agent_deck or not self.opp_deck:
            return []
        return [0]

    def step(self, action: int):
        if not self.agent_deck or not self.opp_deck:
            return self._obs(), float(self._get_outcome()), True, False, self._info()
            
        pool = []
        
        def resolve_war():
            nonlocal pool
            # Draw 1 face up
            if not self.agent_deck:
                return -1 # Opponent wins by default
            if not self.opp_deck:
                return 1 # Agent wins by default
                
            ac = self.agent_deck.pop(0)
            oc = self.opp_deck.pop(0)
            pool.extend([ac, oc])
            self.agent_last_card = ac
            self.opp_last_card = oc
            
            if ac > oc:
                return 1
            elif oc > ac:
                return -1
            else:
                # War!
                for _ in range(3):
                    if self.agent_deck: pool.append(self.agent_deck.pop(0))
                    if self.opp_deck: pool.append(self.opp_deck.pop(0))
                return resolve_war()
                
        winner = resolve_war()
        
        # In actual war, the winner puts cards at the bottom of their deck. 
        # To avoid infinite loops in some games, shuffle the pool before adding it back.
        self._rng.shuffle(pool)
        
        if winner == 1:
            self.agent_deck.extend(pool)
            outcome_round = 1
        elif winner == -1:
            self.opp_deck.extend(pool)
            outcome_round = -1
        else:
            outcome_round = 0
            
        self.state.turn += 1
        
        terminated = not self.agent_deck or not self.opp_deck
        truncated = self.state.turn >= self.max_moves
        
        outcome = 0
        if terminated or truncated:
            outcome = self._get_outcome()
            
        self._record_history(0, 0, outcome)
        return self._obs(), float(outcome), terminated, truncated, self._info()

    def _get_outcome(self) -> int:
        if len(self.agent_deck) > len(self.opp_deck):
            self.state.agent_score = 1
            return 1
        elif len(self.opp_deck) > len(self.agent_deck):
            self.state.opponent_score = 1
            return -1
        return 0

    def _record_history(self, agent_move: int, opp_move: int, outcome: int) -> None:
        self.state.history.append(
            RoundResult(
                turn=self.state.turn,
                agent_move=agent_move,
                opponent_move=opp_move,
                outcome=outcome,
            )
        )

    def _obs(self) -> np.ndarray:
        return np.array([
            len(self.agent_deck),
            len(self.opp_deck),
            self.agent_last_card,
            self.opp_last_card
        ], dtype=np.float32)

    def render(self) -> str:
        return (
            f"War: agent_deck={len(self.agent_deck)} opp_deck={len(self.opp_deck)} "
            f"score={self.state.agent_score}:{self.state.opponent_score}"
        )

    def legal_actions(self) -> list[int]:
        return self.legal_moves()

    def _info(self) -> dict[str, Any]:
        legal = self.legal_moves()
        return {
            "legal_moves": legal,
            "n_legal": len(legal),
        }
