---
name: add-game-environment
description: >-
  Add a new Gymnasium-style game environment. Use when creating a new game,
  board, or workload generator under environments/, or when wiring a game into
  the dashboard or benchmarks.
---

# Add a game environment

## Checklist

```
- [ ] Implement env class with reset / step / legal_moves
- [ ] Seeded determinism (same seed => same trajectory)
- [ ] Export from environments/__init__.py
- [ ] Add GAME_RULE_SPECS entry in environments/game_specs.py if dashboard-facing
- [ ] Register in evaluation runner / agent benchmarks if needed
- [ ] Tests in tests/ mirroring test_environment.py or test_future_games.py
- [ ] pytest -q passes
```

## Contract

```python
def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
    ...

def step(self, action: int):
    # returns obs, reward, terminated, truncated, info
    ...

def legal_moves(self) -> list[int]:
    ...
```

`info` must include `legal_moves`. Prefer shared `EnvState` / `RoundResult` patterns from existing envs.

## Patterns

- Full arena games: `environments/rps_plus.py`, `tic_tac_toe.py`, `chess_env.py`
- Lightweight smoke games: `environments/future_games.py` (`_SimpleEnv` base)
- Future games register in `FUTURE_GAME_ENVS` dict

## Rules

- Illegal moves: raise `ValueError` for main arena games.
- RPS+ counters: import `BEST_COUNTER` from `rps_plus.py` only.
- Keep envs free of heavy optional deps unless lazy-imported (chess uses python-chess).
