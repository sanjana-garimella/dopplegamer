---
name: add-agent
description: >-
  Add a new agent policy to AGENT_REGISTRY. Use when creating RL, SFT, impostor,
  heuristic, or agentic agents under agents/ or impostor/, or wiring checkpoints.
---

# Add an agent

## Checklist

```
- [ ] Subclass agents.base.Agent
- [ ] Implement act(obs, info); optional reset / observe
- [ ] Respect info["legal_moves"] (never return illegal actions)
- [ ] register_agent("name", Cls) in agents/__init__.py
- [ ] Checkpoint path via agents.checkpoints.resolve_checkpoint if trained
- [ ] Smoke: python scripts/benchmark.py agents --agents <name> --rounds 10
- [ ] pytest -q
```

## Contract

```python
class MyAgent(Agent):
    name = "my_agent"

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = info["legal_moves"]
        ...

    def reset(self, seed: int | None = None) -> None:
        ...

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        ...
```

Base `Agent.act` validates `obs`, `info`, and non-empty `legal_moves`.

## Registration

```python
# agents/__init__.py
register_agent("my_agent", MyAgent)
```

Aliases (e.g. `ppo` / `rl`) are fine if they point at the same class.

## Checkpoints

Use `resolve_checkpoint("sft" | "rl" | "bc_rl" | ...)` from `agents/checkpoints.py`.
Prefer `checkpoints/<name>_real` then `*_best` patterns already listed there.

## Rules

- RPS+ best-response: import `BEST_COUNTER` from `environments/rps_plus.py`.
- Fail open without checkpoints (heuristic / random fallback) so CI stays green.
- Do not hard-code action space size; always use `legal_moves`.
