---
name: debug-with-tests
description: >-
  Reproduce bugs with a failing test first, fix, and record the root cause.
  Use when debugging failures, fixing correctness bugs, or investigating
  flaky or wrong metrics, counters, or training crashes.
---

# Debug with tests

Superpowers-style discipline for this repo.

## Workflow

```
- [ ] Reproduce: write or extend a failing test under tests/
- [ ] Confirm failure: pytest path/to/test.py -q
- [ ] Fix the minimal root cause (prefer single source of truth)
- [ ] Confirm pass: pytest path/to/test.py -q then pytest -q
- [ ] Record gotcha in .cursor/memory/learnings.md if non-obvious
```

## Patterns that already bit us

| Symptom | Likely cause |
|---------|----------------|
| Agent always loses to POWER | Local counter maps POWER -> RECHARGE |
| SFT data KeyError on opponent context | Wrong column (`win_rate_vs_opponent`) |
| SFT train TypeError on TrainingArguments | Use `eval_strategy`, not `evaluation_strategy` |
| Engine works only on GPU laptop | Missing mock fallback in factory |

## Rules

- Prefer a regression test over a one-off script.
- Do not "fix" by silencing exceptions unless the product intentionally falls back (engines, optional checkpoints).
- If the bug is a duplicated constant, centralize it (see `BEST_COUNTER`) rather than patching every copy only.
- Append learnings with date and file paths; keep entries short.
