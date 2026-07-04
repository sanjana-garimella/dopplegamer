---
name: memory
description: >-
  Read and update project memory under .cursor/memory/. Use at the start of
  non-trivial work in this repo, when discovering a gotcha, or when making an
  architecture decision. Covers learnings.md and decisions.md.
---

# Project memory

Persistent notes for agents working in Doppelgamer. Separate from runtime
`agents/agentic/memory.py` (in-game agent memory).

## Files

| File | Purpose |
|------|---------|
| `.cursor/memory/learnings.md` | Gotchas, API renames, bug patterns |
| `.cursor/memory/decisions.md` | Architecture and process decisions with rationale |

## When to read

At the start of non-trivial work (multi-file changes, new env/agent/engine,
debugging, training, benchmarks), read both files before editing.

## When to write

Append a dated entry (newest at the top) when you:

- Hit a non-obvious failure (wrong column name, deprecated API, bad counter map)
- Choose or change an architecture pattern others should follow

### Entry format

**learnings.md:**

```markdown
## YYYY-MM-DD

- **Short title:** One or two sentences. Include file paths when relevant.
```

**decisions.md:**

```markdown
## YYYY-MM-DD

- **Decision name:** What we chose. Rationale: why.
```

## Rules

- Append only; do not rewrite history unless correcting a factual error.
- Keep entries short. Link to files instead of pasting large code blocks.
- Do not store secrets, tokens, or personal credentials.
