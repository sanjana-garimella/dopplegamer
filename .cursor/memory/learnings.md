# Learnings

Append dated entries when you hit a gotcha. Newest at the top.

## 2026-07-03

- **Publication blockers addressed:** `preble`/`infercept` are remote-only (`PREBLE_BASE_URL` / `INFERCEPT_BASE_URL`); HF ablations stay `hf_*`. Throughput modes: sequential / engine_batch (vLLM) / threaded_clients (remote). Scheduling profiler reports `host_wait_ms` (wall-cpu), not serving scheduler. Prefill/decode never invents 70/30. Protocol: `scripts/run_publication_benchmark.py`.
- **API/engine hardening:** Lazy engine init (`setup_inference_engines(..., engines=[...])`); fallback rows labeled `requested→actual` via `_FallbackEngine`; `/benchmark` allowlists models, requires `DOPPELGAMER_API_KEY` for non-mock, caps list lengths, hides loader errors. Default battery excludes untrained ngram/lstm. Fidelity reuses reference agent play when present. Runner clamps illegal actions before `env.step` (RPS+ still raises if illegal reaches the env).
- **Credibility fixes:** Real engines fail loud unless `allow_fallback`; vLLM TTFT uses timestamp deltas; HF baseline is single-pass; systems prompts are game-driven; agent `rounds` = games (20 turns each for RPS+); fidelity references heuristic (or sft if present); Preble/Infercept renamed to `hf_prefix_cache` / `hf_tool_interrupt` with aliases.
- Agent toolkit lives under `.cursor/` (rules, memory, skills) plus root `AGENTS.md`. Runtime `agents/agentic/memory.py` is in-game agent memory, unrelated to this folder.

## 2026-06-24

- **transformers 5.x TrainingArguments:** `evaluation_strategy` was removed. Use `eval_strategy="epoch"` in `agents/sft/train.py` or training crashes on startup.
- **RPS+ POWER counter:** Mapping POWER to RECHARGE is wrong (RECHARGE loses to POWER). Best non-losing reply is POWER (tie). Canonical table is `BEST_COUNTER` in `environments/rps_plus.py`. Copies lived in `agents/base.py`, `impostor/player_profiles.py`, `impostor/metrics.py`, `agents/agentic/react_agent.py`, `agents/agentic/crew.py`.
- **SFT opponent context:** `opponent_style_summary` exposes `win_rate_vs_opponent`, not `player_win_rate` (`agents/sft/data.py`).
- **Test suite:** ~130 tests, all CPU/mock. Run `pytest -q` after changes.
- **vLLM install:** `requirements.txt` installs CPU torch; installing `vllm` replaces it with a CUDA build. See `docs/brev_setup.md` for GPU setup.
