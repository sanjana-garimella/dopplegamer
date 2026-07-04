# Learnings

Append dated entries when you hit a gotcha. Newest at the top.

## 2026-07-03

- **`PlayerProfileManager` missing-table bug:** `data/game_data.db` is gitignored, so a fresh clone (e.g. new Brev instance) starts with zero tables. `PlayerProfileManager` only called `init_extended_db` (the `EXTENDED_TABLES` set) in some methods and never called `init_db` (which creates `player_profiles`, in `ALL_TABLES`), so `create()`/`get()`/`list_all()` raised `sqlite3.OperationalError: no such table: player_profiles` as the very first DB touch. This broke `test_agent_descriptions.py` / `test_checkpoints.py` / `test_live_game_ui.py` on collection on a fresh checkout (masked locally because a long-lived dev `data/game_data.db` already had the schema). Fix: `PlayerProfileManager.__init__` now calls `init_db(self.db_path)` unconditionally (idempotent `CREATE TABLE IF NOT EXISTS`).
- **vLLM + FlashInfer needs `nvcc`:** vLLM's default sampler (FlashInfer top-k/top-p) JIT-compiles a CUDA kernel on first `LLM(...)` run, requiring the full CUDA *toolkit* (`nvcc`), not just the runtime `nvidia-smi` reports. Rented GPU boxes (Brev L4/A10G images) commonly lack `nvcc`, so vLLM fails with `RuntimeError: Could not find nvcc...` on load. Fix: `serving/vllm_server.py` sets `os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")` before importing `vllm`, forcing the PyTorch-native sampler (slower, no JIT compile needed). See `docs/brev_setup.md` Troubleshooting.
- **Host-wait bug:** `scheduling_overhead_ms` must be `None` when unreported; default `0.0` made the profiler never use wall−cpu. Publication script asserts `metric == host_wait_ms` for local engines.
- **SB3 training:** wrap RPS+ with `LegalActionWrapper` so illegal POWER does not crash PPO.
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
