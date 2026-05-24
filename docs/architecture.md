# Architecture

## Inference Engines

Every engine emits the same normalized metric schema (`serving/base.py`). Swap by name from CLI or API, no code changes needed.

| Engine | Description |
|--------|-------------|
| baseline | HuggingFace autoregressive generation |
| vllm | vLLM with OpenAI-compatible server |
| preble | Shared-prefix / KV-cache reuse |
| infercept | Split prefill and resumed-decode |
| mock | Deterministic CPU engine for CI and local dev |

---

## Profiling Modules

Located in `analysis/`

| Module | Measures |
|--------|----------|
| kv_cache_profiler.py | KV-cache memory growth over sequence length and turn count |
| prefill_decode_split.py | Prefill vs. decode latency decomposition |
| throughput_benchmark.py | Requests/sec across concurrency levels |
| scheduling_overhead.py | CPU time outside model execution |

---

## Data Layer

Default DB: `data/game_data.db`  
Schema: `data/schemas.py`  
Override path: `DOPPELGAMER_DB_PATH`

| Table | Contents |
|-------|----------|
| inference_benchmarks | Per-engine, per-turn latency/token/KV metrics keyed by run ID |
| games, rounds | Gameplay traces and turn-level actions |
| agent_results | Policy benchmark summaries |
| player_profiles, impostor_results, detection_sessions | Clone/evaluation artifacts |

SQLite foreign keys are enabled at connection setup. DB files and secrets are gitignored.
