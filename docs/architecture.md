# Architecture

## Inference Engines

Every engine emits the same normalized metric schema (`serving/base.py`). Swap by name from CLI or API. Only requested engines are constructed (`setup_inference_engines(cfg, engines=[...])`).

| Engine | Description |
|--------|-------------|
| baseline | HuggingFace single-pass autoregressive generation (not thread-safe) |
| vllm | vLLM library mode; `generate_batch` for continuous batching |
| hf_prefix_cache | Local HF ablation of shared-prefix KV reuse |
| hf_tool_interrupt | Local HF ablation of tool-interrupt resume |
| preble | **Remote only** — OpenAI-compatible client; set `PREBLE_BASE_URL` or `DOPPELGAMER_PREBLE_URL` |
| infercept | **Remote only** — OpenAI-compatible client; set `INFERCEPT_BASE_URL` or `DOPPELGAMER_INFERCEPT_URL` |
| mock | Deterministic CPU engine for CI and local dev |

Do not report `hf_prefix_cache` / `hf_tool_interrupt` as Preble or InferCept. Those names require a live cluster URL and fail loud without it.

Default systems benchmarks use `baseline` and `vllm` only. Real models fail loud unless `allow_fallback=true`. Fallback rows are stored as `requested→actual` (for example `vllm→huggingface`).

Factory: `inference/setup_inference_engines.py`. Remote client: `serving/openai_compat.py`.

---

## Profiling Modules

Located in `analysis/`

| Module | Measures |
|--------|----------|
| kv_cache_profiler.py | Formula estimates **or** engine-reported `kv_cache_mb` via `profile_engine_kv` |
| prefill_decode_split.py | Prefill vs decode from engine TTFT/TPOT (no invented 70/30 splits) |
| throughput_benchmark.py | `sequential` (HF), `engine_batch` (vLLM), or `threaded_clients` (remote only) |
| scheduling_overhead.py | Host idle wait (`wall - cpu`, includes GPU wait) or engine-reported overhead |

Publication protocol:

```bash
python scripts/run_publication_benchmark.py --model distilgpt2 --rounds 50 --out results/publication
```

Writes metadata, profiler JSON, and CSV under `--out`. Default DB: `data/publication_run.db`.

---

## Evaluation runner

`evaluation/runner.py`:

- **Agents:** `rounds` = number of games (RPS+ uses 20 turns each). Win/loss/tie are game-level.
- **Fidelity:** reference policy is `sft` if in the agent list, else `heuristic`; reference play is reused when that agent is already run.
- **Systems:** game-driven multi-turn prompts with a shared system prefix (not unique `[turn=N]` suffixes).
- **Illegal actions:** runner clamps to a legal move; RPS+ `env.step` still raises if an illegal action reaches the env.

---

## Data Layer

Default DB: `data/game_data.db`  
Schema: `data/schemas.py`  
Override path: `DOPPELGAMER_DB_PATH`

| Table | Contents |
|-------|----------|
| inference_benchmarks | Per-engine latency/token/KV/prefix-cache/`actual_backend` metrics |
| games, rounds | Gameplay traces and turn-level actions |
| agent_results | Policy summaries (`trained_vs_fallback`, `checkpoint_path` when present) |
| player_profiles, impostor_results, detection_sessions | Clone/evaluation artifacts |

SQLite foreign keys are enabled at connection setup. DB files and secrets are gitignored.

Export: `python scripts/export_results.py --db data/game_data.db --out results/`.

---

## HTTP API

`main.py` (Doppelgamer API):

- `POST /benchmark` with allowlisted `model_name` (`mock`, `distilgpt2`, `gpt2`, plus `DOPPELGAMER_ALLOWED_MODELS`)
- Non-mock models require `DOPPELGAMER_API_KEY` and header `X-API-Key`
- Loader errors are not echoed to clients

---

## GPU setup

See [brev_setup.md](brev_setup.md) for NVIDIA Brev install, publication steps, Preble/InferCept URLs, and cost estimates.
