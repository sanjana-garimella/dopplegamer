# Doppelgamer

LLM inference benchmarking with agentic game workloads.

## Overview

Doppelgamer is an LLM inference-systems benchmark for comparing serving engines under agentic game workloads. It standardizes multi-engine execution across Hugging Face, vLLM, KV-cache-aware serving variants, and deterministic mocks, then records TTFT, TPOT, throughput, scheduling overhead, token counts, and KV-cache pressure for reproducible systems evaluation.

## Inference & Serving Stack

- **Engine registry**: `inference/setup_inference_engines.py` maps stable engine names to concrete `InferenceEngine` implementations:
  - `baseline`: Hugging Face autoregressive generation path.
  - `vllm`: vLLM library/server execution path with OpenAI-compatible launch support.
  - `preble`: shared-prefix / KV-cache reuse benchmark path.
  - `infercept`: split generation and resumed-decode benchmark path.
  - `mock`: deterministic local/CI engine for reproducible control runs.
- **Common serving contract**: `serving/base.py` defines the engine interface and normalized result schema:
  - prompt tokens, output tokens
  - TTFT and TPOT
  - total request latency
  - KV-cache memory footprint
  - scheduling overhead
  - prefix-cache hit/miss counts
- **Backend implementations**:
  - `serving/baseline_hf.py`: Hugging Face `transformers` baseline for single-stream autoregressive decoding.
  - `serving/vllm_server.py`: vLLM-backed generation and OpenAI-compatible server launch command construction.
  - `serving/preble_benchmark.py`: prefix-sharing benchmark for KV-cache optimized workloads.
  - `serving/infercept_benchmark.py`: prefill/decode split and resumed generation path.
  - `serving/quantization.py`: model/tokenizer loading utilities for quantized inference experiments.
- **Comparability model**:
  - every engine emits the same metric schema;
  - engines can be swapped by name from CLI/API configuration;
  - mock engines preserve benchmark plumbing without GPU or vLLM availability;
  - unsupported native paths fall back to baseline/mock behavior for local execution.
- **Primary measured surfaces**:
  - TTFT: prefill latency / first-token responsiveness;
  - TPOT: decode latency per generated token;
  - throughput: concurrency-sensitive request completion behavior;
  - KV-cache growth: memory pressure as prompt and generated token counts increase;
  - scheduling overhead: runtime/control-plane cost outside model execution;
  - prefix-cache efficiency: cached vs uncached prompt-token accounting.

## Benchmarking & Pipeline Execution

- **CLI entrypoint**: `scripts/benchmark.py`
  - `systems`: runs inference-engine comparisons over synthetic or agent-generated prompts.
  - `profiling`: runs specialized systems profilers.
  - `agents`: runs policy/game benchmarks used as workload generators.
- **Systems benchmark flow**:

```text
CLI/API request
  -> engine registry
  -> selected serving backend(s)
  -> normalized InferenceResult records
  -> SQLite `inference_benchmarks`
  -> dashboard/notebook analysis
```

- **Profiling modules**:
  - `analysis/kv_cache_profiler.py`: KV-cache memory model over sequence length and turn count.
  - `analysis/prefill_decode_split.py`: prefill vs decode latency decomposition.
  - `analysis/throughput_benchmark.py`: concurrency sweep and throughput measurement.
  - `analysis/scheduling_overhead.py`: CPU/runtime scheduling overhead isolation.
- **Evaluation runner**: `evaluation/runner.py`
  - coordinates agent and systems benchmark execution;
  - persists engine metrics and agent outcomes;
  - aggregates run IDs, per-turn measurements, and summary statistics.
- **Recorded systems metrics**:
  - `prompt_tokens`, `output_tokens`
  - `ttft_ms`, `tpot_ms`, `total_latency_ms`
  - `kv_cache_mb`
  - `scheduling_overhead_ms`
  - engine, model, quantization, run ID, and turn index
- **Reproducibility controls**:
  - deterministic mock engines for CI/local validation;
  - bounded FastAPI request schema;
  - seeded agent/environment workloads;
  - shared SQLite persistence across CLI, API, dashboard, and notebooks.

## System Architecture

```text
CLI / FastAPI / Streamlit
        |
        v
Benchmark orchestration (`evaluation/runner.py`)
        |
        +--> Inference engines (`inference/`, `serving/`)
        |     - Hugging Face baseline
        |     - vLLM
        |     - KV-cache optimized variants
        |     - deterministic mocks
        |
        +--> Workload generators (`agents/`, `environments/`)
        |     - modular agent policies
        |     - Gymnasium-style game environments
        |     - turn-level prompts/actions/rewards
        |
        +--> Profilers (`analysis/`)
        |     - KV-cache growth
        |     - TTFT/TPOT split
        |     - throughput/concurrency
        |     - scheduling overhead
        |
        v
SQLite persistence (`data/schemas.py`)
        |
        v
Dashboard / notebooks / reports
```

- **FastAPI backend**: `main.py` exposes `/benchmark`, validates requests with Pydantic, bounds run sizes, and rejects unsafe database paths.
- **Streamlit UI**: `streamlit_app.py` and `dashboard/` visualize inference benchmarks, game workloads, player profiles, and evaluation outputs.
- **Agent framework**: `agents/` provides heuristic, profile-aware, SFT, RL/BC-RL, agentic LLM, checkpoint-backed, and impostor policies.
- **Environment layer**: `environments/` implements Gymnasium-compatible `reset` / `step` workloads with discrete actions and turn-level observations.

## Data Layer

- **Schema owner**: `data/schemas.py`
- **Primary systems table**: `inference_benchmarks`
  - stores per-engine, per-turn latency/token/KV metrics;
  - keyed by run ID, engine, and turn;
  - includes model and quantization metadata.
- **Workload/evaluation tables**:
  - `games`, `rounds`: gameplay traces and turn-level actions;
  - `agent_results`: policy benchmark summaries;
  - `player_profiles`, `impostor_results`, `detection_sessions`: clone/evaluation artifacts;
  - additional report, replay, slice, ladder, and study-block tables for dashboard workflows.
- **Persistence model**:
  - default database: `data/game_data.db`;
  - override: `DOPPELGAMER_DB_PATH`;
  - SQLite foreign keys enabled at connection setup;
  - local DB files and secrets are ignored by default.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```bash
python scripts/benchmark.py systems --engines baseline vllm preble infercept --model mock --rounds 20
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock
python scripts/benchmark.py profiling --type prefill_decode --engine baseline --model mock
```

```bash
uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py
pytest -q
```

Optional native vLLM execution requires a compatible vLLM/GPU environment; local CPU/CI runs can use `--model mock`.
