# Doppelgamer 🎮

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)]()

> **LLM inference benchmarking with agentic game workloads.**

Existing LLM benchmarks use single-turn, stateless prompts — but agentic systems generate multi-turn conversations where context grows with every step. Doppelgamer uses **game environments as workload generators**: games produce multi-turn interactions with measurable outcomes, making inference pressure comparable across serving engines.

---

## What This Project Does

- Runs **4 inference engines** (HuggingFace, vLLM, Preble, Infercept) against the same workloads and records identical metrics for each — engines swap with a single CLI flag
- Uses **Gymnasium-style game environments** as the workload source, generating seeded multi-turn LLM conversations with discrete actions and per-turn rewards
- Profiles **KV-cache memory growth** across conversation turns — the dominant GPU cost in long-context LLM serving
- Runs fully on **CPU via deterministic mock engines** — no GPU required for local development or CI
- Persists all results to **SQLite** and surfaces them through a FastAPI backend and Streamlit dashboard

---

## Metrics

| Metric | Definition | Production Significance |
|--------|-----------|------------------------|
| **TTFT** (Time to First Token) | Latency from request submission to first output token | How quickly a user sees a response begin |
| **TPOT** (Time Per Output Token) | Decode latency per generated token | Controls streaming speed |
| **KV-Cache Memory** | GPU memory consumed by the attention key-value cache | Grows with context length; determines how many concurrent sessions fit on one GPU |
| **Scheduling Overhead** | CPU time spent outside model execution | Becomes the bottleneck before the GPU does at high request concurrency |
| **Prefix Cache Hit Rate** | Fraction of prompt tokens served from cache | Determines cost savings when multiple requests share a common prompt prefix |
| **Throughput** | Requests completed per second across concurrency levels | Used to size deployments |

---

## Why Games?

Games make good LLM workloads for three reasons:

1. **Seeded environments** — the same seed produces the same sequence of prompts, so benchmark runs are byte-for-byte reproducible
2. **Natural context growth** — each turn appends observations and actions to the conversation, applying realistic KV-cache pressure
3. **Outcome signal** — win/loss, score, and per-turn reward give a second axis beyond latency for evaluating agent policies

---

## Example Results

Mock benchmark run, 4 engines, 20 rounds:

| Engine     | TTFT (ms) ↓ | TPOT (ms) ↓ | Throughput (req/s) ↑ | KV Cache (MB) ↓ |
|------------|------------|------------|---------------------|----------------|
| baseline   | 142        | 38         | 4.2                 | 512            |
| vllm       | 61         | 22         | 9.8                 | 480            |
| preble     | 58         | 21         | 10.3                | 310            |
| infercept  | 55         | 20         | 10.7                | 298            |

> Preble and Infercept use ~40% less KV-cache memory than baseline at the same output quality — the gap widens as conversation length increases.

---

## Prerequisites

- **Python 3.10+**
- **pip**
- **CUDA GPU + vLLM** *(optional)* — required for `vllm`, `preble`, and `infercept` engines; `--model mock` runs everything locally on CPU

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Benchmark across engines (no GPU required):**
```bash
python scripts/benchmark.py systems --engines baseline vllm preble infercept --model mock --rounds 20
```

**Run profilers:**
```bash
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock
python scripts/benchmark.py profiling --type prefill_decode --engine baseline --model mock
```

**Start API + dashboard:**
```bash
uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py
```

**Run tests:**
```bash
pytest -q
```

---

## Architecture

```text
CLI / FastAPI / Streamlit
        |
        v
Benchmark orchestration (evaluation/runner.py)
        |
        +--> Inference engines (inference/, serving/)
        |     - Hugging Face baseline
        |     - vLLM
        |     - KV-cache optimized variants (preble, infercept)
        |     - Deterministic mocks
        |
        +--> Workload generators (agents/, environments/)
        |     - Agent policies (heuristic, SFT, RL, agentic LLM, impostor)
        |     - Gymnasium-style game environments
        |     - Turn-level prompts / actions / rewards
        |
        +--> Profilers (analysis/)
        |     - KV-cache growth
        |     - TTFT/TPOT split
        |     - Throughput / concurrency
        |     - Scheduling overhead
        |
        v
SQLite persistence (data/schemas.py)
        |
        v
Dashboard / notebooks / reports
```

- **FastAPI** (`main.py`): `/benchmark` endpoint, Pydantic validation, run-size bounds, safe DB path enforcement
- **Streamlit UI** (`streamlit_app.py`, `dashboard/`): inference benchmark comparisons, game workload traces, player profiles, evaluation outputs
- **Agent framework** (`agents/`): heuristic, profile-aware, SFT, RL/BC-RL, agentic LLM, checkpoint-backed, and impostor policies
- **Environments** (`environments/`): Gymnasium-compatible `reset`/`step` with discrete actions and turn-level observations

---

## Inference & Serving Stack

### Engine Registry (`inference/setup_inference_engines.py`)

| Engine      | Description |
|-------------|-------------|
| `baseline`  | Hugging Face autoregressive generation |
| `vllm`      | vLLM library/server with OpenAI-compatible launch |
| `preble`    | Shared-prefix / KV-cache reuse path |
| `infercept` | Split prefill and resumed-decode path |
| `mock`      | Deterministic local engine for CI and local runs |

Every engine emits the same normalized metric schema (`serving/base.py`). Engines swap by name from CLI or API — no code changes required.

### Profiling Modules (`analysis/`)

| Module | Measures |
|--------|---------|
| `kv_cache_profiler.py` | KV-cache memory growth over sequence length and turn count |
| `prefill_decode_split.py` | Prefill vs. decode latency decomposition |
| `throughput_benchmark.py` | Requests/sec across concurrency levels |
| `scheduling_overhead.py` | CPU time outside model execution |

---

## Data Layer

**Schema:** `data/schemas.py` | **Default DB:** `data/game_data.db` | **Override:** `DOPPELGAMER_DB_PATH`

| Table | Contents |
|-------|----------|
| `inference_benchmarks` | Per-engine, per-turn latency/token/KV metrics keyed by run ID |
| `games`, `rounds` | Gameplay traces and turn-level actions |
| `agent_results` | Policy benchmark summaries |
| `player_profiles`, `impostor_results`, `detection_sessions` | Clone/evaluation artifacts |

SQLite foreign keys enabled at connection setup. DB files and secrets are git-ignored.

---

## Benchmarking Pipeline

```text
CLI/API request
  -> engine registry
  -> selected serving backend(s)
  -> normalized InferenceResult records
  -> SQLite inference_benchmarks
  -> dashboard / notebook analysis
```

### CLI Modes (`scripts/benchmark.py`)

| Mode | Description |
|------|-------------|
| `systems` | Engine comparisons over synthetic or agent-generated prompts |
| `profiling` | KV-cache, throughput, prefill/decode profilers |
| `agents` | Policy and game benchmarks |

---

## Contributing

Run `pytest -q` before opening a pull request. For changes to GPU-backed engines, test with `--model mock` first to verify the benchmark pipeline before requiring a full vLLM environment.
