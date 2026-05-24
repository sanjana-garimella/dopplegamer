# Doppelgamer

![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

> LLM inference benchmarking with game-driven multi-turn workloads.

Most LLM benchmarks are single-turn and stateless. Real agentic systems aren't context grows every step, and that's where serving engines diverge. Doppelgamer uses game environments to generate reproducible multi-turn conversations so you can compare inference engines under realistic pressure.

---

## Contents

- [What it does](#what-it-does)
- [Games](#games)
- [Metrics](#metrics)
- [Results](#results)
- [Stack](#stack)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Contributing](#contributing)

---

## What it does

- Runs **4 inference engines** (HuggingFace, vLLM, Preble, Infercept) against identical workloads. Swap engines with one CLI flag.
- Uses **Gymnasium-style game environments** as workload generators seeded, reproducible, multi-turn conversations with discrete actions and per-turn rewards.
- Trains and evaluates **multiple agent policies**: RL (stable-baselines3), SFT (PEFT), LSTM impostors, and logistic regression baselines.
- Profiles **KV-cache memory growth** across turns the dominant GPU cost in long-context serving.
- Runs fully on CPU via deterministic mock engines. No GPU needed for local dev or CI.
- Persists results to SQLite with a FastAPI + Streamlit dashboard.

---

## Games

| Game | Notes |
|------|-------|
| RPS+ | Rock-paper-scissors with Lizard, Power, Recharge, and energy management |
| Tic-Tac-Toe | Fast 3x3 policy checks |
| Connect Four | 7-column gravity board |
| Chess | Full legal move generation via python-chess |
| Othello | 8x8 Reversi with pass turns and disc-flip rules |
| Checkers | Forced captures, kings, multi-jump sequences |
| Gomoku | Five-in-a-row on a configurable board |
| Nim | Pile-taking math game for turn-level strategy eval |

War is available as a research prototype. Not part of the main arena.

---

## Metrics

| Metric | What it measures |
|--------|-----------------|
| TTFT | Time from request to first output token |
| TPOT | Decode latency per generated token |
| KV-Cache Memory | GPU memory consumed by the attention cache — grows with context length |
| Scheduling Overhead | CPU time outside model execution — bottleneck at high concurrency |
| Prefix Cache Hit Rate | Fraction of prompt tokens served from cache |
| Throughput | Requests per second across concurrency levels |

---

## Results

Mock run, 4 engines, 20 rounds:

| Engine | TTFT (ms) | TPOT (ms) | Throughput (req/s) | KV Cache (MB) |
|--------|-----------|-----------|-------------------|---------------|
| baseline | 142 | 38 | 4.2 | 512 |
| vllm | 61 | 22 | 9.8 | 480 |
| preble | 58 | 21 | 10.3 | 310 |
| infercept | 55 | 20 | 10.7 | 298 |

Preble and Infercept use ~40% less KV-cache than baseline. The gap widens as conversation length increases.

---

## Stack

| Layer | Libraries |
|-------|-----------|
| Inference | HuggingFace Transformers, vLLM |
| Fine-tuning | PEFT |
| RL agents | stable-baselines3 |
| Deep learning | PyTorch |
| Game environments | Gymnasium, python-chess |
| Vector storage | ChromaDB |
| ML baselines | scikit-learn |
| Data | pandas, numpy, SQLite |
| API | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |

---

## Quickstart

**Prerequisites:** Python 3.12+. GPU optional — `--model mock` runs everything on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```bash
# Benchmark across engines (no GPU required)
python scripts/benchmark.py systems --engines baseline vllm preble infercept --model mock --rounds 20

# Profilers
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock
python scripts/benchmark.py profiling --type prefill_decode --engine baseline --model mock

# API + dashboard
uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py

# Tests
pytest -q
```

---

## Architecture

```
CLI / FastAPI / Streamlit
        |
        v
Benchmark orchestration (evaluation/runner.py)
        |
        +--> Inference engines (inference/, serving/)
        |     - HuggingFace baseline, vLLM, Preble, Infercept, mock
        |
        +--> Workload generators (agents/, environments/)
        |     - Agent policies: heuristic, SFT, RL, LLM, impostor
        |     - Gymnasium game environments
        |
        +--> Profilers (analysis/)
        |     - KV-cache growth, TTFT/TPOT, throughput, scheduling overhead
        |
        v
SQLite → Dashboard
```

Every engine emits the same normalized metric schema (`serving/base.py`). See [docs/architecture.md](docs/architecture.md) for engine details, profiler descriptions, and DB schema.

---

## Contributing

Run `pytest -q` before opening a PR. For GPU-backed engine changes, test with `--model mock` first.
