# Doppelgamer

![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

> LLM inference benchmarking with game-driven, multi-turn workloads.

Doppelgamer is a benchmarking setup for LLM inference engines. It uses games as the workload: the model plays through a game turn by turn, which creates a long, multi-turn conversation with a growing context. Each game is seeded, so the same run produces the same conversation, and the engines all report the same metrics for speed and memory.

## Contents

- [What it does](#what-it-does)
- [Games](#games)
- [Metrics](#metrics)
- [Results](#results)
- [Stack](#stack)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Contributing](#contributing)

## What it does

- Runs four inference engines (HuggingFace, vLLM, Preble, Infercept) on the same workloads. Switching between them is one CLI flag.
- Treats each game as a workload generator. Give it a seed and you get back the same multi-turn conversation every time, with discrete moves and a reward per turn.
- Trains and evaluates a few kinds of agents: RL (stable-baselines3), SFT (PEFT), LSTM impostors, and logistic regression baselines.
- Tracks how the KV cache grows turn over turn, since that memory is usually what limits long-context serving.
- Includes deterministic mock engines, so everything runs on a laptop with no GPU. CI uses the same path.
- Saves results to SQLite and shows them through a FastAPI backend and a Streamlit dashboard.

## Games

| Game | Notes |
|------|-------|
| RPS+ | Rock-paper-scissors with Lizard, Power, Recharge, and energy management |
| Tic-Tac-Toe | Fast 3x3 policy checks |
| Connect Four | 7-column gravity board |
| Chess | Full legal move generation via python-chess |
| Othello | 8x8 Reversi with pass turns and disc-flip rules |
| Checkers | Forced captures, kings, and multi-jump sequences |
| Gomoku | Five-in-a-row on a configurable board |
| Nim | Pile-taking math game for turn-level strategy evaluation |

War is in the repo as a research prototype, not part of the main arena.

## Metrics

| Metric | What it measures |
|--------|-----------------|
| TTFT | Time from request to first output token |
| TPOT | Decode latency per generated token |
| KV-Cache Memory | GPU memory held by the attention cache. Grows with context length |
| Scheduling Overhead | CPU time spent outside model execution. Becomes the bottleneck at high concurrency |
| Prefix Cache Hit Rate | Fraction of prompt tokens served from cache |
| Throughput | Requests per second across concurrency levels |

## Results

Mock run, four engines, 20 rounds:

| Engine | TTFT (ms) | TPOT (ms) | Throughput (req/s) | KV Cache (MB) |
|--------|-----------|-----------|-------------------|---------------|
| baseline | 142 | 38 | 4.2 | 512 |
| vllm | 61 | 22 | 9.8 | 480 |
| preble | 58 | 21 | 10.3 | 310 |
| infercept | 55 | 20 | 10.7 | 298 |

Preble and Infercept use roughly 40% less KV-cache than the baseline, and the gap grows as conversations get longer.

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

## Quickstart

You need Python 3.12 or newer. A GPU is optional, since `--model mock` runs everything on CPU.

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

# API and dashboard
uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py

# Tests
pytest -q
```

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
SQLite -> Dashboard
```

Every engine returns the same metric schema (`serving/base.py`), so adding a new backend means writing one class and nothing else has to change. See [docs/architecture.md](docs/architecture.md) for engine details, profiler descriptions, and the database schema.

## Contributing

Run `pytest -q` before opening a PR. If you're touching a GPU-backed engine, run it with `--model mock` first so people without a GPU can still run the tests.
