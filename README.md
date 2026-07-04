# Doppelgamer

![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

> LLM inference benchmarking with game-driven, multi-turn workloads.

Doppelgamer benchmarks LLM inference engines using games as the workload. The model plays a game turn by turn, building a long, multi-turn conversation with growing context. Games are seeded, so a run is reproducible, and every engine reports the same speed and memory metrics.

## Contents

- [What it does](#what-it-does)
- [Quickstart](#quickstart)
- [Games](#games)
- [Metrics](#metrics)
- [Results](#results)
- [GPU benchmarks](#gpu-benchmarks)
- [Architecture](#architecture)
- [Stack](#stack)
- [Contributing](#contributing)

## What it does

- Runs inference engines (HuggingFace baseline and vLLM by default) on the same seeded workloads.
- Trains and evaluates agents: RL (stable-baselines3), SFT (PEFT), LSTM/N-gram impostors, and heuristic baselines.
- Tracks KV-cache growth, TTFT/TPOT, prefix reuse, and throughput.
- Ships deterministic mock engines for laptop and CI (`--model mock`).
- Saves results to SQLite and serves them via FastAPI and Streamlit.

Optional engines: remote `preble` / `infercept` (external research systems, not reimplemented here) when `PREBLE_BASE_URL` / `INFERCEPT_BASE_URL` point at a live OpenAI-compatible server, plus local HF ablations `hf_prefix_cache` and `hf_tool_interrupt` that approximate those ideas.

## Quickstart

Python 3.12 or newer. GPU optional; `--model mock` runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```bash
# Systems (mock, no GPU)
python scripts/benchmark.py systems --engines baseline vllm --model mock --rounds 20

# Agents (rounds = games; RPS+ uses 20 turns per game)
python scripts/benchmark.py agents --rounds 50

# Profilers
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock
python scripts/benchmark.py profiling --type prefill_decode --engine baseline --model mock
python scripts/benchmark.py profiling --type scheduling --engine baseline --model mock

# API and dashboard
uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py

# Tests
pytest -q
```

GPU install: `pip install -r requirements-gpu.txt`. See [GPU benchmarks](#gpu-benchmarks) below and [docs/brev_setup.md](docs/brev_setup.md) for NVIDIA Brev setup and cost notes.

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
| Total latency | End-to-end request time (`total_latency_ms`) |
| KV-cache memory | Attention cache size (engine-reported or formula) |
| Prefix cache hit rate | Fraction of prompt tokens served from a shared prefix |
| Throughput | Requests per second; modes: `sequential`, `engine_batch` (vLLM), `threaded_clients` (remote) |
| Host wait | `wall - cpu` around generate (includes GPU wait; not serving-scheduler time) |

## Results

Mock engines share the same timings by design (CI only) and do not rank real backends. For reliable numbers, use the fail-loud GPU protocol in [GPU benchmarks](#gpu-benchmarks): a real model errors out instead of silently falling back, and baseline is compared against vLLM with full metadata and CSV output.

### Validation run: distilgpt2 on an L40S

Single seed, 50 rounds, library mode (the engine's Python API called in-process, not behind an HTTP server). Hardware: NVIDIA L40S, vLLM 0.24.0.

| Engine | Mean TTFT | Mean TPOT | Mean total latency |
|--------|-----------|-----------|---------------------|
| baseline (HF) | 3.63 ms | 2.86 ms | 19.79 ms |
| vllm | 2.06 ms | 2.06 ms | 12.35 ms |

At concurrency 1, vLLM's `engine_batch` mode reaches 114.7 QPS versus 78.9 QPS for the HF baseline's `sequential` mode, and scales to 593.5 QPS at concurrency 8. See [docs/results.md](docs/results.md) for the full throughput sweep and host-wait analysis.

`distilgpt2` (82M params) validates the methodology, not the systems claim. The intended main result is a 1B-plus model (for example `meta-llama/Llama-3.2-1B`) run with the same protocol.

## GPU benchmarks

```bash
pip install -r requirements-gpu.txt
python scripts/run_publication_benchmark.py \
  --model distilgpt2 \
  --rounds 50 \
  --out results/publication
```

Copy `results/publication/` and `data/publication_run.db` off the instance, then stop the GPU. See [docs/brev_setup.md](docs/brev_setup.md) for NVIDIA Brev setup, cost notes, and troubleshooting.

Do not use `--allow-fallback` for these runs. Real models must fail if CUDA/vLLM is missing.

Optional HF ablations on the same box:

```bash
python scripts/benchmark.py systems \
  --engines baseline vllm hf_prefix_cache hf_tool_interrupt \
  --model distilgpt2 --rounds 20
```

## Architecture

```
CLI / FastAPI / Streamlit
        |
        v
Benchmark orchestration (evaluation/runner.py)
        |
        +--> Inference engines (inference/, serving/)
        |     - baseline, vllm, hf_* ablations, remote preble/infercept, mock
        |
        +--> Workload generators (agents/, environments/)
        |     - Agent policies: heuristic, SFT, RL, impostor
        |     - Gymnasium game environments
        |
        +--> Profilers (analysis/)
        |     - KV-cache, TTFT/TPOT, throughput modes, host wait
        |
        v
SQLite -> Dashboard
```

Every engine returns the same metric schema (`serving/base.py`). See [docs/architecture.md](docs/architecture.md) for engines, profilers, and the database schema.

## Stack

| Layer | Libraries |
|-------|-----------|
| Inference | HuggingFace Transformers, vLLM |
| Fine-tuning | PEFT |
| RL agents | stable-baselines3 |
| Deep learning | PyTorch |
| Game environments | Gymnasium, python-chess |
| Optional agentic RAG | ChromaDB |
| Data | pandas, numpy, SQLite |
| API | FastAPI, Pydantic, uvicorn |
| Dashboard | Streamlit, Plotly |

## Contributing

Run `pytest -q` before opening a PR. Touching a GPU-backed engine: verify with `--model mock` first. Do not commit secrets or live API keys.
