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
- [GPU / publication](#gpu--publication)
- [Architecture](#architecture)
- [Contributing](#contributing)

## What it does

- Runs inference engines (HuggingFace baseline and vLLM by default) on the same workloads.
- Optional local HF ablations: `hf_prefix_cache`, `hf_tool_interrupt` (not Preble/InferCept).
- Optional remote engines: `preble` / `infercept` only if `PREBLE_BASE_URL` / `INFERCEPT_BASE_URL` point at OpenAI-compatible servers.
- Treats each game as a workload generator. Same seed, same multi-turn trajectory, discrete moves and a reward per turn.
- Trains and evaluates agents: RL (stable-baselines3), SFT (PEFT), LSTM/N-gram impostors, and heuristic baselines.
- Tracks KV-cache growth, TTFT/TPOT, prefix reuse, and throughput (with explicit modes).
- Includes deterministic mock engines for laptop and CI (`--model mock`).
- Saves results to SQLite and shows them in FastAPI and Streamlit.

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

Mock engines share the same timings by design (CI only). They do not rank real backends.

For publishable numbers, use the fixed GPU protocol (fail-loud, baseline vs vLLM, metadata + CSV):

```bash
python scripts/run_publication_benchmark.py --model distilgpt2 --rounds 50 --out results/publication
```

See [docs/brev_setup.md](docs/brev_setup.md) for NVIDIA Brev setup and cost notes. Report **library-mode** methodology; do not claim Preble/InferCept unless you ran remote clusters.

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

GPU install: `pip install -r requirements-gpu.txt` (see Brev doc).

## GPU / publication

```bash
pip install -r requirements-gpu.txt
python scripts/run_publication_benchmark.py \
  --model distilgpt2 \
  --rounds 50 \
  --out results/publication
```

Copy `results/publication/` and `data/publication_run.db` off the instance, then stop the GPU.

Do not use `--allow-fallback` for publication. Real models must fail if CUDA/vLLM is missing.

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

Every engine returns the same metric schema (`serving/base.py`). See [docs/architecture.md](docs/architecture.md) for engines, profilers, and the database schema. See [docs/brev_setup.md](docs/brev_setup.md) for Brev.

## Contributing

Run `pytest -q` before opening a PR. Touching a GPU-backed engine: verify with `--model mock` first. Do not commit secrets or live API keys.
