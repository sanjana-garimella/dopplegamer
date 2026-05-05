# Doppelgamer

Doppelgamer is a platform for behavioral game-agent imitation. It records human gameplay, trains player-specific clone models, and evaluates whether clones are distinguishable from the source player in controlled matches.

## Overview

The repository provides:
- gameplay data collection and storage
- clone-model training and evaluation
- interactive dashboard workflows
- API and CLI benchmarking interfaces

## Objectives

The project is organized around three questions:
1. Can a model reproduce player-specific behavior over time?
2. Can players distinguish their clone from other opponents?
3. Which behavioral signals are associated with fidelity and detection outcomes?

## Components

- `dashboard/`: Streamlit application UI (`streamlit_app.py` entrypoint)
- `main.py`: FastAPI service (`/` and `/benchmark`)
- `agents/`: agent implementations (heuristic, learned, profile-aware, clone)
- `environments/`: game environments with a common interface
- `evaluation/`: benchmark runners and metrics
- `analysis/`: profiling and systems analysis scripts
- `data/`: SQLite schemas and data utilities
- `scripts/`: command-line workflows

## Repository Structure

```text
doppelgamer/
├── main.py
├── streamlit_app.py
├── dashboard/
├── agents/
├── environments/
├── evaluation/
├── analysis/
├── data/
├── scripts/
├── serving/
├── inference/
├── configs/
└── tests/
```

## Requirements

- Python 3.12+
- pip

Install dependencies from `requirements.txt`.

## Setup

```bash
git clone https://github.com/sanjana-garimella/doppelgamer.git
cd doppelgamer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

Dashboard:

```bash
streamlit run streamlit_app.py
```

API:

```bash
uvicorn main:app --reload --port 8000
```

## Basic Workflow

1. Open `http://localhost:8501`.
2. Create or select a player profile.
3. Play rounds to generate data.
4. Train a clone model from saved sessions.
5. Evaluate clone behavior in live and analysis views.

## CLI Usage

Agent benchmarks:

```bash
python scripts/benchmark.py agents --rounds 100 --seeds 3 --games RPS+
```

Inference-system benchmarks:

```bash
python scripts/benchmark.py systems --engines baseline vllm preble infercept --model mock --rounds 20
```

Profiling:

```bash
python scripts/benchmark.py profiling --type scheduling --engine baseline --model mock
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock
python scripts/benchmark.py profiling --type prefill_decode --engine baseline --model mock
```

## API Usage

Health check:

```bash
curl http://localhost:8000/
```

Benchmark request:

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "rounds": 100,
    "engines": ["baseline", "vllm", "preble", "infercept"],
    "agents": ["sft", "rl", "bcrl", "agentic"],
    "db_path": "data/game_data.db"
  }'
```

## Data and Persistence

- Default database path: `data/game_data.db`
- Environment override: `DOPPELGAMER_DB_PATH`
- For containers, mount persistent storage for database files

## Testing

```bash
pytest -q
```

## Deployment

Included files:
- `Dockerfile`
- `Procfile`
- `.streamlit/config.toml`
- `.env.example`

Container example:

```bash
docker build -t doppelgamer .
docker run -p 8501:8501 -v $(pwd)/data:/data doppelgamer
```

For Streamlit Community Cloud, use `streamlit_app.py` as entrypoint.

## Security Notes

- Keep secrets in environment variables or local secret files.
- Do not commit `.env` or `.streamlit/secrets.toml`.
- Keep local database files out of version control unless intentionally shared.

## Contributing

1. Create a branch.
2. Add or update tests as needed.
3. Run `pytest -q`.
4. Open a pull request with a technical summary.

## License

MIT License.
# Doppelgamer

Doppelgamer is a research and engineering platform for behavioral game-agent imitation.  
It records human gameplay, trains player-specific clone models, and evaluates whether clones are distinguishable from the source player in controlled live matches.

The repository combines:
- data collection and storage
- clone-model training and evaluation
- interactive gameplay and analysis UI
- API and CLI entrypoints for benchmarking

## Scope

The project focuses on three linked questions:
1. Can a model reproduce player-specific behavior over time?
2. Can players distinguish their own clone from other opponents?
3. Which behavioral signals contribute to clone fidelity and detection outcomes?

## Core Capabilities

- **Gameplay capture**: logs moves and outcomes to SQLite for per-player datasets.
- **Clone training**: supports `ngram`, `lstm`, and mixture-style impostor modeling workflows.
- **Live evaluation**: runs human-vs-agent and human-vs-clone sessions through Streamlit pages.
- **Benchmarking**: measures agent performance and inference-system latency via CLI and API.
- **Research instrumentation**: includes drift tracking, A/B style clone comparisons, and study-oriented tables.

## System Components

- **Dashboard**: Streamlit application (`streamlit_app.py`, `dashboard/`).
- **API**: FastAPI service with benchmark endpoint (`main.py`).
- **Agents**: heuristic, learning-based, and profile-aware agents (`agents/`).
- **Environments**: multiple board/strategy game environments under a common interface (`environments/`).
- **Evaluation and analysis**: benchmark runners and profiling modules (`evaluation/`, `analysis/`).
- **Storage**: SQLite schemas and data utilities (`data/`).

## Repository Layout

```text
doppelgamer/
├── main.py
├── streamlit_app.py
├── dashboard/
├── agents/
├── environments/
├── evaluation/
├── analysis/
├── data/
├── scripts/
├── serving/
├── inference/
├── configs/
└── tests/
```

## Requirements

- Python 3.12+
- pip
- OS: macOS/Linux/WSL recommended

Install dependencies from:
- `requirements.txt` (runtime and common development dependencies)

## Quick Start

```bash
git clone https://github.com/sanjana-garimella/doppelgamer.git
cd doppelgamer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run the dashboard:

```bash
streamlit run streamlit_app.py
```

Run the API:

```bash
uvicorn main:app --reload --port 8000
```

## First Run Workflow

1. Open the dashboard (`http://localhost:8501`).
2. Create or select a player profile.
3. Play rounds in a supported game mode to generate data.
4. Train a clone model from recorded sessions.
5. Evaluate clone behavior in live matches and analysis pages.

## CLI Workflows

Benchmark agents:

```bash
python scripts/benchmark.py agents --rounds 100 --seeds 3 --games RPS+
```

Benchmark inference systems:

```bash
python scripts/benchmark.py systems --engines baseline vllm preble infercept --model mock --rounds 20
```

Run profiling modes:

```bash
python scripts/benchmark.py profiling --type scheduling --engine baseline --model mock
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock
python scripts/benchmark.py profiling --type prefill_decode --engine baseline --model mock
```

## API Usage

Start service:

```bash
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/
```

Benchmark request example:

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "rounds": 100,
    "engines": ["baseline", "vllm", "preble", "infercept"],
    "agents": ["sft", "rl", "bcrl", "agentic"],
    "db_path": "data/game_data.db"
  }'
```

## Data and Persistence

- Default database path: `data/game_data.db`
- Override with environment variable: `DOPPELGAMER_DB_PATH`
- For container deployments, mount persistent storage for `/data` or an equivalent host path.

## Testing

Run the test suite:

```bash
pytest -q
```

Tests are located in `tests/`.

## Deployment Notes

The repository includes:
- `Dockerfile`
- `Procfile`
- `.streamlit/config.toml`
- `.env.example`

Container example:

```bash
docker build -t doppelgamer .
docker run -p 8501:8501 -v $(pwd)/data:/data doppelgamer
```

For Streamlit Community Cloud deployment, use `streamlit_app.py` as the app entrypoint.

## Security Considerations

- Use environment variables for deployment-specific settings.
- Avoid committing secrets; `.streamlit/secrets.toml` and `.env*` should remain local.
- Keep database files out of version control unless intentionally sharing anonymized fixtures.

## Contributing

1. Create a feature branch.
2. Implement changes with tests where applicable.
3. Run `pytest -q`.
4. Open a pull request with a concise technical description.

## License

MIT License.
<div align="center">

# Doppelgamer

### *Can a bot learn to play exactly like you and fool you into thinking it's human?*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white&style=flat-square)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-130%20passing-22c55e?style=flat-square)](tests/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white&style=flat-square)](main.py)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=flat-square)](dashboard/app.py)
[![PyTorch](https://img.shields.io/badge/model-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)](agents/impostor/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](#)

</div>

---

Most game AI projects ask **"can it win?"**

Doppelgamer asks the harder question: **"can it pass as you?"**

The platform records your gameplay, trains a behavioral clone, and evaluates whether you can detect your own clone in a blind live match. This repo combines data collection, model training, human-vs-clone evaluation, and interactive analysis in one end-to-end system.

## Why this repo matters

- **Research focus**: identity-level imitation, not only game performance
- **End-to-end workflow**: gameplay capture -> clone training -> live Turing test -> analysis
- **Runnable application**: Streamlit dashboard + FastAPI backend + SQLite storage
- **Experimental tooling**: drift tracking, controlled rearing, bias interventions, blind study blocks
- **Engineering baseline**: benchmark scripts and automated tests

---

## Fast Start

```bash
git clone https://github.com/sanjana-garimella/doppelgamer.git
cd doppelgamer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`, create a profile, play a few rounds, and train your first clone.

---

## What you can do

```
You play →  every move logged  →  behavioral clone trained
                                          ↓
               Can you tell it's not you? ←  live Turing test
```

### Core capabilities

- Train and compare **N-gram**, **LSTM**, and **mixture** behavioral clones
- Run blind **human-vs-clone detection** sessions with confidence scoring
- Track **behavioral drift** and run **clone-vs-baseline A/B** evaluations
- Analyze **surprisal**, uncertainty, and per-game strategy fingerprints
- Benchmark serving engines for latency and throughput tradeoffs

### Five research questions baked into the platform

| # | Question | Metric |
|---|----------|--------|
| 1 | Can a bot reproduce your exact play style? | KL divergence, TVD, fidelity score |
| 2 | Can humans detect their own clone? | Turing-test detection rate |
| 3 | How many rounds until fidelity stabilizes? | Fidelity vs. training-data size curves |
| 4 | Does the clone inherit your cognitive biases? | Recency bias, win-streak aggression, loss aversion transfer |
| 5 | LLM Impostor vs. LSTM Impostor — which feels more human? | Side-by-side detection rate |

---

## Cognitive Modeling Framing

Doppelgamer is not just trying to build a stronger game bot. The more interesting claim is that a clone can act as a **cognitive model** of a player: a compact system that predicts not only what move someone will make, but how their behavior changes with experience, pressure, and opponent context.

That framing pulls Doppelgamer closer to work in:

- **Surprisal and prediction-based processing**: how expected was the human move under their own clone?
- **Developmental realism / BabyLM-style scaling**: how much behavior can a clone learn from human-scale data, not internet-scale pretraining?
- **Controlled rearing**: what happens if the clone only sees certain slices of experience, like early sessions, losses, or one game family?
- **Mechanistic interpretability**: what habits, contexts, and internal features actually drive human-like predictions?
- **Causal intervention**: if we suppress a learned bias, does fool rate collapse?

In that sense, Doppelgamer treats game logs the way cognitive modeling work treats language corpora: not as raw fuel for performance, but as structured evidence about how minds behave.

---

## Research Roadmap

The current roadmap is organized around five paper-facing ideas borrowed from cognitive modeling and psycholinguistics.

### 1. Surprisal as a Behavioral Metric

For each human move, evaluate the probability assigned by that player's clone and log:

- move surprisal: `-log p(move | context)`
- clone confidence / entropy
- whether high-surprisal turns are also high-detection turns

This makes fidelity more psychologically meaningful than raw agreement. A clone that puts high probability on the human's actual move is behaving like a predictive cognitive model, not just matching aggregate counts.

### 2. Developmental Training Curves

Instead of training on all historical data at once, train on progressively larger prefixes of a player's session history:

- first 2 sessions
- first 5 sessions
- first 10 sessions
- full history

Then evaluate only on later sessions. This is the Doppelgamer version of a developmentally realistic training curve: how quickly does a clone become recognizably "you" from limited human-scale exposure?

### 3. Controlled Rearing Experiments

Use filtered training sets to isolate which experience slices matter:

- only early sessions
- only recent sessions
- only wins
- only losses
- only one game family
- all sessions except one target game family

These experiments test whether clones learn stable style, transient mood, or game-specific tactics. They also create a strong paper narrative around indirect evidence and selective exposure.

### 4. Mechanistic Clone Analysis

For `ngram`, `lstm`, and `mixture` clones, analyze:

- which histories trigger particular moves
- which contexts create uncertainty spikes
- which opponent conditions shift the policy
- which behavioral features align with fooling humans

The goal is not only to say that a clone works, but to explain what internal structure supports human-like behavior.

### 5. Causal Bias Interventions

Once behavioral features are explicit, intervene on them:

- zero out recency bias
- flatten recharge preference
- remove opponent-conditioned context
- downweight old sessions vs. recent sessions

Then measure the effect on fidelity, fool rate, and detection confidence. This is the cleanest path from descriptive behavioral cloning to causal claims about what habits actually matter.

---

## Three Complementary Tracks

Doppelgamer now supports a research workflow that is intentionally split across three complementary tracks:

### 1. Cognitive Track

This track asks whether the clone is a plausible **cognitive model** of the player.

- per-turn move surprisal: `-log p(human_move | context)`
- controlled rearing experiments over named upbringing slices
- bias intervention tests for recency, recharge, and opponent-conditioning
- failure-case and narrative analyses for suspicious vs convincing turns

### 2. Data-Centric Track

This track treats cloning as a **data lifecycle** problem rather than a one-shot training job.

- player dataset cards: rounds, sessions, game coverage, drift coverage, trainability
- persisted session-slice registry for reproducible experiment subsets
- drift-based retraining triggers when recent snapshots move beyond threshold
- one-click paper battery from the Player Profile page

### 3. Systems Track

This track makes Doppelgamer legible as an **ML systems platform** as well as a behavior platform.

- clone fidelity vs latency frontier
- lightweight and quantized-style LSTM serving variants
- runtime-condition studies under interactive, research, and paper-quality budgets
- cost / throughput / quality surfaces across inference engines

---

## Paper Plan

The current paper story should be framed around three linked questions:

1. **Can a clone predict a human's future moves under temporal drift?**
2. **Can a clone fool the human it was trained on in a blinded live Turing test?**
3. **What behavioral mechanisms make the clone convincing or suspicious?**

That maps naturally onto the implemented system:

- **Prediction / generalization**
  - session-ordered generalization sweep
  - canonical baseline battery
  - paired per-player significance tests

- **Detection / human study**
  - blind study blocks
  - standardized confidence prompts
  - human-vs-human confusion baseline
  - power analysis for target sample size

- **Mechanism / interpretation**
  - behavioral drift snapshots
  - per-game strategy fingerprints
  - failure-case gallery
  - narrative figure: *The Clone Learns You*
  - uncertainty-aware clone confidence
  - controlled rearing and bias intervention studies

If the project is presented this way, Doppelgamer reads less like "AI that imitates game moves" and more like a controlled platform for studying **identity, predictability, and behavioral representation**.

---

## Tech Stack

| Layer | What's There |
|-------|-------------|
| **API** | FastAPI + Pydantic — `/benchmark`, profile endpoints |
| **Dashboard** | Streamlit (8 pages) + Plotly — hub, live arena, player profile, clone ops, benchmarks |
| **Clone Models** | N-gram transition model, LSTM sequence model (PyTorch, optional), online adaptation |
| **RL Agents** | PPO, BC+RL warm-start (Stable-Baselines3), supervised fine-tuning |
| **LLM Agent** | ReAct loop on LangGraph — Reason → Act → Observe → Reflect each turn |
| **Inference Engines** | HuggingFace, vLLM, Preble, InferCept — benchmarked live |
| **Environments** | 22 games, all Gymnasium-compatible |
| **Storage** | SQLite — foreign keys enforced, parameterized queries, indexed hot paths |
| **Tests** | 130 automated tests, all green |

---

## The Clone Models

Two models trained on your recorded move history:

### NGramImpostor
Builds a transition table — given your last N moves, what do you play next? Fast, interpretable, trains in milliseconds. Backsoff to unigram when context is unseen.

```python
agent = NGramImpostor(n=2)
agent.train(your_move_sequences)

# What would you play next, given this context?
probs = agent.predict_proba(history=[ROCK, PAPER, ROCK])
next_move = agent.select_action(history=[ROCK, PAPER, ROCK])
score = agent.fidelity_score(held_out_sequences)   # 1 - TVD
```

### LSTMImpostor
Sequence model with learned embeddings. Captures longer-range behavioral patterns — win streaks, tilt spirals, momentum shifts. PyTorch optional; falls back to uniform random if unavailable.

```python
agent = LSTMImpostor(hidden_size=64, num_layers=2)
stats = agent.train(sequences, epochs=20)
# stats → {"final_loss": 0.83, "loss_curve": [...]}

agent.save("checkpoints/impostor/player_xyz_lstm.pt")
```

**Both models expose the same interface:** `predict_proba()` → distribution, `predict()` → sampled move, `select_action()` → alias, `fidelity_score()` → scalar.

Both clone models now also support:
- lightweight **sequence-aware explanations** for why they predicted a move
- optional **cross-game player embeddings** as conditioning context
- **online adaptation** after new human sessions are saved
- **uncertainty / entropy estimates** for suspicious-turn analysis

The training stack now also includes:
- a **mixture-of-habits** clone that blends N-gram, LSTM, and heuristic priors
- **opponent-conditioned** training context so imitation can vary by matchup
- **recency-weighted adaptation** to track behavioral drift over time
- **lightweight / quantized-style LSTM variants** for serving-time tradeoff studies

---

## Agents

Nine agent types spanning the full capability spectrum:

| Agent | Type | Notes |
|-------|------|-------|
| `random` | Baseline | Uniform over legal moves |
| `heuristic` | Rule-based | Counter-frequency + energy awareness |
| `optimal` | Rule-based | Perfect counter to last opponent move |
| `sft` | Neural | Supervised fine-tuning on human gameplay logs |
| `ppo` | Neural | PPO — n_steps=2048, ent_coef=0.01 |
| `bc_rl` | Neural | BC warm-start → PPO fine-tuning |
| `agentic` | LLM | ReAct / Crew / ADK with tool use (analyze_opponent, predict_next_move) |
| `profile_counter` | Adaptive | Reads full DB history per player; predicts and counters next move |
| `adaptive_router` | Meta | Switches between experts live based on rolling win rate |
| `ngram` | Clone | Behavioral clone trained from saved player history |
| `lstm` | Clone | Sequence-model clone trained from saved player history |

The `profile_counter` and `adaptive_router` agents use real per-session DB state — they're not stateless. They get worse opponents that adapt.

---

## Environments

**Strategy Classics** — full rule fidelity, used in Live Arena:

| Game | Action Space | Notes |
|------|-------------|-------|
| RPS+ | Discrete(6) | Energy mechanics — POWER and RECHARGE moves |
| Chess | Discrete(N legal) | python-chess backend, illegal move rejection |
| Othello | Discrete(64) | Flip logic, pass handling |
| Connect Four | Discrete(7) | Gravity, 4-direction win detection |
| Checkers | Discrete(4096) | Forced jumps, multi-jump, king promotion |
| Tic-Tac-Toe | Discrete(9) | Standard rules |

**Research Games** — 16 compact behavioral sims:
War · Nim · Gomoku · 2048 · Wordle · Sudoku · Pac-Man · Candy Crush · Minecraft · Among Us · Clash Royale · Flappy Bird · Ludo · UNO · Scrabble · Monopoly

All 22 implement the same interface: `reset()`, `step()`, `render()`, `legal_moves()`, `legal_actions()`.

Live Arena now layers on top of these envs with:
- **best-of-N match series**
- **Draft Mode Clone Ladder** for escalating RPS+ clone gauntlets
- **per-game hint engine**
- **pause / undo / restart / play again controls**
- **opening / trap challenges** for Chess, Checkers, Connect Four, and Gomoku
- **friend-vs-clone** sessions using another saved player's clone source
- **blind Turing-study blocks** with standardized confidence prompts and hidden opponent labels

---

## Inference Benchmarking

Four serving engines, benchmarked on real gameplay latency:

```bash
python scripts/benchmark.py systems --rounds 20 --model mock
```

Metrics captured per-turn: **TTFT** (time-to-first-token), **TPOT** (time-per-output-token), **KV cache growth**, **scheduling overhead**, **prefix cache hit rate**.

Profiling sweeps available:

```bash
python scripts/benchmark.py profiling --type throughput --model mock
python scripts/benchmark.py profiling --type prefill_decode --model mock
```

All stored to SQLite → `inference_benchmarks` table → Plotly charts in dashboard.

---

## Data Model

**Core gameplay tables** (stable bootstrap schema):

```sql
games               -- game_id, agent_name, game_type, scores, timestamps
rounds              -- per-turn moves, outcome, energy state   [indexed on game_id]
player_profiles     -- behavioral_signature_json, win_rate, games_played
agent_results       -- win_rate, fidelity, action_kl, latency per evaluation run
inference_benchmarks -- ttft, tpot, kv_cache, scheduling_overhead per turn
impostor_results    -- fidelity_score, kl_divergence, tvd per trained clone
detection_sessions  -- Turing-test outcomes: detected_as_human, confidence, game_id
```

Every `behavioral_signature_json` is a live-blended fingerprint: move distribution, aggression rate, recharge rate, counter-move rate. New game data is weighted in proportionally.

**Extended research tables** (initialized when clone research pages run):

```sql
behavioral_snapshots -- time-series signature snapshots for drift tracking
clone_ab_runs        -- controlled clone-vs-baseline evaluation blocks
counterfactual_replays -- replay summaries against alternate opponents
shareable_reports    -- persisted one-page clone summaries
clone_ladder_runs    -- rung-by-rung results for Draft Mode Clone Ladder
blind_study_blocks   -- randomized hidden-condition schedules for blinded studies
dataset_slices       -- named, reproducible session subsets for controlled rearing
```

---

## Architecture

```
                      ┌─────────────┐
                      │   FastAPI   │  /benchmark  /profiles
                      └──────┬──────┘
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
    ┌──────▼──────┐                   ┌────────▼────────┐
    │  Evaluation │                   │ Inference Layer  │
    │   Runner    │                   │  4 engines       │
    └──────┬──────┘                   └────────┬─────────┘
           │                                   │
    ┌──────▼──────┐                   ┌────────▼─────────┐
    │   9 Agents  │                   │  Serving          │
    │  (registry) │                   │  HF · vLLM        │
    └──────┬──────┘                   │  Preble · InferCept│
           │                          └──────────────────┘
    ┌──────▼──────┐
    │  22 Envs    │   SQLite ← data/schemas.py
    │  (Gym API)  │   7 tables, FK enforced
    └─────────────┘

           Impostor Pipeline
           ─────────────────
           Collector → SQLite → ImpostorTrainer
                                      ↓
            NGramImpostor  LSTMImpostor  MixtureImpostor
                                      ↓
           fidelity_score · uncertainty · explanations · embeddings
                                      ↓
    A/B blocks · drift snapshots · counterfactual replay · blind study blocks · detection session
```

---

## Quickstart

```bash
git clone https://github.com/sanjana-garimella/doppelgamer.git
cd doppelgamer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

pytest                              # 130 tests, all green
streamlit run streamlit_app.py      # game hub on localhost:8501
```

Minimum runtime: Python 3.12+.

**Dashboard flow:**
1. Create a player profile, or log in by typing an existing profile name / subject ID
2. Play **Train Clone Match** — every move is logged to SQLite
3. Use **Live Arena** controls: best-of-N series, hints, replay, undo, and opening challenges
4. Go to **Player Profile → Train My Clone**
5. Review **behavioral timeline**, **drift**, **per-game fingerprint**, and **clone fidelity**
6. Open **Clone Leaderboard** for fool rate, Turing-test ops, and controlled A/B blocks

```bash
# Collect data headlessly
python -m data.collector --game "RPS+" --rounds 500

# Benchmark agents
python scripts/benchmark.py agents --games "RPS+" Chess Othello --rounds 100 --seeds 3

# Import external data
python scripts/import_external_games.py chess-pgn --pgn lichess.pgn --max-games 10000
```

### Deploy

For a simple container deployment:

```bash
docker build -t doppelgamer .
docker run -p 8501:8501 -v $(pwd)/data:/data doppelgamer
```

For PaaS-style deployment, the repo now includes:
- `Dockerfile`
- `Procfile`
- `.streamlit/config.toml`
- `.env.example`

Environment variables:

```bash
export DOPPELGAMER_DB_PATH=/absolute/path/to/game_data.db
export DOPPELGAMER_PUBLIC_BASE_URL=https://your-app.example.com
```

Persistence note:
- By default, local development uses `data/game_data.db`
- In Docker, the recommended persistent path is `/data/game_data.db`
- Mount a host volume or managed disk so profiles, matches, clone reports, and study logs survive restarts

Recommended pre-public deployment checklist:
1. Set up persistent storage for `data/game_data.db`
2. Review privacy / retention language for real users
3. Run `pytest -q`
4. Smoke-test `Live Arena`, `Train My Clone`, and `Clone Leaderboard` on the deployed URL

### Free deployment: Streamlit Community Cloud

Streamlit Community Cloud is the cleanest free option for this project’s dashboard. It is officially described by Streamlit as a free platform for personal, educational, and non-commercial apps, and deployment is done from a GitHub repository with a Streamlit entrypoint file.[Streamlit Community Cloud overview](https://docs.streamlit.io/deploy/streamlit-community-cloud) [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

This repo is prepared for that flow with:
- `streamlit_app.py` at the repo root
- `requirements.txt`
- `.streamlit/config.toml`

Deployment steps:
1. Create a GitHub repository and upload this project
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click **Create app**
4. Select your GitHub repo and branch
5. Set the entrypoint file to `streamlit_app.py`
6. In Advanced settings, optionally set secrets / env config if you need a non-default DB path
7. Deploy

Important limitation for free Streamlit hosting:
- local SQLite data does **not** automatically persist unless you provide persistent storage outside the ephemeral app container
- for a lightweight demo, this is fine
- for real user studies, use managed persistence or a mounted volume on a different host

---

## Security

| Concern | Approach |
|---------|----------|
| SQL injection | Table-name allowlist + parameterized queries throughout |
| Path traversal | `main.py` rejects `..` and absolute paths in `db_path` |
| Foreign keys | `PRAGMA foreign_keys = ON` in every connection |
| Profile isolation | Every query scoped to `agent_name = ?` |
| Page gating | Gameplay and research pages are blocked until a profile is selected |

---

## Project Structure

```
doppelgamer/
├── main.py                      FastAPI entrypoint
├── dashboard/
│   ├── app.py                   Hub — live arena, clone ops, research snapshot
│   └── pages/
│       ├── live_game.py         Live Arena — controls, hints, challenges, clone play
│       ├── player_profile.py    Profile analytics, clone training, drift, reports
│       └── impostor_leaderboard.py  Clone ops console + fool-rate leaderboards
├── agents/
│   ├── adaptive_router.py       Meta-agent: expert switching on live win rate
│   ├── profile_counter.py       History-aware per-player counter agent
│   ├── rl/                      PPO + BC+RL (Stable-Baselines3)
│   ├── sft/                     Supervised fine-tuning
│   ├── agentic/                 LLM agents — ReAct, Crew, ADK
│   └── impostor/                NGramImpostor + LSTMImpostor + trainer
├── environments/                22 Gymnasium-compatible game implementations
├── serving/                     HF · vLLM · Preble · InferCept engines
├── evaluation/                  Benchmark runner, metrics, W&B logger
├── analysis/                    TTFT/TPOT profiling, KV cache, throughput sweeps
├── impostor/                    5-experiment suite, player profiles, detection metrics
├── data/                        SQLite schemas, collector, importers, backfill
├── scripts/                     CLI — benchmark, import, train, profile
├── tests/                       130 automated tests
└── configs/                     YAML — agents, serving, experiments
```

---

## Status

| Track | Status |
|-------|--------|
| Game environments (22) | Complete |
| Agent registry (9 types) | Complete |
| Clone pipeline (NGram + LSTM + online adaptation) | Complete |
| Impostor experiments (5 designs) | Infrastructure complete, live data collection in progress |
| Behavioral drift / A/B / counterfactual research tooling | Complete |
| Shareable reports / friend-vs-clone loop | Complete |
| Inference benchmarking (4 engines) | Complete |
| Live Streamlit dashboard | Complete |
| FastAPI serving layer | Complete |
| 130 automated tests | All passing |

---

<div align="center">

Built by [Sanjana Garimella](https://github.com/sanjana-garimella)

*The best Impostor doesn't just win, it makes you doubt yourself.*

</div>
