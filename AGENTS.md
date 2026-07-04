# Agent guide for Doppelgamer

Doppelgamer benchmarks LLM inference engines with game-driven, multi-turn workloads. Games generate seeded conversations; engines report the same latency and memory metrics. Agents (RL, SFT, impostor, heuristic) play those games. Results land in SQLite and show up in FastAPI + Streamlit.

Python 3.12+. GPU is optional; `--model mock` runs everything on CPU.

## Directory map

| Path | Role |
|------|------|
| `environments/` | Gymnasium-style games (`reset` / `step` / `legal_moves`) |
| `agents/` | Policies: base, RL, SFT, agentic, impostor, adaptive router |
| `serving/` | Inference engines (HF baseline, vLLM, HF ablations, remote clients) |
| `inference/` | Engine factory (`setup_inference_engines`) and quantization helpers |
| `evaluation/` | Benchmark runner (`runner.py`) |
| `analysis/` | KV-cache, throughput, prefill/decode, host-wait profilers |
| `data/` | SQLite schema, collector, features, importers |
| `dashboard/` | Streamlit UI |
| `impostor/` | Player profiles and impostor-effect metrics |
| `scripts/` | CLI (`benchmark.py`, `run_publication_benchmark.py`, export, training) |
| `tests/` | pytest suite (CPU / mock only) |
| `docs/` | Architecture and Brev setup |

## Key commands

```bash
# Tests (run after any code change)
pytest -q

# Benchmarks (mock = no GPU)
python scripts/benchmark.py agents --rounds 50
python scripts/benchmark.py systems --engines baseline vllm --model mock --rounds 20
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock

# Publication protocol (GPU, fail-loud)
python scripts/run_publication_benchmark.py --model distilgpt2 --rounds 50 --out results/publication

# API and dashboard
uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py
```

Results go to `data/game_data.db` (override with `DOPPELGAMER_DB_PATH`). Publication script defaults to `data/publication_run.db`.

## Conventions

1. **Mock-first.** Develop and CI with `--model mock`. Real models fail loud unless `allow_fallback=True` / `--allow-fallback`. Only requested engines are constructed (lazy load).
2. **Primary engines:** `baseline`, `vllm`. **HF ablations:** `hf_prefix_cache`, `hf_tool_interrupt` (local only; not Preble/InferCept). **Remote:** `preble` / `infercept` require `PREBLE_BASE_URL` / `INFERCEPT_BASE_URL`.
3. **Canonical RPS+ counters.** `BEST_COUNTER` lives only in `environments/rps_plus.py`. Import it; never map POWER to RECHARGE.
4. **Engine metric schema.** Every engine returns `InferenceResult` from `serving/base.py`. Fallback rows are labeled `requested→actual`.
5. **Agent contract.** Subclass `agents.base.Agent`, implement `act(obs, info)` (`legal_moves` required), optional `reset` / `observe`. Register with `register_agent` in `agents/__init__.py`. Default battery excludes untrained `ngram` / `lstm`.
6. **Env contract.** `reset(seed=...)`, `step(action)`, `legal_moves()`, info with `legal_moves`. Same seed, same trajectory. RPS+ raises on illegal actions (runner clamps before step).
7. **Agent metrics.** `rounds` is games, not turns. RPS+ uses 20 turns per game. Fidelity references `sft` if present else `heuristic`.
8. **HTTP API.** `/benchmark` allowlists `model_name`, requires `DOPPELGAMER_API_KEY` for non-mock models, does not echo loader errors.
9. **Profilers.** Throughput modes: `sequential` / `engine_batch` / `threaded_clients`. Host wait is wall−cpu, not serving-scheduler time.
10. **Do not commit or push** unless the user asks.

## Further reading

- [docs/architecture.md](docs/architecture.md): engines, profilers, DB schema
- [docs/brev_setup.md](docs/brev_setup.md): GPU instance setup and cost notes
- `.cursor/memory/`: project learnings and decisions
- `.cursor/skills/`: workflow skills for benchmarks, new envs/agents/engines, debugging
