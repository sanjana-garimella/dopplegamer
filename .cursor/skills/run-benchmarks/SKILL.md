---
name: run-benchmarks
description: >-
  Run agent, systems, and profiling benchmarks via scripts/benchmark.py. Use
  when the user asks to benchmark engines, agents, throughput, TTFT, TPOT,
  prefill/decode, or scheduling overhead, or mentions mock vs real models.
---

# Run benchmarks

CLI entrypoint: `python scripts/benchmark.py`.

## Agents (policy win rates)

```bash
python scripts/benchmark.py agents --rounds 50 --seeds 3 --games RPS+
# optional: --agents random heuristic rl --db data/game_data.db
```

Uses `evaluation/runner.py`. `--rounds` is games (RPS+ uses 20 turns each). No GPU required for non-LLM agents.

## Systems (inference engines)

```bash
# Local / CI (deterministic mock engines)
python scripts/benchmark.py systems --engines baseline vllm --model mock --rounds 20

# Real model (GPU for vLLM); fails loud if deps missing
python scripts/benchmark.py systems --engines baseline vllm --model distilgpt2 --rounds 50

# Optional HF ablations (not Preble/InferCept)
python scripts/benchmark.py systems --engines baseline hf_prefix_cache --model mock --rounds 10

# Publication protocol (preferred for papers)
python scripts/run_publication_benchmark.py --model distilgpt2 --rounds 50 --out results/publication

# Allow silent fallback only for local experimentation
python scripts/benchmark.py systems --engines vllm --model distilgpt2 --allow-fallback
```

`model_name == "mock"` forces `_MockEngine` for requested names. Real models raise
`EngineLoadError` unless `--allow-fallback`. Only requested engines are constructed.

`preble` / `infercept` need `PREBLE_BASE_URL` / `INFERCEPT_BASE_URL` (remote OpenAI-compatible).

## Profiling

```bash
python scripts/benchmark.py profiling --type throughput --engine baseline --model mock
python scripts/benchmark.py profiling --type prefill_decode --engine baseline --model mock
python scripts/benchmark.py profiling --type scheduling --engine baseline --model mock
```

Types: `throughput` (mode auto: sequential / engine_batch / threaded_clients),
`prefill_decode`, `scheduling` (host wait = wall−cpu, not serving scheduler).

## Results

- Default DB: `data/game_data.db` (override: `DOPPELGAMER_DB_PATH`)
- Publication DB: `data/publication_run.db`
- Export: `python scripts/export_results.py --db data/game_data.db --out results/`
- View: `uvicorn main:app --reload --port 8000` and `streamlit run streamlit_app.py`

## Rules

- Prefer `--model mock` unless the user wants real GPU numbers.
- Publication: baseline + vLLM only, no `--allow-fallback`. See `docs/brev_setup.md`.
- After changing engine or runner code, run a short mock systems benchmark and `pytest -q`.
