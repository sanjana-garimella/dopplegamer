# Running Doppelgamer on NVIDIA Brev

Practical steps for a GPU instance on NVIDIA Brev: install, run the publication
protocol (baseline vs vLLM), optional training, and cost ballparks.

Laptop/CI uses `--model mock` (no GPU). Use Brev for real HuggingFace and vLLM
numbers, or for SFT on a 1B-class model.

**Before you start:** the instance must have the *current* repo (publication
script, fail-loud engines, host-wait fix, `prompt_seed`, `requirements-gpu.txt`).
Push your latest commits to GitHub and clone/pull them, or `rsync`/`scp` your
local tree. A clone of an older `main` will miss publication-readiness fixes.

## Contents

- [What you actually need a GPU for](#what-you-actually-need-a-gpu-for)
- [Picking an instance](#picking-an-instance)
- [First-time setup](#first-time-setup)
- [Publication run (recommended)](#publication-run-recommended)
- [Ad-hoc benchmarks](#ad-hoc-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Preble / InferCept](#preble--infercept)
- [Training the checkpoints](#training-the-checkpoints)
- [Estimated cost](#estimated-cost)
- [Keeping the bill down](#keeping-the-bill-down)

## What you actually need a GPU for

| Task | GPU needed | Notes |
|------|-----------|-------|
| Mock engine benchmarks | No | `--model mock` on CPU, used by CI |
| HuggingFace baseline on a real model | Helpful | Small models run on CPU but slowly |
| vLLM engine | Yes | vLLM needs CUDA to initialize |
| SFT LoRA fine-tuning | Yes | `agents/sft/train.py`, 1B fits on 24 GB |
| RL / BC+RL training | No (GPU optional) | MLP policy, mostly CPU |
| Impostor training (LSTM/N-gram) | No | Small model, fine on CPU |

Workloads use short generations (`max_new_tokens` around 8). Cost is dominated by
model download/load and instance idle time, not decode length.

## Picking an instance

| Model size | Suggested GPU | VRAM |
|------------|---------------|------|
| distilgpt2, GPT-2, Llama-3.2-1B | L4 or A10G | 24 GB |
| Llama-3.2-3B | A10G or A100 40 GB | 24-40 GB |
| 7B-8B | A100 40/80 GB | 40-80 GB |

Sweet spot for the paper path: **one L4 or A10G, disk ≥ 50 GB**.

## First-time setup

1. Create a GPU instance (Ubuntu + CUDA). Check the driver:

   ```bash
   nvidia-smi
   ```

2. Python **3.12+**, clone **current** code, venv:

   ```bash
   git clone https://github.com/sanjana-garimella/dopplegamer.git
   cd dopplegamer
   # or: git checkout <your-branch> after push
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   ```

   Most Brev/Ubuntu base images only have `python3` (no `python` alias). Once the
   venv is activated `python` resolves correctly inside it; outside the venv use
   `python3` explicitly, or `sudo apt install python-is-python3`.

3. GPU deps (pulls CUDA torch via vLLM):

   ```bash
   pip install -r requirements-gpu.txt
   ```

   - `requirements-gpu.txt` = `requirements.txt` + `vllm`, `accelerate`, `bitsandbytes`.
   - Gated models (Llama): `huggingface-cli login`.

4. Env file:

   ```bash
   cp .env.example .env
   ```

5. Smoke-test (CPU mock, no GPU burn):

   ```bash
   pytest -q
   python scripts/benchmark.py systems --engines baseline vllm --model mock --rounds 20
   ```

Do **not** pass `--allow-fallback` for publication runs. Real models must fail loud
if vLLM/CUDA is missing.

## Publication run (recommended)

One command: systems benchmark (game-driven prompts), host-wait / prefill-decode /
throughput profiles, metadata, CSV export.

```bash
python scripts/run_publication_benchmark.py \
  --model distilgpt2 \
  --rounds 50 \
  --seed 0 \
  --out results/publication
```

Stronger main result (needs HF access for Llama). Repeat with `--seed 1` and
`--seed 2` into separate `--out` / `--db` paths if you want multi-seed tables:

```bash
python scripts/run_publication_benchmark.py \
  --model meta-llama/Llama-3.2-1B \
  --rounds 50 \
  --seed 0 \
  --out results/publication_1b \
  --db data/publication_1b.db
```

The script fails loud if host-wait profiling does not use `host_wait_ms` (wall −
CPU) for local engines. Do not pass `--allow-fallback`.

Copy **off the instance before delete**:

- `results/publication/` (metadata.json, CSVs, `host_wait.json`, throughput JSON)
- `data/publication_run.db` (or whatever `--db` you set)

```bash
# example from your laptop
scp -r <brev-host>:~/dopplegamer/results/publication ./results/
scp <brev-host>:~/dopplegamer/data/publication_run.db ./data/
```

Then stop/delete the Brev instance.

Methodology to report: **library-mode** baseline vs vLLM; throughput mode is
`engine_batch` for vLLM and `sequential` for HF; host-wait is **wall − CPU**
(includes GPU wait), not serving-scheduler time; vLLM batch per-request latency
may be estimated (`latency_estimated` in extras); prefix hits are a client-side
heuristic unless you document otherwise.

## Ad-hoc benchmarks

```bash
python scripts/benchmark.py systems \
  --engines baseline vllm \
  --model distilgpt2 \
  --rounds 50

python scripts/benchmark.py profiling --type throughput --engine vllm --model distilgpt2
python scripts/benchmark.py profiling --type prefill_decode --engine vllm --model distilgpt2
python scripts/benchmark.py profiling --type scheduling --engine vllm --model distilgpt2

python scripts/export_results.py --db data/game_data.db --out results/
```

Optional local HF ablations (not Preble/InferCept):

```bash
python scripts/benchmark.py systems \
  --engines baseline vllm hf_prefix_cache hf_tool_interrupt \
  --model distilgpt2 \
  --rounds 20
```

## Troubleshooting

**`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist` (from `flashinfer`) when loading vLLM.** vLLM's default
top-k/top-p sampler is FlashInfer, which JIT-compiles a CUDA kernel the first
time `LLM(...)` runs. That needs the full CUDA *toolkit* (`nvcc`), not just the
driver/runtime that `nvidia-smi` reports, and most rented GPU images (Brev
included) only ship the runtime. `serving/vllm_server.py` sets
`VLLM_USE_FLASHINFER_SAMPLER=0` by default to force the PyTorch-native sampler
and skip the JIT compile entirely, so this should not surface with current
code. If you still hit it (e.g. calling `vllm` directly outside this repo),
either export `VLLM_USE_FLASHINFER_SAMPLER=0` yourself or install the CUDA
toolkit (`sudo apt install cuda-toolkit`) so `nvcc` is on `PATH`.

**`sqlite3.OperationalError: no such table: player_profiles`** on a fresh
clone. This was a real bug (`PlayerProfileManager` didn't run `init_db` before
its first query) and is fixed in current code; `data/game_data.db` is
gitignored so every fresh clone starts with no DB file until something
initializes it. If you still see this on an old checkout, `git pull` or run
`python3 -c "from data.schemas import init_db; init_db('data/game_data.db')"`.

## Preble / InferCept

These names are **remote-only**. They need a live OpenAI-compatible server:

```bash
export PREBLE_BASE_URL=http://preble-host:8000
export INFERCEPT_BASE_URL=http://infercept-host:8000
python scripts/benchmark.py systems --engines preble infercept --model your-served-model --rounds 20
```

Without those URLs, loading `preble` / `infercept` errors on purpose. Use
`hf_prefix_cache` / `hf_tool_interrupt` for local ablations and label them as such.

## Training the checkpoints

```bash
python -m agents.sft.train \
  --model meta-llama/Llama-3.2-1B \
  --data data/game_data.db \
  --output checkpoints/sft_best \
  --epochs 3 --lora-rank 16

# Optional: --quantize-4bit on 24 GB for larger bases

# LegalActionWrapper clamps illegal RPS+ moves (e.g. POWER at low energy)
python -m agents.rl.train --timesteps 100000 --output checkpoints/ppo_best
python scripts/train_real_checkpoints.py
```

Copy `checkpoints/` and the DB off the box before teardown.

## Estimated cost

On-demand ballparks (confirm in the Brev console):

| GPU | Typical | Good for |
|-----|---------|----------|
| L4 (24 GB) | ~$0.70-1.10 / hr | publication protocol, 1B |
| A10G (24 GB) | ~$1.00-1.50 / hr | 1B-3B, SFT |
| A100 80 GB | ~$2.00-3.50 / hr | 7B |

At ~$1.20/hr (L4/A10G):

| Activity | Time | Cost |
|----------|------|------|
| Setup + downloads | 20-40 min | $0.40-0.80 |
| Publication protocol, 50 rounds, small model | 30-60 min | $0.60-1.20 |
| **Minimum paper systems pass** | **~1-1.5 hr** | **~$1-2** |
| + SFT 1B | +30-90 min | +$0.60-1.80 |
| Full sitting | 2-4 hr | **~$3-6** |

## Keeping the bill down

- Delete or stop the instance as soon as files are copied; enable auto-stop if available.
- Debug with `--model mock` locally; only final configs on GPU.
- Prefer distilgpt2 then 1B; skip live Preble/InferCept unless you already run those clusters.
- One session for install + run + copy; do not leave GPUs idle overnight.
