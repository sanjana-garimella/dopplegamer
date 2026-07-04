# Benchmark results

Reference numbers from the publication protocol, for sanity-checking new runs.
They are a methodology validation, not a systems claim.

## distilgpt2, single seed, NVIDIA L40S

One `--model distilgpt2 --rounds 50 --seed 0` run in library mode (the engine's
Python API called in-process, not behind an HTTP server). Stack: vLLM 0.24.0,
transformers 5.13.0, torch 2.11.0+cu130.

### Latency

| Engine | Mean TTFT | Mean TPOT | Mean total latency | p95 total latency |
|--------|-----------|-----------|---------------------|---------------------|
| baseline (HF) | 3.63 ms | 2.86 ms | 19.79 ms | 21.40 ms |
| vllm | 2.06 ms | 2.06 ms | 12.35 ms | 12.80 ms |

### Throughput sweep

`engine_batch` for vLLM, `sequential` for baseline.

| Engine | Concurrency | Throughput | p50 latency |
|--------|-------------|------------|--------------|
| baseline | 1 | 78.9 QPS | 12.60 ms |
| vllm | 1 | 114.7 QPS | 8.57 ms |
| vllm | 2 | 180.1 QPS | 4.61 ms |
| vllm | 4 | 316.5 QPS | 3.13 ms |
| vllm | 8 | 593.5 QPS | 1.68 ms |

### Host wait

Host-wait profiling (wall minus CPU around `generate`, not serving-scheduler
time) shows vLLM spending approximately 84% of wall time waiting on the GPU
(`mean_host_wait_pct`), versus near-zero for the HF baseline at this batch size.
This is consistent with vLLM's asynchronous scheduling overlapping host-side work
that the baseline performs synchronously.

`distilgpt2` (82M params) validates that the protocol runs end to end. The
Llama-3.2-1B run is the intended main result.
