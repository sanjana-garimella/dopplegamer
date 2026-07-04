---
name: add-inference-engine
description: >-
  Add a new LLM serving backend. Use when implementing an inference engine under
  serving/, wiring setup_inference_engines, or extending InferenceResult metrics.
---

# Add an inference engine

## Checklist

```
- [ ] Subclass serving.base.InferenceEngine
- [ ] Implement generate(prompt, max_new_tokens) -> InferenceResult
- [ ] Fill shared metric fields (ttft_ms, tpot_ms, kv_cache_mb, prefix cache)
- [ ] Add configure_<name> in inference/setup_inference_engines.py (lazy engines list)
- [ ] Raise EngineLoadError when deps/GPU missing unless allow_fallback
- [ ] Wrap fallbacks with _FallbackEngine (requested→actual labeling)
- [ ] Set supports_concurrent_clients / supports_engine_batch when applicable
- [ ] Include in mock catalog when model_name == "mock"
- [ ] Verify: scripts/benchmark.py systems --engines <name> --model mock
- [ ] Optional real: --model distilgpt2 on GPU
- [ ] pytest -q (tests/test_serving.py patterns)
```

## Contract

```python
class MyEngine(InferenceEngine):
    name = "my_engine"

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        ...
```

Required `InferenceResult` fields: `text`, `prompt_tokens`, `output_tokens`,
`ttft_ms`, `tpot_ms`, `total_latency_ms`. Optional: `kv_cache_mb`,
`scheduling_overhead_ms`, `prefix_cache_hit_tokens`, `prefix_cache_miss_tokens`,
`extra`.

## Factory pattern

```python
def configure_my_engine(cfg: EngineConfig) -> InferenceEngine:
    try:
        from serving.my_engine import MyEngine
        return MyEngine(model_name=cfg.model_name)
    except Exception as exc:
        if cfg.allow_fallback:
            return _MockEngine("my_engine")
        raise EngineLoadError(...) from exc
```

When `cfg.model_name == "mock"`, return only `_MockEngine` instances (no heavy imports).

## Rules

- Do not invent a parallel metrics type; extend `InferenceResult` if needed.
- Prefer library-mode timing for benchmarks (see `serving/vllm_server.py`).
- Keep GPU-only imports inside `__init__` / `generate`, not at module import time if avoidable.
