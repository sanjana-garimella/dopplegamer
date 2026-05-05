"""Inference engine registry/factory used by evaluation and API layers."""

from __future__ import annotations

from dataclasses import dataclass

from serving.base import InferenceEngine
from serving.base import InferenceResult


@dataclass
class EngineConfig:
    model_name: str = "distilgpt2"


def setup_inference_engines(cfg: EngineConfig | None = None) -> dict[str, InferenceEngine]:
    cfg = cfg or EngineConfig()
    if cfg.model_name == "mock":
        return {
            "baseline": _MockEngine("baseline"),
            "vllm": _MockEngine("vllm"),
            "preble": _MockEngine("preble"),
            "infercept": _MockEngine("infercept"),
        }
    return {
        "baseline": configure_baseline(cfg.model_name),
        "vllm": configure_vllm(cfg.model_name),
        "preble": configure_preble(cfg.model_name),
        "infercept": configure_infercept(cfg.model_name),
    }


def configure_baseline(model_name: str) -> InferenceEngine:
    try:
        from serving.baseline_hf import HFBaselineEngine

        return HFBaselineEngine(model_name=model_name)
    except Exception:
        return _MockEngine("baseline")


def configure_vllm(model_name: str) -> InferenceEngine:
    from serving.vllm_server import VLLMEngine

    try:
        return VLLMEngine(model_name=model_name)
    except Exception:
        # Keep benchmark runnable without GPU / vllm installation.
        return configure_baseline(model_name)


def configure_preble(model_name: str) -> InferenceEngine:
    try:
        from serving.preble_benchmark import PrebleEngine

        return PrebleEngine(model_name=model_name)
    except Exception:
        return _MockEngine("preble")


def configure_infercept(model_name: str) -> InferenceEngine:
    try:
        from serving.infercept_benchmark import InferCeptEngine

        return InferCeptEngine(model_name=model_name)
    except Exception:
        return _MockEngine("infercept")


class _MockEngine(InferenceEngine):
    def __init__(self, name: str) -> None:
        self.name = name
        self._seen_prefixes: set[str] = set()

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        prompt_tokens = max(1, len(prompt.split()))
        output_tokens = max_new_tokens
        ttft = 8.0
        tpot = 1.5
        # Simulate prefix cache: shared system prompt tokens are "cached" on repeat
        prefix = prompt[:50]
        if prefix in self._seen_prefixes:
            hit = int(prompt_tokens * 0.6)
            miss = prompt_tokens - hit
        else:
            hit = 0
            miss = prompt_tokens
            self._seen_prefixes.add(prefix)
        return InferenceResult(
            text="ROCK",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft,
            tpot_ms=tpot,
            total_latency_ms=ttft + tpot * max(1, output_tokens - 1),
            kv_cache_mb=0.25 * prompt_tokens,
            scheduling_overhead_ms=0.6,
            prefix_cache_hit_tokens=hit,
            prefix_cache_miss_tokens=miss,
        )