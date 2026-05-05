"""Common interface for inference engines so the benchmark runner can swap them."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class InferenceResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    kv_cache_mb: float = 0.0
    scheduling_overhead_ms: float = 0.0
    # SGLang-style prefix cache metrics (RadixAttention, NeurIPS 2024)
    prefix_cache_hit_tokens: int = 0   # tokens served from cached prefix
    prefix_cache_miss_tokens: int = 0  # tokens that required fresh computation
    extra: dict = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from KV prefix cache (0–1).

        Higher = more prefix reuse across multi-turn agent traces.
        Ref: SGLang RadixAttention (Zheng et al., NeurIPS 2024).
        """
        total = self.prefix_cache_hit_tokens + self.prefix_cache_miss_tokens
        return self.prefix_cache_hit_tokens / total if total > 0 else 0.0


class InferenceEngine(ABC):
    name: str = "engine"

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        ...

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


class _Timer:
    def __enter__(self) -> "_Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
