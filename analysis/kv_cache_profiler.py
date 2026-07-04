"""KV cache growth: formula estimates and engine-reported measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class KVSample:
    turn: int
    tokens: int
    kv_cache_mb: float
    source: str = "formula"  # formula | engine


def estimate_kv_cache_mb(
    *,
    tokens: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_value: int = 2,
) -> float:
    raw = 2 * n_layers * n_kv_heads * head_dim * tokens * bytes_per_value
    return raw / (1024 * 1024)


def track_kv_cache_growth(
    *,
    turns: int,
    tokens_per_turn: int,
    n_layers: int = 24,
    n_kv_heads: int = 8,
    head_dim: int = 128,
) -> list[KVSample]:
    """Synthetic growth curve (formula only). Prefer profile_engine_kv for papers."""
    samples: list[KVSample] = []
    for turn in range(1, turns + 1):
        tokens = turn * tokens_per_turn
        samples.append(
            KVSample(
                turn=turn,
                tokens=tokens,
                kv_cache_mb=estimate_kv_cache_mb(
                    tokens=tokens,
                    n_layers=n_layers,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                ),
                source="formula",
            )
        )
    return samples


def profile_engine_kv(
    engine: Any,
    prompts: list[str],
    *,
    max_new_tokens: int = 4,
) -> list[KVSample]:
    """Measure KV from engine-reported `kv_cache_mb` on growing prompts."""
    samples: list[KVSample] = []
    for turn, prompt in enumerate(prompts, start=1):
        result = engine.generate(prompt, max_new_tokens=max_new_tokens)
        tokens = int(result.prompt_tokens) + int(result.output_tokens)
        samples.append(
            KVSample(
                turn=turn,
                tokens=tokens,
                kv_cache_mb=float(result.kv_cache_mb),
                source="engine",
            )
        )
    return samples
