"""KV cache growth estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVSample:
    turn: int
    tokens: int
    kv_cache_mb: float


def estimate_kv_cache_mb(
    *,
    tokens: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_value: int = 2,
) -> float:
    # K and V tensors for every token.
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
            )
        )
    return samples
