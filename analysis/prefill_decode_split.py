"""Prefill vs decode latency splitting and profiling.

LLM inference has two distinct phases:
  - Prefill:  Model reads the entire prompt in parallel → builds the KV cache.
              Bottleneck: GPU compute throughput.  Measured as TTFT.
  - Decode:   Model generates output tokens one at a time, each depending on
              the previous.  Bottleneck: GPU memory bandwidth.  Measured as TPOT.

This module provides:
  - LatencyBreakdown: structured result for a single inference call
  - PrefillDecodeProfiler: wraps any InferenceEngine and collects per-call splits
  - aggregate_splits: compute summary statistics across many calls
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from analysis.stats_utils import percentile


# ---------------------------------------------------------------------------
# Core breakdown dataclass
# ---------------------------------------------------------------------------

@dataclass
class LatencyBreakdown:
    """Per-call prefill/decode split measured for one inference request."""
    ttft_ms: float        # Time To First Token  — prefill latency
    tpot_ms: float        # Time Per Output Token — decode latency per token
    output_tokens: int    # Number of tokens generated in decode phase
    prompt_tokens: int = 0
    engine_name: str   = "unknown"
    quantization: str  = "fp16"

    @property
    def decode_ms(self) -> float:
        """Total decode phase duration."""
        return max(0.0, self.tpot_ms * max(0, self.output_tokens - 1))

    @property
    def total_ms(self) -> float:
        """End-to-end request latency."""
        return self.ttft_ms + self.decode_ms

    @property
    def prefill_fraction(self) -> float:
        """Fraction of total latency spent in prefill."""
        t = self.total_ms
        return self.ttft_ms / t if t > 0 else 0.0

    @property
    def decode_fraction(self) -> float:
        """Fraction of total latency spent in decode."""
        t = self.total_ms
        return self.decode_ms / t if t > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name":       self.engine_name,
            "quantization":      self.quantization,
            "prompt_tokens":     self.prompt_tokens,
            "output_tokens":     self.output_tokens,
            "ttft_ms":           self.ttft_ms,
            "tpot_ms":           self.tpot_ms,
            "decode_ms":         self.decode_ms,
            "total_ms":          self.total_ms,
            "prefill_fraction":  self.prefill_fraction,
            "decode_fraction":   self.decode_fraction,
        }


# ---------------------------------------------------------------------------
# Per-engine profiler
# ---------------------------------------------------------------------------

@dataclass
class SplitSummary:
    """Aggregate statistics over many LatencyBreakdown samples."""
    engine_name: str
    n: int
    mean_ttft_ms: float     = 0.0
    p95_ttft_ms: float      = 0.0
    mean_tpot_ms: float     = 0.0
    p95_tpot_ms: float      = 0.0
    mean_total_ms: float    = 0.0
    p95_total_ms: float     = 0.0
    mean_prefill_pct: float = 0.0
    mean_decode_pct: float  = 0.0

    def summary(self) -> str:
        return (
            f"[{self.engine_name}] n={self.n} | "
            f"TTFT mean={self.mean_ttft_ms:.2f}ms p95={self.p95_ttft_ms:.2f}ms | "
            f"TPOT mean={self.mean_tpot_ms:.2f}ms p95={self.p95_tpot_ms:.2f}ms | "
            f"total mean={self.mean_total_ms:.2f}ms p95={self.p95_total_ms:.2f}ms | "
            f"prefill={self.mean_prefill_pct:.1f}% decode={self.mean_decode_pct:.1f}%"
        )

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "engine_name": self.engine_name,
            "n": self.n,
            "mean_ttft_ms": self.mean_ttft_ms,
            "p95_ttft_ms": self.p95_ttft_ms,
            "mean_tpot_ms": self.mean_tpot_ms,
            "p95_tpot_ms": self.p95_tpot_ms,
            "mean_total_ms": self.mean_total_ms,
            "p95_total_ms": self.p95_total_ms,
            "mean_prefill_pct": self.mean_prefill_pct,
            "mean_decode_pct": self.mean_decode_pct,
        }


def aggregate_splits(breakdowns: list[LatencyBreakdown]) -> SplitSummary:
    """Compute aggregate statistics from a list of LatencyBreakdown samples."""
    if not breakdowns:
        return SplitSummary(engine_name="unknown", n=0)

    name    = breakdowns[0].engine_name
    ttfts   = [b.ttft_ms        for b in breakdowns]
    tpots   = [b.tpot_ms        for b in breakdowns]
    totals  = [b.total_ms       for b in breakdowns]
    pre_frc = [b.prefill_fraction * 100 for b in breakdowns]
    dec_frc = [b.decode_fraction  * 100 for b in breakdowns]

    return SplitSummary(
        engine_name=name,
        n=len(breakdowns),
        mean_ttft_ms=statistics.mean(ttfts),
        p95_ttft_ms=percentile(ttfts, 95),
        mean_tpot_ms=statistics.mean(tpots),
        p95_tpot_ms=percentile(tpots, 95),
        mean_total_ms=statistics.mean(totals),
        p95_total_ms=percentile(totals, 95),
        mean_prefill_pct=statistics.mean(pre_frc),
        mean_decode_pct=statistics.mean(dec_frc),
    )


# ---------------------------------------------------------------------------
# Profiler that wraps an InferenceEngine
# ---------------------------------------------------------------------------

class PrefillDecodeProfiler:
    """Profile prefill vs decode latency by calling engine.generate() many times.

    The engine is expected to return an InferenceResult with ttft_ms and
    tpot_ms populated. Missing TTFT is treated as all wall time in prefill;
    failed calls are skipped (no invented 70/30 splits).
    """

    def __init__(self, warmup_runs: int = 3, measure_runs: int = 20) -> None:
        self.warmup_runs  = warmup_runs
        self.measure_runs = measure_runs

    def profile_engine(
        self,
        engine: Any,
        prompts: list[str],
        max_new_tokens: int = 8,
        quantization: str = "fp16",
    ) -> list[LatencyBreakdown]:
        """Run the engine on each prompt (cycling if needed) and collect splits.

        Args:
            engine:        An InferenceEngine instance.
            prompts:       List of prompt strings (cycled to reach measure_runs).
            max_new_tokens: Tokens to generate per call.
            quantization:  Label for the result (e.g. "int4", "fp16").

        Returns:
            List of LatencyBreakdown, one per measurement call.
        """
        engine_name = getattr(engine, "name", "unknown")
        n_prompts   = len(prompts) if prompts else 1
        if not prompts:
            prompts = ["You are playing RPS+. Choose a move."]

        # Warmup
        for i in range(self.warmup_runs):
            try:
                engine.generate(prompts[i % n_prompts], max_new_tokens=max_new_tokens)
            except Exception as warmup_exc:
                print(f"warmup failed for {engine_name}: {warmup_exc}")

        results: list[LatencyBreakdown] = []
        for i in range(self.measure_runs):
            prompt = prompts[i % n_prompts]
            t0 = time.perf_counter()
            try:
                result = engine.generate(prompt, max_new_tokens=max_new_tokens)
                wall_ms = (time.perf_counter() - t0) * 1_000

                # Extract TTFT/TPOT from result or estimate.
                ttft_ms = getattr(result, "ttft_ms", None)
                tpot_ms = getattr(result, "tpot_ms", None)
                n_out   = getattr(result, "output_tokens", max_new_tokens)
                n_in    = getattr(result, "prompt_tokens", len(prompt.split()))

                total_ms = getattr(result, "total_latency_ms", None) or wall_ms
                if ttft_ms is None or ttft_ms <= 0:
                    # Honest fallback: attribute all wall time to TTFT when split unknown.
                    ttft_ms = float(total_ms)
                if tpot_ms is None or tpot_ms < 0:
                    decode_ms = max(0.0, float(total_ms) - float(ttft_ms))
                    tpot_ms = decode_ms / max(1, n_out - 1) if n_out > 1 else 0.0

                results.append(LatencyBreakdown(
                    ttft_ms=float(ttft_ms),
                    tpot_ms=float(tpot_ms),
                    output_tokens=n_out,
                    prompt_tokens=n_in,
                    engine_name=engine_name,
                    quantization=quantization,
                ))
            except Exception:
                # Skip failed calls rather than inventing a 70/30 split.
                continue

        return results


# ---------------------------------------------------------------------------
# Convenience: compare multiple engines
# ---------------------------------------------------------------------------

def compare_engines(
    engines: dict[str, Any],
    prompts: list[str] | None = None,
    max_new_tokens: int = 8,
    warmup_runs: int = 3,
    measure_runs: int = 20,
) -> dict[str, SplitSummary]:
    """Profile and compare prefill/decode splits across multiple engines.

    Returns a dict of {engine_name: SplitSummary}.
    """
    if prompts is None:
        prompts = [
            "You are playing RPS+ turn 1. Predict next move.",
            "You are playing RPS+ turn 20 with long history. Predict next move.",
            "Tool result indicates opponent favors ROCK. Choose best response.",
        ]
    profiler = PrefillDecodeProfiler(warmup_runs=warmup_runs, measure_runs=measure_runs)
    summaries: dict[str, SplitSummary] = {}
    for name, engine in engines.items():
        breakdowns = profiler.profile_engine(engine, prompts, max_new_tokens=max_new_tokens)
        # Override engine_name with the dict key for consistent labelling
        for b in breakdowns:
            b.engine_name = name
        summaries[name] = aggregate_splits(breakdowns)
        summaries[name].engine_name = name
    return summaries
