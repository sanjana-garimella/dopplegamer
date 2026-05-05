"""CPU scheduling overhead measurement.

Replicates the WukLab scheduling overhead study methodology:
  - Runs N inference calls through a callable
  - Uses perf_counter (wall clock) vs process_time (CPU time) to isolate
    time spent in OS scheduling, Python overhead, and batching logic
  - Reports: mean/p50/p95/p99 latency, mean overhead %, throughput

Usage:
    from analysis.scheduling_overhead import SchedulingProfiler, overhead_pct

    profiler = SchedulingProfiler(warmup_runs=5, measure_runs=50)
    report = profiler.profile(my_engine.generate, prompt="hello")
    print(report.summary())
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Simple per-call overhead calculation (kept for backward compat)
# ---------------------------------------------------------------------------

def overhead_pct(total_latency_ms: float, scheduling_overhead_ms: float) -> float:
    """Return scheduling overhead as a percentage of total latency."""
    if total_latency_ms <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * scheduling_overhead_ms / total_latency_ms))


# ---------------------------------------------------------------------------
# Per-call measurement
# ---------------------------------------------------------------------------

@dataclass
class CallMeasurement:
    wall_ms: float        # perf_counter elapsed (wall clock)
    cpu_ms: float         # process_time elapsed (CPU only)
    overhead_ms: float    # wall - cpu: OS scheduling + Python overhead
    overhead_pct: float   # overhead_ms / wall_ms * 100


def measure_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, CallMeasurement]:
    """Time a single callable, splitting wall-clock vs CPU time."""
    # Snapshot both clocks before the call.
    t_wall_start = time.perf_counter()
    t_cpu_start  = time.process_time()

    result = fn(*args, **kwargs)

    t_wall_end = time.perf_counter()
    t_cpu_end  = time.process_time()

    wall_ms     = (t_wall_end - t_wall_start) * 1_000
    cpu_ms      = (t_cpu_end  - t_cpu_start)  * 1_000
    # Clamp to zero: process_time can occasionally exceed perf_counter on VMs.
    sched_ms    = max(0.0, wall_ms - cpu_ms)
    sched_pct   = overhead_pct(wall_ms, sched_ms)

    return result, CallMeasurement(
        wall_ms=wall_ms,
        cpu_ms=cpu_ms,
        overhead_ms=sched_ms,
        overhead_pct=sched_pct,
    )


# ---------------------------------------------------------------------------
# Profiling report
# ---------------------------------------------------------------------------

@dataclass
class SchedulingReport:
    engine_name: str
    n_calls: int
    wall_ms: list[float]  = field(default_factory=list)
    cpu_ms: list[float]   = field(default_factory=list)
    sched_ms: list[float] = field(default_factory=list)

    # Derived statistics (populated by SchedulingProfiler.profile)
    mean_wall_ms: float    = 0.0
    p50_wall_ms: float     = 0.0
    p95_wall_ms: float     = 0.0
    p99_wall_ms: float     = 0.0
    mean_sched_ms: float   = 0.0
    mean_sched_pct: float  = 0.0
    throughput_qps: float  = 0.0   # queries per second

    def _pct(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def compute(self) -> None:
        """Populate all derived statistics from raw sample lists."""
        if not self.wall_ms:
            return
        self.mean_wall_ms   = statistics.mean(self.wall_ms)
        self.p50_wall_ms    = self._pct(self.wall_ms, 50)
        self.p95_wall_ms    = self._pct(self.wall_ms, 95)
        self.p99_wall_ms    = self._pct(self.wall_ms, 99)
        self.mean_sched_ms  = statistics.mean(self.sched_ms)
        self.mean_sched_pct = overhead_pct(self.mean_wall_ms, self.mean_sched_ms)
        total_s = sum(self.wall_ms) / 1_000
        self.throughput_qps = self.n_calls / total_s if total_s > 0 else 0.0

    def summary(self) -> str:
        return (
            f"[{self.engine_name}] n={self.n_calls} | "
            f"mean={self.mean_wall_ms:.2f}ms p50={self.p50_wall_ms:.2f}ms "
            f"p95={self.p95_wall_ms:.2f}ms p99={self.p99_wall_ms:.2f}ms | "
            f"sched={self.mean_sched_ms:.2f}ms ({self.mean_sched_pct:.1f}%) | "
            f"throughput={self.throughput_qps:.1f} QPS"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "n_calls":        float(self.n_calls),
            "mean_wall_ms":   self.mean_wall_ms,
            "p50_wall_ms":    self.p50_wall_ms,
            "p95_wall_ms":    self.p95_wall_ms,
            "p99_wall_ms":    self.p99_wall_ms,
            "mean_sched_ms":  self.mean_sched_ms,
            "mean_sched_pct": self.mean_sched_pct,
            "throughput_qps": self.throughput_qps,
        }


# ---------------------------------------------------------------------------
# High-level profiler
# ---------------------------------------------------------------------------

class SchedulingProfiler:
    """Measure scheduling overhead for any callable (e.g. engine.generate)."""

    def __init__(self, warmup_runs: int = 3, measure_runs: int = 30) -> None:
        self.warmup_runs  = warmup_runs
        self.measure_runs = measure_runs

    def profile(
        self,
        fn: Callable[..., Any],
        *args: Any,
        engine_name: str = "unknown",
        **kwargs: Any,
    ) -> SchedulingReport:
        """Run warmup then measurement passes, return a populated SchedulingReport."""
        # Warmup — discard results
        for _ in range(self.warmup_runs):
            try:
                fn(*args, **kwargs)
            except Exception:
                pass

        report = SchedulingReport(engine_name=engine_name, n_calls=self.measure_runs)

        for _ in range(self.measure_runs):
            _, m = measure_call(fn, *args, **kwargs)
            report.wall_ms.append(m.wall_ms)
            report.cpu_ms.append(m.cpu_ms)
            report.sched_ms.append(m.overhead_ms)

        report.compute()
        return report


# ---------------------------------------------------------------------------
# Convenience: benchmark multiple engines in one shot
# ---------------------------------------------------------------------------

def benchmark_engines(
    engines: dict[str, Any],
    prompt: str = "You are playing RPS+. Choose a move.",
    max_new_tokens: int = 4,
    warmup_runs: int = 3,
    measure_runs: int = 20,
) -> dict[str, SchedulingReport]:
    """Profile scheduling overhead across multiple InferenceEngine instances.

    Args:
        engines:       Mapping of {name: engine} where engine has .generate().
        prompt:        Prompt string to use for all calls.
        max_new_tokens: Tokens to generate per call.
        warmup_runs:   Number of throwaway warmup calls per engine.
        measure_runs:  Number of timed measurement calls per engine.

    Returns:
        Mapping of {name: SchedulingReport}.
    """
    profiler = SchedulingProfiler(warmup_runs=warmup_runs, measure_runs=measure_runs)
    reports: dict[str, SchedulingReport] = {}
    for name, engine in engines.items():
        reports[name] = profiler.profile(
            engine.generate,
            prompt,
            max_new_tokens=max_new_tokens,
            engine_name=name,
        )
    return reports
