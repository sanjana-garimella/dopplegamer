"""Host-side wait measurement (not GPU serving-scheduler overhead).

Measures wall-clock vs process CPU time around `engine.generate`. On GPU runs,
`wall - cpu` is dominated by waiting on the device, not the vLLM/HF batch
scheduler. Prefer `InferenceResult.scheduling_overhead_ms` when the engine
reports it; otherwise report `host_wait_ms` with an explicit metric label.

Publishable wording: "host idle wait (includes GPU wait)", never "scheduler overhead".
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def overhead_pct(total_latency_ms: float, scheduling_overhead_ms: float) -> float:
    """Return a component as a percentage of total latency."""
    if total_latency_ms <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * scheduling_overhead_ms / total_latency_ms))


@dataclass
class CallMeasurement:
    wall_ms: float
    cpu_ms: float
    host_wait_ms: float  # wall - cpu (GPU wait + OS idle), not serving scheduler
    host_wait_pct: float
    engine_reported_overhead_ms: float | None = None

    # Backward-compatible aliases
    @property
    def overhead_ms(self) -> float:
        return self.host_wait_ms

    @property
    def overhead_pct(self) -> float:
        return self.host_wait_pct


def measure_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, CallMeasurement]:
    """Time a single callable, splitting wall-clock vs CPU time."""
    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    result = fn(*args, **kwargs)

    t_wall_end = time.perf_counter()
    t_cpu_end = time.process_time()

    wall_ms = (t_wall_end - t_wall_start) * 1_000
    cpu_ms = (t_cpu_end - t_cpu_start) * 1_000
    host_wait_ms = max(0.0, wall_ms - cpu_ms)
    engine_overhead = getattr(result, "scheduling_overhead_ms", None)
    if engine_overhead is not None:
        try:
            engine_overhead = float(engine_overhead)
        except (TypeError, ValueError):
            engine_overhead = None

    return result, CallMeasurement(
        wall_ms=wall_ms,
        cpu_ms=cpu_ms,
        host_wait_ms=host_wait_ms,
        host_wait_pct=overhead_pct(wall_ms, host_wait_ms),
        engine_reported_overhead_ms=engine_overhead,
    )


@dataclass
class SchedulingReport:
    """Host-wait profile. Field names kept for API stability."""

    engine_name: str
    n_calls: int
    metric: str = "host_wait_ms"  # or engine_scheduling_overhead_ms
    wall_ms: list[float] = field(default_factory=list)
    cpu_ms: list[float] = field(default_factory=list)
    sched_ms: list[float] = field(default_factory=list)

    mean_wall_ms: float = 0.0
    p50_wall_ms: float = 0.0
    p95_wall_ms: float = 0.0
    p99_wall_ms: float = 0.0
    mean_sched_ms: float = 0.0
    mean_sched_pct: float = 0.0
    throughput_qps: float = 0.0

    def _pct(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def compute(self) -> None:
        if not self.wall_ms:
            return
        self.mean_wall_ms = statistics.mean(self.wall_ms)
        self.p50_wall_ms = self._pct(self.wall_ms, 50)
        self.p95_wall_ms = self._pct(self.wall_ms, 95)
        self.p99_wall_ms = self._pct(self.wall_ms, 99)
        self.mean_sched_ms = statistics.mean(self.sched_ms) if self.sched_ms else 0.0
        self.mean_sched_pct = overhead_pct(self.mean_wall_ms, self.mean_sched_ms)
        total_s = sum(self.wall_ms) / 1_000
        self.throughput_qps = self.n_calls / total_s if total_s > 0 else 0.0

    def summary(self) -> str:
        return (
            f"[{self.engine_name}] metric={self.metric} n={self.n_calls} | "
            f"mean={self.mean_wall_ms:.2f}ms p50={self.p50_wall_ms:.2f}ms "
            f"p95={self.p95_wall_ms:.2f}ms p99={self.p99_wall_ms:.2f}ms | "
            f"host_wait={self.mean_sched_ms:.2f}ms ({self.mean_sched_pct:.1f}%) | "
            f"throughput={self.throughput_qps:.1f} QPS"
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "metric": self.metric,
            "n_calls": float(self.n_calls),
            "mean_wall_ms": self.mean_wall_ms,
            "p50_wall_ms": self.p50_wall_ms,
            "p95_wall_ms": self.p95_wall_ms,
            "p99_wall_ms": self.p99_wall_ms,
            "mean_host_wait_ms": self.mean_sched_ms,
            "mean_host_wait_pct": self.mean_sched_pct,
            "throughput_qps": self.throughput_qps,
        }


class SchedulingProfiler:
    """Measure host idle wait around generate (not serving-scheduler time)."""

    def __init__(self, warmup_runs: int = 3, measure_runs: int = 30) -> None:
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs

    def profile(
        self,
        fn: Callable[..., Any],
        *args: Any,
        engine_name: str = "unknown",
        **kwargs: Any,
    ) -> SchedulingReport:
        for _ in range(self.warmup_runs):
            try:
                fn(*args, **kwargs)
            except Exception:
                pass

        report = SchedulingReport(engine_name=engine_name, n_calls=self.measure_runs)
        host_waits: list[float] = []
        engine_overheads: list[float] = []

        for _ in range(self.measure_runs):
            _, m = measure_call(fn, *args, **kwargs)
            report.wall_ms.append(m.wall_ms)
            report.cpu_ms.append(m.cpu_ms)
            host_waits.append(m.host_wait_ms)
            if m.engine_reported_overhead_ms is not None:
                engine_overheads.append(m.engine_reported_overhead_ms)

        if len(engine_overheads) == self.measure_runs:
            report.metric = "engine_scheduling_overhead_ms"
            report.sched_ms = engine_overheads
        else:
            report.metric = "host_wait_ms"
            report.sched_ms = host_waits
        report.compute()
        return report


def benchmark_engines(
    engines: dict[str, Any],
    prompt: str = "You are playing RPS+. Choose a move.",
    max_new_tokens: int = 4,
    warmup_runs: int = 3,
    measure_runs: int = 20,
) -> dict[str, SchedulingReport]:
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
