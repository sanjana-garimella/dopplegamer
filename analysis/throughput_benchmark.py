"""Concurrent agent load testing for throughput benchmarking.

Measures decisions-per-second under realistic concurrent load using
a thread pool to simulate N simultaneous agent clients.

Usage:
    from analysis.throughput_benchmark import ThroughputBenchmark
    bench = ThroughputBenchmark(concurrency=4, total_requests=100)
    report = bench.run(engine, prompt="Decide move for RPS+")
    print(report.summary())
"""

from __future__ import annotations

import queue
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def decisions_per_second(n_decisions: int, elapsed_seconds: float) -> float:
    """Compute throughput rate."""
    if elapsed_seconds <= 0:
        return 0.0
    return n_decisions / elapsed_seconds


@dataclass
class RequestResult:
    request_id: int
    latency_ms: float
    success: bool
    error: str = ""


@dataclass
class ThroughputReport:
    engine_name: str
    concurrency: int
    total_requests: int
    results: list[RequestResult] = field(default_factory=list)

    elapsed_s: float       = 0.0
    throughput_qps: float  = 0.0
    success_rate: float    = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float  = 0.0
    p95_latency_ms: float  = 0.0
    p99_latency_ms: float  = 0.0
    max_latency_ms: float  = 0.0

    def _pct(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    def compute(self, elapsed_s: float) -> None:
        self.elapsed_s    = elapsed_s
        successes         = [r for r in self.results if r.success]
        self.success_rate = len(successes) / max(1, len(self.results))
        self.throughput_qps = len(successes) / max(1e-6, elapsed_s)
        lats = [r.latency_ms for r in successes]
        if lats:
            self.mean_latency_ms = statistics.mean(lats)
            self.p50_latency_ms  = self._pct(lats, 50)
            self.p95_latency_ms  = self._pct(lats, 95)
            self.p99_latency_ms  = self._pct(lats, 99)
            self.max_latency_ms  = max(lats)

    def summary(self) -> str:
        return (
            f"[{self.engine_name}] c={self.concurrency} n={self.total_requests} "
            f"elapsed={self.elapsed_s:.2f}s | "
            f"throughput={self.throughput_qps:.1f} QPS "
            f"success={self.success_rate:.1%} | "
            f"mean={self.mean_latency_ms:.2f}ms "
            f"p95={self.p95_latency_ms:.2f}ms "
            f"p99={self.p99_latency_ms:.2f}ms"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "concurrency": float(self.concurrency),
            "total_requests": float(self.total_requests),
            "elapsed_s": self.elapsed_s,
            "throughput_qps": self.throughput_qps,
            "success_rate": self.success_rate,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "max_latency_ms": self.max_latency_ms,
        }


class ThroughputBenchmark:
    """Fire concurrent requests to measure sustained throughput."""

    def __init__(
        self,
        concurrency: int = 4,
        total_requests: int = 100,
        request_timeout_s: float = 60.0,
    ) -> None:
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.request_timeout_s = request_timeout_s

    def run(
        self,
        engine_or_fn: Any,
        prompt: str = "You are playing RPS+. Choose a move.",
        max_new_tokens: int = 4,
        engine_name: str = "unknown",
    ) -> ThroughputReport:
        """Run the concurrent load test against an InferenceEngine or callable."""
        if callable(getattr(engine_or_fn, "generate", None)):
            def fn(p: str, n: int) -> Any:
                return engine_or_fn.generate(p, max_new_tokens=n)
        else:
            fn = engine_or_fn  # type: ignore[assignment]

        report = ThroughputReport(
            engine_name=engine_name,
            concurrency=self.concurrency,
            total_requests=self.total_requests,
        )
        result_q: queue.Queue[RequestResult] = queue.Queue()
        id_q: queue.Queue[int] = queue.Queue()
        for i in range(self.total_requests):
            id_q.put(i)

        def worker() -> None:
            while True:
                try:
                    req_id = id_q.get_nowait()
                except queue.Empty:
                    return
                t0 = time.perf_counter()
                try:
                    fn(prompt, max_new_tokens)
                    result_q.put(RequestResult(req_id, (time.perf_counter() - t0) * 1_000, True))
                except Exception as exc:
                    result_q.put(RequestResult(req_id, (time.perf_counter() - t0) * 1_000, False, str(exc)))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.concurrency)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.request_timeout_s)
        elapsed_s = time.perf_counter() - t_start

        while not result_q.empty():
            report.results.append(result_q.get_nowait())

        report.compute(elapsed_s)
        return report


def concurrency_sweep(
    engine: Any,
    prompt: str = "You are playing RPS+. Choose a move.",
    max_new_tokens: int = 4,
    levels: list[int] | None = None,
    requests_per_level: int = 50,
    engine_name: str = "unknown",
) -> list[ThroughputReport]:
    """Sweep concurrency levels to find the throughput saturation point."""
    if levels is None:
        levels = [1, 2, 4, 8]
    return [
        ThroughputBenchmark(concurrency=c, total_requests=requests_per_level).run(
            engine, prompt=prompt, max_new_tokens=max_new_tokens,
            engine_name=f"{engine_name}@c{c}"
        )
        for c in levels
    ]
