"""Throughput measurement with explicit modes.

Modes:
  - sequential: one request at a time (publishable single-stream QPS)
  - engine_batch: engine.generate_batch (vLLM continuous batching)
  - threaded_clients: only if engine.supports_concurrent_clients (remote servers)

HF baseline never uses threaded clients (not thread-safe).
"""

from __future__ import annotations

import queue
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

ThroughputMode = Literal["sequential", "engine_batch", "threaded_clients"]


def decisions_per_second(n_decisions: int, elapsed_seconds: float) -> float:
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
    mode: ThroughputMode = "sequential"
    results: list[RequestResult] = field(default_factory=list)

    elapsed_s: float = 0.0
    throughput_qps: float = 0.0
    success_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def _pct(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    def compute(self, elapsed_s: float) -> None:
        self.elapsed_s = elapsed_s
        successes = [r for r in self.results if r.success]
        self.success_rate = len(successes) / max(1, len(self.results))
        self.throughput_qps = len(successes) / max(1e-6, elapsed_s)
        lats = [r.latency_ms for r in successes]
        if lats:
            self.mean_latency_ms = statistics.mean(lats)
            self.p50_latency_ms = self._pct(lats, 50)
            self.p95_latency_ms = self._pct(lats, 95)
            self.p99_latency_ms = self._pct(lats, 99)
            self.max_latency_ms = max(lats)

    def summary(self) -> str:
        return (
            f"[{self.engine_name}] mode={self.mode} c={self.concurrency} "
            f"n={self.total_requests} elapsed={self.elapsed_s:.2f}s | "
            f"throughput={self.throughput_qps:.1f} QPS "
            f"success={self.success_rate:.1%} | "
            f"mean={self.mean_latency_ms:.2f}ms "
            f"p50={self.p50_latency_ms:.2f}ms "
            f"p95={self.p95_latency_ms:.2f}ms"
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
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


def resolve_throughput_mode(engine: Any, requested: ThroughputMode | None = None) -> ThroughputMode:
    if requested == "sequential":
        return "sequential"
    if requested == "engine_batch":
        if callable(getattr(engine, "generate_batch", None)):
            return "engine_batch"
        raise ValueError("engine does not implement generate_batch")
    if requested == "threaded_clients":
        if getattr(engine, "supports_concurrent_clients", False):
            return "threaded_clients"
        raise ValueError(
            "engine does not support concurrent clients; use sequential or engine_batch"
        )
    # Auto: prefer true engine batching (vLLM), then remote threaded clients.
    if getattr(engine, "supports_engine_batch", False):
        return "engine_batch"
    if getattr(engine, "supports_concurrent_clients", False):
        return "threaded_clients"
    return "sequential"


class ThroughputBenchmark:
    def __init__(
        self,
        concurrency: int = 1,
        total_requests: int = 100,
        request_timeout_s: float = 60.0,
        mode: ThroughputMode | None = None,
    ) -> None:
        self.concurrency = max(1, concurrency)
        self.total_requests = total_requests
        self.request_timeout_s = request_timeout_s
        self.mode = mode

    def run(
        self,
        engine_or_fn: Any,
        prompt: str = "You are playing RPS+. Choose a move.",
        max_new_tokens: int = 4,
        engine_name: str = "unknown",
    ) -> ThroughputReport:
        mode = self.mode
        if mode is None:
            if callable(getattr(engine_or_fn, "generate", None)):
                mode = resolve_throughput_mode(engine_or_fn)
                if mode == "engine_batch" and self.concurrency <= 1:
                    mode = "sequential"
            else:
                mode = "sequential"

        report = ThroughputReport(
            engine_name=engine_name,
            concurrency=self.concurrency,
            total_requests=self.total_requests,
            mode=mode,
        )

        if mode == "engine_batch":
            self._run_engine_batch(engine_or_fn, prompt, max_new_tokens, report)
        elif mode == "threaded_clients":
            self._run_threaded(engine_or_fn, prompt, max_new_tokens, report)
        else:
            self._run_sequential(engine_or_fn, prompt, max_new_tokens, report)
        return report

    def _run_sequential(self, engine: Any, prompt: str, max_new_tokens: int, report: ThroughputReport) -> None:
        t_start = time.perf_counter()
        for i in range(self.total_requests):
            t0 = time.perf_counter()
            try:
                if callable(getattr(engine, "generate", None)):
                    engine.generate(prompt, max_new_tokens=max_new_tokens)
                else:
                    engine(prompt, max_new_tokens)
                report.results.append(RequestResult(i, (time.perf_counter() - t0) * 1000.0, True))
            except Exception as exc:
                report.results.append(
                    RequestResult(i, (time.perf_counter() - t0) * 1000.0, False, str(exc))
                )
        report.compute(time.perf_counter() - t_start)

    def _run_engine_batch(self, engine: Any, prompt: str, max_new_tokens: int, report: ThroughputReport) -> None:
        batch_size = self.concurrency
        remaining = self.total_requests
        req_id = 0
        t_start = time.perf_counter()
        while remaining > 0:
            n = min(batch_size, remaining)
            prompts = [prompt] * n
            t0 = time.perf_counter()
            try:
                results = engine.generate_batch(prompts, max_new_tokens=max_new_tokens)
                wall = (time.perf_counter() - t0) * 1000.0
                per = wall / max(1, len(results))
                for _ in results:
                    report.results.append(RequestResult(req_id, per, True))
                    req_id += 1
            except Exception as exc:
                wall = (time.perf_counter() - t0) * 1000.0
                for _ in range(n):
                    report.results.append(RequestResult(req_id, wall / n, False, str(exc)))
                    req_id += 1
            remaining -= n
        report.compute(time.perf_counter() - t_start)

    def _run_threaded(self, engine: Any, prompt: str, max_new_tokens: int, report: ThroughputReport) -> None:
        if callable(getattr(engine, "generate", None)):
            def fn(p: str, n: int) -> Any:
                return engine.generate(p, max_new_tokens=n)
        else:
            fn = engine

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
                    result_q.put(
                        RequestResult(req_id, (time.perf_counter() - t0) * 1_000, False, str(exc))
                    )

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


def concurrency_sweep(
    engine: Any,
    prompt: str = "You are playing RPS+. Choose a move.",
    max_new_tokens: int = 4,
    levels: list[int] | None = None,
    requests_per_level: int = 50,
    engine_name: str = "unknown",
) -> list[ThroughputReport]:
    """Sweep load levels using the best safe mode for the engine."""
    mode = resolve_throughput_mode(engine)
    if mode == "sequential":
        levels = [1]
    elif levels is None:
        levels = [1, 2, 4, 8]
    return [
        ThroughputBenchmark(concurrency=c, total_requests=requests_per_level, mode=mode).run(
            engine,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            engine_name=f"{engine_name}@c{c}",
        )
        for c in levels
    ]
