"""Tests for publication-readiness fixes (host-wait, seeds, clamps, API)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analysis.scheduling_overhead import SchedulingProfiler, measure_call
from analysis.stats_utils import percentile
from analysis.throughput_benchmark import resolve_throughput_mode
from data.features import filter_aggregated_agent_results
from data.schemas import connect, init_db
from environments.legal_action_wrapper import LegalActionWrapper
from environments.rps_plus import Move, RPSPlusEnv
from evaluation.runner import run_benchmark
from inference.setup_inference_engines import (
    EngineConfig,
    EngineLoadError,
    setup_inference_engines,
)
from serving.base import InferenceResult


def test_percentile_p50_of_two_is_midpoint():
    assert percentile([10.0, 20.0], 50) == pytest.approx(15.0)


def test_measure_call_ignores_default_none_overhead():
    def fn():
        return InferenceResult(
            text="x",
            prompt_tokens=1,
            output_tokens=1,
            ttft_ms=1.0,
            tpot_ms=0.0,
            total_latency_ms=1.0,
            scheduling_overhead_ms=None,
        )

    _, m = measure_call(fn)
    assert m.engine_reported_overhead_ms is None
    assert m.host_wait_ms >= 0.0


def test_scheduling_profiler_uses_host_wait_for_mock_without_overhead():
    # Force mock generate to report None overhead via a thin wrapper.
    class _NoOverhead:
        name = "x"
        supports_concurrent_clients = False
        supports_engine_batch = False

        def generate(self, prompt, max_new_tokens=8):
            return InferenceResult(
                text="ROCK",
                prompt_tokens=2,
                output_tokens=max_new_tokens,
                ttft_ms=1.0,
                tpot_ms=1.0,
                total_latency_ms=1.0 + 1.0 * max(0, max_new_tokens - 1),
                scheduling_overhead_ms=None,
            )

    profiler = SchedulingProfiler(warmup_runs=0, measure_runs=3)
    report = profiler.profile(_NoOverhead().generate, "hi", max_new_tokens=2, engine_name="x")
    assert report.metric == "host_wait_ms"


def test_scheduling_profiler_uses_engine_overhead_when_set():
    class _WithOverhead:
        def generate(self, prompt, max_new_tokens=8):
            return InferenceResult(
                text="ROCK",
                prompt_tokens=1,
                output_tokens=1,
                ttft_ms=1.0,
                tpot_ms=0.0,
                total_latency_ms=1.0,
                scheduling_overhead_ms=2.5,
            )

    profiler = SchedulingProfiler(warmup_runs=0, measure_runs=2)
    report = profiler.profile(_WithOverhead().generate, "hi", engine_name="y")
    assert report.metric == "engine_scheduling_overhead_ms"
    assert report.mean_sched_ms == pytest.approx(2.5)


def test_mock_engine_does_not_claim_batch_or_threads():
    engines = setup_inference_engines(EngineConfig(model_name="mock"), engines=["baseline"])
    eng = engines["baseline"]
    assert eng.supports_concurrent_clients is False
    assert eng.supports_engine_batch is False
    assert resolve_throughput_mode(eng) == "sequential"


def test_legal_action_wrapper_clamps_power_at_zero_energy():
    env = LegalActionWrapper(RPSPlusEnv(max_turns=5, seed=1, starting_energy=0))
    env.reset(seed=1)
    # POWER is illegal at energy 0; wrapper must not raise.
    obs, reward, term, trunc, info = env.step(int(Move.POWER))
    assert info["action"] != int(Move.POWER)
    assert info["action"] in info["legal_moves"]


def test_prompt_seed_is_returned():
    db = Path("data") / "_tmp_prompt_seed.db"
    if db.exists():
        db.unlink()
    result = run_benchmark(
        rounds=2,
        engines=["baseline"],
        agents=[],
        db_path=db,
        model_name="mock",
        prompt_seed=7,
    )
    assert result["prompt_seed"] == 7
    db.unlink(missing_ok=True)


def test_fallback_engine_row_name_in_db(tmp_path: Path):
    db = tmp_path / "fb.db"
    # allow_fallback with a nonsense model yields mock under requested name.
    result = run_benchmark(
        rounds=1,
        engines=["vllm"],
        agents=[],
        db_path=db,
        model_name="not-a-real-model-xyz",
        allow_fallback=True,
    )
    conn = connect(db)
    try:
        engines = [r[0] for r in conn.execute("SELECT DISTINCT engine FROM inference_benchmarks")]
    finally:
        conn.close()
    assert any("→" in e or e == "vllm" for e in engines)
    assert result["run_id"]


def test_filter_aggregated_agent_results():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"run_id": "abc", "agent_name": "random::RPS+_seed0", "win_rate": 0.1},
            {"run_id": "abc_agg", "agent_name": "random::RPS+", "win_rate": 0.2},
        ]
    )
    out = filter_aggregated_agent_results(df)
    assert len(out) == 1
    assert out.iloc[0]["agent_name"] == "random::RPS+"


def test_fidelity_reference_heuristic_when_no_sft(tmp_path: Path):
    db = tmp_path / "fid.db"
    result = run_benchmark(
        rounds=2,
        engines=[],
        agents=["random", "heuristic"],
        db_path=db,
        model_name="mock",
        n_seeds=1,
    )
    assert result["fidelity_reference"] == "heuristic"
    assert all(row["fidelity_reference"] == "heuristic" for row in result["game_breakdown"])


def test_api_requires_key_for_real_models(monkeypatch):
    monkeypatch.delenv("DOPPELGAMER_API_KEY", raising=False)
    from fastapi.testclient import TestClient
    import main as main_mod

    # Reload app state: _API_KEY is module-level at import.
    monkeypatch.setattr(main_mod, "_API_KEY", "")
    client = TestClient(main_mod.app)
    resp = client.post(
        "/benchmark",
        json={"rounds": 1, "engines": [], "agents": [], "model_name": "distilgpt2"},
    )
    assert resp.status_code == 403


def test_api_rejects_non_allowlisted_model(monkeypatch):
    monkeypatch.setattr("main._API_KEY", "secret")
    from fastapi.testclient import TestClient
    import main as main_mod

    client = TestClient(main_mod.app)
    resp = client.post(
        "/benchmark",
        json={"rounds": 1, "engines": [], "agents": [], "model_name": "meta-llama/x"},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 422


def test_api_rejects_bad_db_path(monkeypatch):
    monkeypatch.setattr("main._API_KEY", "")
    from fastapi.testclient import TestClient
    import main as main_mod

    client = TestClient(main_mod.app)
    resp = client.post(
        "/benchmark",
        json={"rounds": 1, "engines": [], "agents": [], "model_name": "mock", "db_path": "../x.db"},
    )
    assert resp.status_code == 422
