import pytest
from analysis.scheduling_overhead import SchedulingProfiler
from analysis.throughput_benchmark import ThroughputBenchmark
from analysis.prefill_decode_split import PrefillDecodeProfiler
from inference.setup_inference_engines import setup_inference_engines, EngineConfig

def test_scheduling_profiler():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg)
    engine = engines["baseline"]
    
    profiler = SchedulingProfiler(warmup_runs=1, measure_runs=2)
    report = profiler.profile(engine.generate, prompt="test", engine_name="mock_baseline")
    
    assert report.engine_name == "mock_baseline"
    assert report.n_calls == 2
    assert len(report.wall_ms) == 2
    assert report.mean_wall_ms > 0

def test_throughput_benchmark():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg)
    engine = engines["baseline"]
    
    bench = ThroughputBenchmark(concurrency=2, total_requests=4)
    report = bench.run(engine, prompt="test", engine_name="mock_baseline")
    
    assert report.engine_name == "mock_baseline"
    assert report.concurrency == 2
    assert len(report.results) == 4
    assert report.throughput_qps > 0
    assert report.mode in {"sequential", "engine_batch", "threaded_clients"}


def test_throughput_hf_stays_sequential_when_forced():
    from analysis.throughput_benchmark import resolve_throughput_mode

    class _HFLike:
        supports_concurrent_clients = False
        supports_engine_batch = False

    assert resolve_throughput_mode(_HFLike()) == "sequential"


def test_scheduling_report_labels_host_wait():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg, engines=["baseline"])
    profiler = SchedulingProfiler(warmup_runs=0, measure_runs=2)
    report = profiler.profile(engines["baseline"].generate, "x", max_new_tokens=2, engine_name="b")
    assert report.metric in {"host_wait_ms", "engine_scheduling_overhead_ms"}
    assert "host_wait" in report.summary()

def test_prefill_decode_profiler():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg)
    engine = engines["baseline"]
    
    profiler = PrefillDecodeProfiler(warmup_runs=1, measure_runs=2)
    results = profiler.profile_engine(engine, prompts=["test"], max_new_tokens=4)
    
    assert len(results) == 2
    assert results[0].ttft_ms > 0
    assert results[0].tpot_ms > 0
