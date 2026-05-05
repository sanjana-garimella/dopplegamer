import pytest
from serving.base import InferenceResult
from inference.setup_inference_engines import setup_inference_engines, EngineConfig

def test_inference_engines_setup():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg)
    
    assert "baseline" in engines
    assert "vllm" in engines
    assert "preble" in engines
    assert "infercept" in engines

def test_mock_engine_generate():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg)
    engine = engines["baseline"]
    
    result = engine.generate("test prompt")
    assert isinstance(result, InferenceResult)
    assert result.text == "ROCK"
    assert result.prompt_tokens > 0
    assert result.output_tokens > 0
    assert result.ttft_ms > 0
    assert result.tpot_ms > 0
    assert result.total_latency_ms > 0
