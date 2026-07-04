import pytest

from serving.base import InferenceResult
from inference.setup_inference_engines import (
    EngineConfig,
    EngineLoadError,
    _FallbackEngine,
    setup_inference_engines,
    validate_model_name,
)


def test_inference_engines_setup():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg)

    assert "baseline" in engines
    assert "vllm" in engines
    assert "hf_prefix_cache" in engines
    assert "hf_tool_interrupt" in engines
    assert "preble" in engines
    assert "infercept" in engines


def test_lazy_engine_construction_only_requested():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg, engines=["baseline"])
    assert list(engines.keys()) == ["baseline"]
    assert "vllm" not in engines


def test_mock_engine_generate():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg, engines=["baseline"])
    engine = engines["baseline"]

    result = engine.generate("test prompt")
    assert isinstance(result, InferenceResult)
    assert result.text == "ROCK"
    assert result.prompt_tokens > 0
    assert result.output_tokens > 0
    assert result.ttft_ms > 0
    assert result.tpot_ms > 0
    assert result.total_latency_ms > 0


def test_mock_metric_identity():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg, engines=["baseline"])
    result = engines["baseline"].generate("shared system prompt for cache", max_new_tokens=8)
    expected = result.ttft_ms + result.tpot_ms * max(1, result.output_tokens - 1)
    assert result.total_latency_ms == pytest.approx(expected)


def test_real_model_fails_loud_without_fallback():
    cfg = EngineConfig(model_name="not-a-real-model-xyz", allow_fallback=False)
    with pytest.raises(EngineLoadError):
        setup_inference_engines(cfg, engines=["baseline"])


def test_real_model_allow_fallback_labels_metrics():
    cfg = EngineConfig(model_name="not-a-real-model-xyz", allow_fallback=True)
    engines = setup_inference_engines(cfg, engines=["vllm"])
    assert "vllm" in engines
    assert isinstance(engines["vllm"], _FallbackEngine)
    result = engines["vllm"].generate("hi", max_new_tokens=2)
    assert result.extra.get("fallback") is True
    assert result.extra.get("requested_engine") == "vllm"
    assert result.extra.get("actual_backend") == "mock"


def test_validate_model_name_allowlist():
    assert validate_model_name("mock") == "mock"
    assert validate_model_name("distilgpt2") == "distilgpt2"
    with pytest.raises(ValueError, match="allowlisted"):
        validate_model_name("meta-llama/Llama-3.2-1B")
    with pytest.raises(ValueError, match="filesystem path"):
        validate_model_name("/etc/passwd")


def test_preble_requires_remote_url_on_real_model():
    cfg = EngineConfig(model_name="distilgpt2", allow_fallback=False)
    with pytest.raises(EngineLoadError, match="PREBLE_BASE_URL"):
        setup_inference_engines(cfg, engines=["preble"])


def test_hf_ablation_name_is_not_preble_alias():
    cfg = EngineConfig(model_name="mock")
    engines = setup_inference_engines(cfg, engines=["hf_prefix_cache", "preble"])
    assert engines["hf_prefix_cache"].name == "hf_prefix_cache"
    assert engines["preble"].name == "preble"
